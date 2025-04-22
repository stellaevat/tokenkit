# Tokenizer Transfer via tokenkit

## Overview

This guide will walk you through the process of transferring a pretrained model to a new tokenizer using tokenkit.

First, follow the installation instructions in the [README](../README.md).

Then, the scripts in `examples/` provide a starting point for transferring a model to a new tokenizer. For example:

```bash
bash examples/llama3_to_qwen2_tokenizer_gpu.sh
# or on TPU: examples/llama3_to_qwen2_tokenizer_tpu.sh
```

This will distill the [Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model to the [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) tokenizer. Let's have a look at what it runs:

```bash
# examples/llama3_to_qwen2_tokenizer_gpu.sh
NAME=llama3_to_qwen2_tokenizer
python3 scripts/cross_tokenizer_distill.py \
    --config=configs/cross_tokenizer_distill.yaml \
    --overrides \
    losses=[sft,alm_unconstrained] \
    alm_mode=merge_by_space_prob+append_space \
    tokenizer_pair_bias_threshold=0.1 \
    train_model_mode=lora \
    model_lora_rank=64 \
    model_lora_alpha=64 \
    n_data_parallel=1 \
    n_model_parallel=1 \
    steps=5000 \
    eval_interval=1000 \
    save_interval=1000 \
    data.batch_size=64 \
    optimizer.grad_acc_steps=4 \
    data.num_workers=16 \
    student.pretrained_model_name_or_path=benjamin/Llama-3.2-3B-Instruct-flax \
    student.tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3\' \
    target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
    name=$NAME
```

Default arguments are taken from [`configs/cross_tokenizer_distill.yaml`](../configs/cross_tokenizer_distill.yaml). You can keep many of these as-is. Typically, overriding some arguments is enough. Let's go over the overriden arguments in more detail:

```
losses=[sft,alm_unconstrained] \
alm_mode=merge_by_space_prob+append_space \
tokenizer_pair_bias_threshold=0.1 \
```

These arguments configure the losses to optimize and the ALM mode to use. The above configuration should be the best for many cases. Importantly, *it is different to what is described in the [ALM paper](https://arxiv.org/abs/2503.20083)*. In particular, it achieves equivalent or better results without precomputation. A more detailed description is forthcoming in an updated version of our paper.

```
hypernet.architecture=identity \
```

By default, `tokenkit` uses a one-layer embedding projector network (`hypernet.architecture=transformer`). This improves performance but can be memory-intensive, so we disable it here.

```
multitask_aggregation_fn=approx_gradmag_preserve_mag \
```

Use the last-layer gradient magnitudes to approximately reweigh the the multiple objectives (in this case, SFT and ALM) to contribute equally to the final loss gradients. This adds a little extra overhead since we need to backpropagate through the last layer separately for every objective, but it removes the requirement to manually tune loss weights. If you observe that this adds too much overhead, you can skip it and manually tune the loss weights using e.g. `loss_weights=[1.,0.5]`, or leave it out completely to use uniform loss weights (instead of uniform loss *gradient* weights).

```
train_model_mode=lora \
model_lora_rank=64 \
model_lora_alpha=64 \
```

Train the model using LoRA with rank = alpha = 64. `tokenkit` applies LoRA to the QKV projections, attention output projection, as well as the MLP up-, down- and gate projections (see [lora.py](../tokenkit/models/lora.py)).

You can use `train_model_mode=full` to train the full model instead. However, in this case, we need to store a separate copy of the model parameters for the student and the teacher, whereas with LoRA, we can use a single model parameter copy and materialize / dematerialize the LoRA parameters as needed. Storing a separate teacher model copy makes training substantially more memory intensive. A rule of thumb is: for transfer to a similar kind of tokenizer (e.g., another subword tokenizer), LoRA is sufficient. For transfer to a very different tokenizer (e.g., a byte-level tokenizer), full-finetuning helps.

```
n_data_parallel=1 \
n_model_parallel=1 \
```

Data and model parallelism. Set this such that the product of the two is the number of GPUs or TPU cores you have available. Often (especially for larger models) you will want to increase model parallelism and keep data parallelism at 1.

```
steps=5000 \
eval_interval=1000 \
save_interval=1000 \
data.batch_size=64 \
optimizer.grad_acc_steps=4 \
data.num_workers=16 \
```

Train for 5000 steps, evaluate every 1000 steps, save the model every 1000 steps at a global batch size of 64 with 4 gradient accumulation steps (i.e., a local batch size of 16). Evaluation is done via (a fork of) [`lm-evaluation-harness`](https://github.com/bminixhofer/lm-evaluation-harness) and runs the tasks configured via `eval.tasks`.

```
student.pretrained_model_name_or_path=benjamin/Llama-3.2-3B-Instruct-flax \
student.tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3\' \
target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
```

The (local or HF hub) paths to the model to transfer. If we do not specify a separate teacher, the teacher will be the student with the original tokenizer (this is what we want for tokenizer transfer). Notably:

- The model is `benjamin/Llama-3.2-3B-Instruct-flax` since the original `meta-llama/Llama-3.2-3B-Instruct` model is not in Flax format. You can convert supported models to Flax using the `scripts/push_flax_version_to_hub.py` script.
- The tokenizer is specified using a tokenizer spec which differs from the HuggingFace `AutoTokenizer` format by including additional colon-separated tags. For example: `Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3` specifies the Qwen2.5-1B-Instruct tokenizer initially stemming from the Qwen2 model family (`source=Qwen2`) updated to use the special tokens of the Llama3 family instead (`target=Llama3`). See the [byteification](./byteification.md) guide for more details on the interface tokenkit provides to use HuggingFace tokenizers. For our purposes in this guide, it is important that when you transfer across tokenizers, you can choose to either (i) preserve the original special tokens (safer but potentially inconvenient) or (ii) use the special tokens from the new tokenizer (less safe but potentially more convenient). More on this below in [To Keep or to Change Special Tokens?](#to-keep-or-to-change-special-tokens).

```
name=$NAME
```

The name to track the experiment with. By default, `tokenkit` uses [Weights & Biases](https://www.wandb.ai/) to track experiments. 

This is all, you can now transfer your first model!

## Transfer to Bytes

We need to make a couple of adjustments to enable effective transfer to byte-level tokenizers. Let's compare the example config in [`examples/llama3_to_byte_tokenizer_gpu.sh`](../examples/llama3_to_byte_tokenizer_gpu.sh) to the config we used above:

```diff
- losses=[sft,alm_unconstrained] \
+ losses=[sft,alm_unconstrained,alm_latents] \
```

For transfer to bytes, the ALM latent (hidden-state alignment) objective substantially improves performance.

```diff
- train_model_mode=lora \
- model_lora_rank=64 \
- model_lora_alpha=64 \
+ train_model_mode=full \
+ expand_input_ids=true \
+ output_embeddings_mode=untie \
```

We train the full model to give it more capacity to adapt to the fundamental change in tokenization. We also *expand* the input IDs to inject some extra parameters while preserving total FLOPs. What input ID expansion does is: for every byte embedding, add the subword embedding of the longest matching subword ending at this byte position (where the subwords and subword embeddings are taken from the original tokenizer and embedding matrix). Finally, we untie the byte input and output embeddings, since there is no reason to tie them in the byte-level case (we don't save any considerable amount of parameters). This may also marginally improve performance.

```diff
- target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
+ target_tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3:conversion=byte\'
```

The target tokenizer is now specified using a tokenizer spec which includes the conversion to bytes. See the [byteification](./byteification.md#tokenizer-spec) guide for details on this spec.

## Exporting the Model

`tokenkit` uses a custom internal format to checkpoint model fine-tuning parameter diffs. To export a checkpoint to the HuggingFace format, run e.g.:

```bash
# --with_pt exports the model in PyTorch format (in addition to Flax)
python3 scripts/export_checkpoint.py \
    --checkpoint_path=outputs/cross_tokenizer_distill/step_5000 \
    --output=checkpoints/llama3_to_qwen2_tokenizer_hf \
    --with_pt
```

If you are exporting a model which has been trained with input ID expansion, you need to also specify which embeddings and tokenizer to use for expansion, e.g.:

```bash
python3 scripts/export_checkpoint.py \
    --checkpoint_path=outputs/cross_tokenizer_distill/step_5000 \
    --output=checkpoints/llama3_to_bytes \
    --with_pt \
    --expand_input_ids_model=benjamin/Llama-3.2-3B-Instruct-flax \
    --expand_input_ids_tokenizer=meta-llama/Llama-3.2-3B-Instruct:source=Llama3
```

Afterwards, you can load the model as usual using HuggingFace transformers:

```python
from tranformers import AutoModelForCausalLM
from tokenkit.byteify import load_byteify_tokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/llama3_to_bytes", trust_remote_code=True)
tokenizer = load_byteify_tokenizer("meta-llama/Llama-3.2-3B-Instruct:source=Llama3:conversion=byte")

tokens = tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": "Hello, how are you?"}], return_tensors="pt")
output = model.generate(tokens)
print(tokenizer.decode(output[0]))
```

## To Keep or to Change the Special Tokens?

In the above example where we transferred to the Qwen2 tokenizer, by using the target tokenizer spec `Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3`, we transferred to a tokenizer using all the *regular* tokens from the Qwen2 tokenizer, but keeping the special tokens (and the chat template) from the Llama3 tokenizer. We can instead transfer to a tokenizer which is completely equivalent to the Qwen2 tokenizer (regular and special tokens) by specifying it as `Qwen/Qwen2.5-1.5B:source=Qwen2:target=Qwen2`. What to choose depends on your use case:

- *Keeping the special tokens:* This is the safer choice, since the model will not have to learn to use a new chat template format with new special tokens. If you just want to, for example, transfer to a new tokenizer which encodes some domain more efficiently, this is the better choice.
- *Changing the special tokens:* If you are using tokenizer transfer to combine (e.g., ensemble) multiple models, this is more convenient since we don't need to worry about aligning the different special tokens and chat templates to each other (which is quite easy to do, but still inconvenient). However, there's some things to be careful about: for example, transferring Gemma2 to the Llama3 chat template is quite easy since both use similar formats and both use a \<bos\> token. However, transferring Gemma2 to the Qwen2 chat template is not as straightforward *since Gemma2 uses a \<bos\> token, but Qwen2 doesn't*. The model thus has to learn to re-distribute the original attention sink behavior of the \<bos\> token across other tokens. This may or may not work well, depending on the training budget, dataset and so on.

---
<h3 align="center">Next: <a href="./byteification.md">Byteification: A Unified Interface to Tokenizers</a></h3>