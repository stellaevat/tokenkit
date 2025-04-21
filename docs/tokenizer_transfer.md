# Tokenizer Transfer via tokenkit

## Overview

This guide will walk you through the process of transferring a pretrained model to a new tokenizer using tokenkit.

First, follow the installation instructions in the [README](../README.md).

Then, the scripts in `examples/` provide a starting point for transferring a model to a new tokenizer. For example:

```bash
bash examples/llama3_to_qwen2_tokenizer_gpu.sh
# or on TPU: examples/llama3_to_qwen2_tokenizer_tpu.sh
```

This will run training to transfer the [Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model to the [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) tokenizer. Let's have a look at what it runs:

```bash
# examples/llama3_to_qwen2_tokenizer_gpu.sh
NAME=llama3_to_qwen2_tokenizer
python3 scripts/cross_tokenizer_distill.py \
    --config=configs/cross_tokenizer_distill.yaml \
    --overrides \
    losses=[sft,alm_unconstrained] \
    alm_mode=merge_by_space_prob+append_space \
    tokenizer_pair_bias_threshold=0.1 \
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

Default arguments are taken from `configs/cross_tokenizer_distill.yaml`. You can keep many of these as-is. Typically, overriding some arguments is enough. Let's go over the overriden arguments in more detail:

```
losses=[sft,alm_unconstrained] \
alm_mode=merge_by_space_prob+append_space \
tokenizer_pair_bias_threshold=0.1 \
```

These arguments configure the losses to optimize and the ALM mode to use. The above configuration should be the best for many cases. Importantly, *it is different to what is described in the [ALM paper](https://arxiv.org/abs/2503.20083)*. In particular, it achieves equivalent or better results without precomputation. A more detailed description is forthcoming in an updated version of our paper.

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

## Transfer to Bytes

## To Keep or to Change Special Tokens?