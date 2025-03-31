"""
Example Usage:

NAME=gemma_to_qwen_alm
python cross_tokenizer_distill.py \
    losses=[distill_main_path_pair-nobias] \
    distill_main_path_diff_fn=binary_ce \
    max_teacher_length=512 \
    max_student_length=512 \
    n_data_parallel=4 \
    n_model_parallel=4 \
    steps=20000 \
    eval_interval=10000 \
    save_interval=20000 \
    optimizer.learning_rate=1e-5 \
    optimizer.weight_decay=0.0 \
    optimizer.max_grad_norm=null \
    eval.tasks=[arc_easy,arc_challenge,piqa,hellaswag,boolq,arithmetic,mmlu] \
    eval.lengths=[128,256,512,1024,2048] \
    eval.tokens_per_batch=8192 \
    eval.add_bos=true \
    data.batch_size=64 \
    ppl_eval_data.batch_size=16 \
    log_interval=10 \
    sync_interval=100 \
    data=tulu3 \
    use_chat_template=true \
    student.pretrained_model_name_or_path="benjamin/gemma-2-2b-it-flax" \
    student.tokenizer_name=\'google/gemma-2-2b-it:source=Gemma2:target=Qwen2\' \
    tokenizer_pair_data_path=outputs/tokenizer_data/old_source=Gemma2_target=Qwen2 \
    tokenizer_pair_bias_threshold=1e-4 \
    train_model_mode=lora \
    model_lora_rank=64 \
    model_lora_alpha=64 \
    export_to_gcs_bucket=trc-transfer-autockpt \
    num_workers=16 \
    name=$NAME
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any

import datasets
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import wandb
from flax import traverse_util
from flax.training import common_utils, train_state
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, FlaxAutoModelForCausalLM

from tokenkit import data, eval, gcs_utils, utils
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.models import lora, param, sharding
from tokenkit.models.hypernet import Hypernet
from tokenkit.training import checkpoint, collators, losses, lr, opt
from tokenkit.utils import tqdm

logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    logit_mask_teacher: jnp.ndarray
    logit_mask_new: jnp.ndarray
    train_mask: Any
    space_mask_teacher: jnp.ndarray
    space_mask_new: jnp.ndarray
    loss_ema_stats: jnp.ndarray


def _unimplemented_apply_fn(*args, **kwargs):
    raise NotImplementedError("state.apply_fn is not used.")


def get_state(
    model_params,
    original_embeddings,
    new_embeddings,
    teacher_model_params,  # TODO: possible to save memory by sharing student/teacher params if they are the same model and only training LoRA
    teacher_embeddings,
    space_mask_teacher,
    space_mask_new,
    teacher_config,
    student_config,
    args,
    hypernet,
    learning_rate_fn,
    optimizer_kwargs,
    shard_patterns,
):
    dtype = getattr(jnp, args.dtype)

    # pad to multiple
    n_pad_teacher = utils.get_n_pad(
        teacher_embeddings.shape[0], args.pad_to_multiple_of
    )
    n_pad_new = utils.get_n_pad(new_embeddings.shape[0], args.pad_to_multiple_of)

    logit_mask_teacher = np.ones(
        (teacher_embeddings.shape[0] + n_pad_teacher,), dtype=np.float32
    )
    logit_mask_teacher[: teacher_embeddings.shape[0]] = 0.0
    logit_mask_teacher *= utils.get_large_negative_number(
        logit_mask_teacher.dtype, module=np
    )

    logit_mask_new = np.ones((new_embeddings.shape[0] + n_pad_new,), dtype=np.float32)
    logit_mask_new[: new_embeddings.shape[0]] = 0.0
    logit_mask_new *= utils.get_large_negative_number(logit_mask_new.dtype, module=np)

    teacher_embeddings = np.pad(
        teacher_embeddings,
        ((0, n_pad_teacher), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    new_embeddings = np.pad(
        new_embeddings,
        ((0, n_pad_new), (0, 0), (0, 0)),
        mode="edge",
    )
    space_mask_teacher = np.pad(
        space_mask_teacher,
        ((0, n_pad_teacher),),
        mode="constant",
        constant_values=False,
    )
    space_mask_new = np.pad(
        space_mask_new, ((0, n_pad_new),), mode="constant", constant_values=False
    )

    params = {
        "model": model_params,
        "teacher_model": jax.tree.map(lambda x: x.astype(dtype), teacher_model_params),
        "teacher_embeddings": teacher_embeddings,
        "new_embeddings": new_embeddings,
        "loss_weights": jnp.full(
            len(args.losses), fill_value=args.uncertainty_s_init, dtype=jnp.float32
        ),
    }

    if args.add_expanded_input_ids:
        n_pad_original = utils.get_n_pad(
            original_embeddings.shape[0], args.pad_to_multiple_of
        )
        original_embeddings = np.pad(
            original_embeddings,
            ((0, n_pad_original), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        model_params["expanded_input_ids_projection"] = np.zeros(
            (original_embeddings.shape[-1], original_embeddings.shape[-1]), dtype=dtype
        )
        model_params["original_embeddings"] = original_embeddings

    # add latent projectors
    if args.latents_do_project:
        assert (
            args.latents_to_align == "last_hidden_state"
        ), "Latent projectors only implemented for last_hidden_state at the moment"

        model_params["projector_latents"] = utils.init_linear(
            args.seed, new_embeddings.shape[-1], teacher_embeddings.shape[-1], dtype
        )

    # add projectors for dskd baseline
    if "baseline_dskd" in args.losses:
        model_params["projector_t2s"] = utils.init_linear(
            args.seed, teacher_embeddings.shape[-1], new_embeddings.shape[-1], dtype
        )
        model_params["projector_s2t"] = utils.init_linear(
            args.seed, new_embeddings.shape[-1], teacher_embeddings.shape[-1], dtype
        )
        model_params["projector_query"] = utils.init_linear(
            args.seed,
            new_embeddings.shape[-1] * 2,
            teacher_embeddings.shape[-1] * 2,
            dtype,
        )

    if args.train_model_mode == "lora":
        params["model_lora"] = lora.init_lora_params(
            args,
            params["model"],
            model_type=student_config.model_type,
            seed=args.seed,
        )

    def _jit_init_state(params):
        params["hypernet"] = jax.jit(hypernet.init)(
            jax.random.PRNGKey(args.seed),
            params["new_embeddings"][:, None, :, :],
            jnp.ones((params["new_embeddings"].shape[0], 1), dtype=bool),
            jnp.arange(params["new_embeddings"].shape[0], dtype=jnp.int32),
        )

        train_mask = utils.label_by_prefix(
            params,
            [
                [
                    ("hypernet", "non_trainable"),
                    False,
                ],  # pax / praxis convention
                [("hypernet",), True],
                [
                    ("teacher_model",),
                    False,
                ],
                [
                    ("teacher_embeddings",),
                    False,
                ],
                [
                    (
                        "model",
                        "original_embeddings",
                    ),
                    False,
                ],
                [
                    "model.*(projector_query|projector_s2t|projector_t2s|projector_latents).*",
                    True,
                ],
                [
                    ("model", "expanded_input_ids_projection"),
                    True,
                ],
                [
                    ("model",),
                    (True if args.train_model_mode == "full" else False),
                ],
                [
                    ("model_lora",),
                    True,
                ],
                [
                    ("new_embeddings",),
                    True if args.train_embeddings else False,
                ],
                [
                    ("loss_weights",),
                    True,
                ],
            ],
        )

        params = jax.tree.map(
            # TODO: do we need to keep any specific params in fp32, e.g LayerNorm?
            lambda x, trainable: (
                x.astype(jnp.float32) if trainable else x.astype(dtype)
            ),
            params,
            train_mask,
        )

        return TrainState.create(
            apply_fn=_unimplemented_apply_fn,
            params=params,
            logit_mask_teacher=logit_mask_teacher,
            logit_mask_new=logit_mask_new,
            train_mask=train_mask,
            tx=opt.get_optimizer(train_mask, learning_rate_fn, **optimizer_kwargs),
            space_mask_teacher=space_mask_teacher,
            space_mask_new=space_mask_new,
            loss_ema_stats=jnp.full(
                (len(args.losses), 2), fill_value=jnp.nan, dtype=jnp.float32
            ),
        )

    state_shape = jax.eval_shape(_jit_init_state, params)
    state_shardings = sharding.get_sharding_fn(shard_patterns, student_config.mesh)(
        state_shape
    )
    in_shardings = state_shardings.params.copy()
    del in_shardings["hypernet"]  # initialized in jit

    params = sharding.to_global_array(params, in_shardings)
    state = jax.jit(
        _jit_init_state,
        in_shardings=(in_shardings,),
        out_shardings=state_shardings,
        donate_argnums=(0,),
    )(params)
    if not args.debug:
        jax.tree.map(lambda x: x.delete(), params)  # make sure params are deleted

    return state, state_shardings


def cross_entropy(
    logits,
    labels,
    attention_mask,
    logits_already_shifted=False,
    logit_mask=None,
    denom=None,
):
    shift_logits = logits[..., :-1, :] if not logits_already_shifted else logits
    shift_labels = labels[..., 1:]
    shift_attention_mask = attention_mask[..., 1:]

    if logit_mask is not None:
        shift_logits = shift_logits + logit_mask[None, None, :]

    return (
        optax.softmax_cross_entropy(
            shift_logits, common_utils.onehot(shift_labels, shift_logits.shape[-1])
        )
        * shift_attention_mask
    ).mean() / (denom if denom is not None else shift_attention_mask.mean())


def pad_embeddings_with_random(embeddings, tokenizer, seed=1234):
    n_embed_diff = len(tokenizer) - len(embeddings)

    embeddings_mean = embeddings.mean(0)
    embeddings_std = embeddings.std(0)

    return np.concatenate(
        [
            embeddings,
            np.random.RandomState(seed).standard_normal(
                (n_embed_diff, *embeddings.shape[1:]),
            )
            * embeddings_std[None]
            + embeddings_mean[None],
        ]
    )


@hydra.main(
    version_base=None, config_path="../configs", config_name="cross_tokenizer_distill"
)
def my_app(args: DictConfig) -> None:
    logger.info(pformat(OmegaConf.to_object(args)))

    if args.debug:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        mesh = jax.sharding.Mesh([jax.devices("cpu")], ["data", "model"])
    else:
        mesh = sharding.get_mesh(args.n_data_parallel, args.n_model_parallel)

    output_dir = Path(args.output)
    # clear previous output dir
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    OmegaConf.save(config=args, f=output_dir / "args.yaml", resolve=True)

    if hasattr(args, "teacher"):
        teacher_config = AutoConfig.from_pretrained(**args.teacher)
    else:
        teacher_config = AutoConfig.from_pretrained(**args.student)

    student_config = AutoConfig.from_pretrained(**args.student)

    teacher_config.max_length = args.max_teacher_length
    if not args.debug and args.max_teacher_length % 128 == 0:
        teacher_config._attn_implementation = "pallas_flash_attention"
    else:
        logger.warning(
            "Using eager attention implementation for teacher (max length not divisible by 128 or debug)"
        )
        teacher_config._attn_implementation = "eager"

    student_config.max_length = args.max_student_length
    if not args.debug and args.max_student_length % 128 == 0:
        student_config._attn_implementation = "pallas_flash_attention"
    else:
        logger.warning(
            "Using eager attention implementation for student (max length not divisible by 128 or debug)"
        )
        student_config._attn_implementation = "eager"

    teacher_config.mesh = student_config.mesh = mesh

    dtype = getattr(jnp, args.dtype)

    # prepare dataset
    dataset = data.get_dataset(**args.data, seed=args.seed)
    ppl_eval_data = data.get_dataset(**args.ppl_eval_data, seed=args.seed)

    student_model_kwargs = dict(args.student)
    teacher_model_kwargs = (
        dict(args.teacher) if hasattr(args, "teacher") else dict(args.student)
    )
    original_student_tokenizer_name = student_model_kwargs.pop("tokenizer_name")
    teacher_tokenizer_name = teacher_model_kwargs.pop("tokenizer_name")
    target_tokenizer_name = args.target_tokenizer_name

    tokenizer_teacher = load_byteify_tokenizer(teacher_tokenizer_name)
    tokenizer_student_original = load_byteify_tokenizer(original_student_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(target_tokenizer_name)

    if args.tokens_to_add is not None:
        logger.info("Adding tokens:", args.tokens_to_add)
        target_tokenizer.add_tokens(args.tokens_to_add)

    if "baseline_mined" in args.losses or (
        any(loss.startswith("distill_side_path") for loss in args.losses)
        and args.side_path_mapping_mode == "mined"
    ):
        mined_mapping = np.load(
            Path(args.tokenizer_pair_data_path) / "mined_mapping.npy"
        )
        mined_distances = json.load(
            open(Path(args.tokenizer_pair_data_path) / "mined_distances.json")
        )
        logger.info("Average MinED distance:", np.mean(list(mined_distances.values())))
    else:
        mined_mapping = mined_distances = None

    if any(loss.startswith("distill_side_path") for loss in args.losses):
        student_mapping, teacher_mapping = utils.get_side_path_mappings(
            teacher_tokenizer=tokenizer_teacher,
            student_tokenizer=target_tokenizer,
            mode=args.side_path_mapping_mode,
            tokenizer_pair_data_path=args.tokenizer_pair_data_path,
            tokenizer_pair_bias_threshold=(
                args.tokenizer_pair_bias_threshold_side_path
                if args.tokenizer_pair_bias_threshold_side_path is not None
                else args.tokenizer_pair_bias_threshold
            ),
            mined_mapping=mined_mapping,
        )
        logger.info(
            f"Using {len(student_mapping)}/{len(target_tokenizer)} student tokens for side path alignment."
        )
    else:
        student_mapping = teacher_mapping = None

    model_params = param.load_params(**student_model_kwargs)
    if hasattr(args, "teacher") is not None:
        teacher_model_params = param.load_params(**teacher_model_kwargs)
    else:
        teacher_model_params = model_params

    if args.debug:
        model_params = param.strip_layers(model_params, student_config, 1)
        teacher_model_params = param.strip_layers(
            teacher_model_params, teacher_config, 1
        )

    teacher_model = FlaxAutoModelForCausalLM.from_config(
        teacher_config,
        dtype=dtype,
        _do_init=False,
        input_shape=(args.n_data_parallel, args.max_teacher_length),
    )
    new_model = FlaxAutoModelForCausalLM.from_config(
        student_config,
        dtype=dtype,
        _do_init=False,
        input_shape=(args.n_data_parallel, args.max_student_length),
    )
    if args.gradient_checkpointing:
        new_model.enable_gradient_checkpointing()

    n_embd = param.get(
        model_params, param.get_input_embedding_path(student_config.model_type)
    ).shape[-1]

    embeddings, model_params = param.stack_embeddings(
        model_params,
        student_config,
        pop_embeddings=True,
    )
    teacher_embeddings, teacher_model_params = param.stack_embeddings(
        teacher_model_params,
        teacher_config,
        pop_embeddings=True,
    )
    embeddings = embeddings[: len(tokenizer_student_original)]
    teacher_embeddings = teacher_embeddings[: len(tokenizer_teacher)]

    if len(teacher_embeddings) < len(tokenizer_teacher):
        logger.warning(
            "Teacher embeddings are smaller than teacher tokenizer, padding embeddings with random embeddings."
        )
        teacher_embeddings = pad_embeddings_with_random(
            teacher_embeddings, tokenizer_teacher, seed=args.seed
        )

    if len(embeddings) < len(tokenizer_student_original):
        logger.warning(
            "Student embeddings are smaller than student tokenizer, padding embeddings with random embeddings."
        )
        embeddings = pad_embeddings_with_random(
            embeddings, tokenizer_student_original, seed=args.seed
        )

    if args.target_tokenizer == "keep":
        new_embeddings = embeddings
        if len(new_embeddings) < len(target_tokenizer):
            logger.warning(
                "Student embeddings are smaller than target tokenizer, padding embeddings with random embeddings."
            )
            new_embeddings = pad_embeddings_with_random(
                new_embeddings, target_tokenizer, seed=args.seed
            )
        overlapping_embeddings_mask = np.ones(len(target_tokenizer), dtype=bool)
    else:
        diff_embeddings, original_to_new_indices, diff_indices = utils.fvt(
            tokenizer_student_original,
            target_tokenizer,
            embeddings,
        )
        new_embeddings = embeddings[original_to_new_indices]
        if len(diff_indices) > 0:
            new_embeddings[diff_indices] = diff_embeddings

        overlapping_embeddings_mask = np.ones(len(target_tokenizer), dtype=bool)
        overlapping_embeddings_mask[diff_indices] = False
        logger.warning(
            f"{sum(~overlapping_embeddings_mask)} non-overlapping embeddings"
        )

    if args.output_embeddings_mode == "untie" and new_embeddings.shape[1] == 1:
        student_config.tie_word_embeddings = False
        new_embeddings = jnp.tile(new_embeddings, (1, 2, 1))

    space_mask_teacher = utils.get_space_mask(tokenizer_teacher)
    space_mask_new = utils.get_space_mask(target_tokenizer)

    hypernet = Hypernet(
        dtype=dtype,
        hidden_size=n_embd,
        num_embeddings=1 if student_config.tie_word_embeddings else 2,
        max_seq_length=1,
        vocab_size=len(target_tokenizer),  # TODO: implement vocab padding
        **args.hypernet,
    )

    optimizer_kwargs = OmegaConf.to_object(args.optimizer)
    learning_rate_fn = lr.linear_warmup_linear_decay_with_linear_prefix(
        optimizer_kwargs.pop("learning_rate"),
        args.steps,
        args.warmup_steps,
        args.prefix_steps,
        args.prefix_lr,
    )

    shard_patterns = {
        **sharding.get_shard_patterns(teacher_config.model_type),
        **sharding.get_shard_patterns(student_config.model_type),
        **sharding.get_shard_patterns("hypernet"),
    }
    state, state_shardings = get_state(
        model_params=model_params,
        original_embeddings=embeddings,
        new_embeddings=new_embeddings,
        teacher_model_params=teacher_model_params,
        teacher_embeddings=teacher_embeddings,
        space_mask_teacher=space_mask_teacher,
        space_mask_new=space_mask_new,
        teacher_config=teacher_config,
        student_config=student_config,
        args=args,
        hypernet=hypernet,
        learning_rate_fn=learning_rate_fn,
        optimizer_kwargs=optimizer_kwargs,
        shard_patterns=shard_patterns,
    )

    teacher_config.vocab_size = teacher_model.config.vocab_size = state.params[
        "teacher_embeddings"
    ].shape[0]
    student_config.vocab_size = new_model.config.vocab_size = state.params[
        "new_embeddings"
    ].shape[0]
    logger.info(
        f"Updated source vocab size: {teacher_config.vocab_size} (after padding)"
    )
    logger.info(
        f"Updated target vocab size: {student_config.vocab_size} (after padding)"
    )

    overlapping_embeddings_mask = np.pad(
        overlapping_embeddings_mask,
        (
            0,
            utils.get_n_pad(
                len(overlapping_embeddings_mask), student_config.vocab_size
            ),
        ),
        mode="constant",
        constant_values=False,
    )
    hypernet_fn, teacher_model_fn, new_model_fn = (
        hypernet.apply,
        teacher_model.__call__,
        new_model.__call__,
    )
    utils.param_report(state.params, state.train_mask)
    train_mask = jax.device_get(state.train_mask)

    collator = collators.TokenizerAlignerCollator(
        tokenizer_teacher,
        target_tokenizer,
        max_teacher_length=args.max_teacher_length,
        max_student_length=args.max_student_length,
        special_tokens_mode=args.special_tokens_mode,
        with_expanded_input_ids=args.add_expanded_input_ids,
        use_chat_template=args.use_chat_template,
        chat_template_mode=args.chat_template_mode,
        loss_mask_mode=args.loss_mask_mode,
        tokenizer_pair_data_path=args.tokenizer_pair_data_path,
        tokenizer_pair_bias_threshold=args.tokenizer_pair_bias_threshold,
    )

    train_dataloader = StatefulDataLoader(
        dataset.get_torch_dataset(),
        batch_size=1,  # batched internally
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    ppl_eval_dataloader = torch.utils.data.DataLoader(
        ppl_eval_data.get_torch_dataset(),
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    if jax.process_index() == 0:
        wandb.init(project="tokenkit", name=args.name, config=OmegaConf.to_object(args))
        wandb.run.log_code()

    def predict_embeddings(params):  # TODO: add indices for subsampling
        embeddings = params["new_embeddings"]
        embeddings = jax.lax.with_sharding_constraint(
            embeddings, NamedSharding(mesh, P("model", None, "data"))
        )

        predicted_embeddings = hypernet_fn(
            params["hypernet"],
            embeddings[:, None, :, :],
            jnp.ones((embeddings.shape[0], 1), dtype=bool),
            jnp.arange(embeddings.shape[0], dtype=jnp.int32),
        )

        return jax.lax.with_sharding_constraint(
            predicted_embeddings, NamedSharding(mesh, P("model", None, "data"))
        )

    def compute_inputs_embeds(model_params, input_ids, expanded_input_ids):
        input_embeddings = param.get(
            model_params, param.get_input_embedding_path(student_config.model_type)
        )

        # NOTE: this assumes Llama/Gemma-style where position embeddings are part of attention mechanism
        if args.add_expanded_input_ids:
            standard_inputs_embeds = jnp.take(
                input_embeddings,
                input_ids,
                axis=0,
            )
            expanded_inputs_embeds = jnp.take(
                model_params["original_embeddings"][:, 0, :],
                expanded_input_ids,
                axis=0,
            )

            inputs_embeds = standard_inputs_embeds + expanded_inputs_embeds
        else:
            inputs_embeds = jnp.take(
                input_embeddings,
                input_ids,
                axis=0,
            )

        return inputs_embeds

    def train_step(state, batch, global_batch):
        def compute_loss(params):
            scalar_report = {}  # extra logging

            if args.train_model_mode == "lora":
                model_params_with_lora = lora.materialize_lora(
                    params["model"],
                    params["model_lora"],
                    alpha=args.model_lora_alpha,
                )
            else:
                model_params_with_lora = params["model"]
            predicted_embeddings = predict_embeddings(params)
            model_params_with_predicted_embeddings = param.assign_embeddings(
                model_params_with_lora,
                predicted_embeddings,
                config=student_config,
            )

            teacher_model_params = param.assign_embeddings(
                params["teacher_model"],
                params["teacher_embeddings"],
                config=teacher_config,
            )

            need_teacher = len([loss for loss in args.losses if loss != "clm"]) > 0
            if need_teacher:
                teacher_out = teacher_model_fn(
                    input_ids=batch["input_ids_original"],
                    params=teacher_model_params,
                    dropout_rng=None,
                    train=False,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                teacher_logits = (
                    teacher_out.logits.astype(jnp.float32)
                    + state.logit_mask_teacher[None, None]
                )
                # inexcplicably, `{student,teacher}_log_softmax_max` may be >0 without the jnp.clip(..)
                # clip and keep monitoring...
                teacher_logprobs = jnp.clip(
                    jax.nn.log_softmax(teacher_logits, axis=-1), max=0.0
                )
                scalar_report["teacher_log_softmax_max"] = teacher_logprobs.max()
                teacher_probs = jnp.exp(teacher_logprobs)
                scalar_report["teacher_exp_log_softmax_max"] = teacher_probs.max()
            else:
                teacher_out = teacher_logits = teacher_logprobs = teacher_probs = None

            inputs_embeds_new = compute_inputs_embeds(
                model_params_with_predicted_embeddings,
                batch["input_ids_new"],
                batch.get("expanded_input_ids_new"),
            )

            student_out = new_model_fn(
                input_ids=None,
                inputs_embeds=inputs_embeds_new,
                params=model_params_with_predicted_embeddings,
                dropout_rng=None,
                train=False,
                output_hidden_states=True,
                output_attentions=True,
            )
            student_logits = (
                student_out.logits.astype(jnp.float32)
                + state.logit_mask_new[None, None]
            )

            # inexplicably, `{student,teacher}_log_softmax_max` may be >0 without the jnp.clip(..)
            # clip and keep monitoring...
            student_logprobs = jnp.clip(
                jax.nn.log_softmax(student_logits, axis=-1), max=0.0
            )
            scalar_report["student_log_softmax_max"] = student_logprobs.max()
            student_probs = jnp.exp(student_logprobs)
            scalar_report["student_exp_log_softmax_max"] = student_probs.max()

            scalar_report["n_teacher_used_tokens"] = (
                global_batch["alignment_matrix_b"].any(-1).sum(-1).mean()
            )
            scalar_report["n_student_used_tokens"] = (
                global_batch["alignment_matrix_a"].any(-1).sum(-1).mean()
            )
            scalar_report["loss_mask_original_mean"] = global_batch[
                "loss_mask_original"
            ][:, 1:].mean()
            scalar_report["loss_mask_new_mean"] = global_batch["loss_mask_new"][
                :, 1:
            ].mean()

            loss_args = losses.LossArgs(
                state=state,
                params=params,
                batch=batch,
                global_batch=global_batch,
                teacher_config=teacher_config,
                new_config=student_config,
                teacher_out=teacher_out,
                student_out=student_out,
                tokenizer_teacher=tokenizer_teacher,
                tokenizer_new=target_tokenizer,
                teacher_probs=teacher_probs,
                teacher_logprobs=teacher_logprobs,
                teacher_logits=teacher_logits,
                student_probs=student_probs,
                student_logprobs=student_logprobs,
                student_logits=student_logits,
                predicted_embeddings=predicted_embeddings,
                scalar_report=scalar_report,
            )

            total_loss = 0.0
            loss_ema_stats = state.loss_ema_stats

            for loss_idx, loss in enumerate(args.losses):
                if loss == "clm":
                    current_loss = losses.compute_clm_loss(args, loss_args)
                elif loss == "distill_latents":
                    current_loss = losses.compute_distill_latents_loss(args, loss_args)
                elif loss.startswith("distill_alm"):
                    kind = loss[len("distill_alm_") :]
                    if len(kind) == 0:
                        kind = "unbiased"
                    current_loss = losses.compute_alm_loss(
                        chunk_kind=kind,
                        args=args,
                        loss_args=loss_args,
                    )
                elif loss.startswith("distill_alm_side_path"):
                    kind = loss[len("distill_alm_side_path_") :]
                    if len(kind) == 0:
                        kind = "unbiased"
                    current_loss = losses.compute_alm_side_path_loss(
                        chunk_kind=kind,
                        student_mapping=student_mapping,
                        teacher_mapping=teacher_mapping,
                        args=args,
                        loss_args=loss_args,
                    )
                elif loss == "baseline_dskd":
                    current_loss = losses.compute_baseline_dskd_loss(args, loss_args)
                elif loss == "baseline_uld":
                    current_loss = losses.compute_baseline_uld_loss(args, loss_args)
                elif loss == "baseline_mined":
                    current_loss = losses.compute_baseline_mined_loss(
                        mined_mapping, args, loss_args
                    )

                weight = (
                    args.loss_weights[loss_idx]
                    if args.loss_weights is not None
                    else 1.0
                )
                if args.loss_schedules is not None:
                    if args.loss_schedules[loss_idx] == "cosine":
                        weight = (
                            weight * (1 + jnp.cos(jnp.pi * state.step / args.steps)) / 2
                        )
                    elif args.loss_schedules[loss_idx] == "reverse_cosine":
                        weight = (
                            weight * (1 - jnp.cos(jnp.pi * state.step / args.steps)) / 2
                        )
                    elif args.loss_schedules[loss_idx] == "linear":
                        weight = weight * state.step / args.steps
                    elif args.loss_schedules[loss_idx] == "constant":
                        pass
                    else:
                        raise ValueError(
                            "Invalid loss schedule: {}".format(
                                args.loss_schedules[loss_idx]
                            )
                        )

                scalar_report[f"loss/{loss}"] = current_loss
                scalar_report[f"loss/{loss}_weight"] = weight

                if args.loss_weight_mode == "balance":
                    total_loss += (
                        weight * current_loss / jax.lax.stop_gradient(current_loss)
                    )
                elif args.loss_weight_mode == "uncertainty":
                    uncertainty_s = params["loss_weights"][loss_idx]
                    total_loss += weight * current_loss * jnp.exp(-uncertainty_s) + (
                        uncertainty_s / 2
                    )
                    scalar_report[f"loss/uncertainty_s_{loss}"] = uncertainty_s
                elif args.loss_weight_mode == "ema":
                    loss_ema_stats = loss_ema_stats.at[loss_idx, 0].set(
                        jnp.where(
                            jnp.isnan(loss_ema_stats[loss_idx, 0]),
                            current_loss,
                            args.ema_alpha * loss_ema_stats[loss_idx, 0]
                            + (1 - args.ema_alpha) * current_loss,
                        )
                    )
                    loss_ema_stats = loss_ema_stats.at[loss_idx, 1].set(
                        jnp.where(
                            jnp.isnan(loss_ema_stats[loss_idx, 1]),
                            1.0,
                            args.ema_alpha * loss_ema_stats[loss_idx, 1]
                            + (1 - args.ema_alpha)
                            * (current_loss - loss_ema_stats[loss_idx, 0]) ** 2,
                        )
                    )

                    running_std = jnp.maximum(
                        jnp.sqrt(loss_ema_stats[loss_idx, 1]), 1e-6
                    )
                    normalized_loss = (
                        current_loss - loss_ema_stats[loss_idx, 0]
                    ) / running_std
                    scalar_report[f"loss/{loss}_normalized"] = normalized_loss
                    scalar_report[f"loss/{loss}_ema_mean"] = loss_ema_stats[loss_idx, 0]
                    scalar_report[f"loss/{loss}_ema_var"] = loss_ema_stats[loss_idx, 1]
                    total_loss += weight * normalized_loss
                else:
                    total_loss += weight * current_loss

            return total_loss, (scalar_report, loss_ema_stats)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, (scalar_report, loss_ema_stats)), grad = grad_fn(state.params)

        def prefix_freeze(grad):
            for key in grad:
                if key not in {"new_embeddings"}:
                    grad[key] = jax.tree.map(lambda x: jnp.zeros_like(x), grad[key])

            grad["new_embeddings"] *= ~overlapping_embeddings_mask[:, None, None]

            return grad

        grad = jax.lax.cond(
            state.step < args.prefix_steps, prefix_freeze, lambda grad: grad, grad
        )
        new_state = state.apply_gradients(grads=grad, loss_ema_stats=loss_ema_stats)

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
            **scalar_report,
        }
        return new_state, metrics

    def eval_step(state, batch):
        if args.train_model_mode == "lora":
            model_params_with_lora = lora.materialize_lora(
                state.params["model"],
                state.params["model_lora"],
                alpha=args.model_lora_alpha,
            )
        else:
            model_params_with_lora = state.params["model"]
        predicted_embeddings = predict_embeddings(state.params)
        model_params_with_embeddings = param.assign_embeddings(
            model_params_with_lora,
            predicted_embeddings,
            config=student_config,
        )

        inputs_embeds_new = compute_inputs_embeds(
            model_params_with_embeddings,
            batch["input_ids_new"],
            batch.get("expanded_input_ids_new"),
        )

        logits = new_model_fn(
            input_ids=None,
            inputs_embeds=inputs_embeds_new,
            params=model_params_with_embeddings,
            dropout_rng=None,
            train=False,
        ).logits

        loss = cross_entropy(
            logits,
            batch["input_ids_new"],
            batch["attention_mask_new"],
            logit_mask=state.logit_mask_new,
        )

        return {"loss": loss}

    batch_shardings = jax.tree.map(
        lambda x: NamedSharding(mesh, x), collator.get_batch_pspecs()
    )

    jtrain_step = jax.jit(
        train_step,
        in_shardings=(state_shardings, batch_shardings, batch_shardings),
        out_shardings=(state_shardings, None),
        donate_argnums=(0,),
    )
    jeval_step = jax.jit(
        eval_step,
        in_shardings=(state_shardings, batch_shardings),
    )
    jpredict_embeddings = jax.jit(
        predict_embeddings,
        in_shardings=(state_shardings.params,),
        out_shardings=NamedSharding(mesh, P("model", None, "data")),
    )
    if args.train_model_mode == "lora":
        jmaterialize_lora = jax.jit(
            lora.materialize_lora,
            in_shardings=(
                state_shardings.params["model"],
                state_shardings.params["model_lora"],
            ),
            out_shardings=state_shardings.params["model"],
            donate_argnums=(0,),
            static_argnums=(2,),
        )
        jdematerialize_lora = jax.jit(
            lora.dematerialize_lora,
            in_shardings=(
                state_shardings.params["model"],
                state_shardings.params["model_lora"],
            ),
            out_shardings=state_shardings.params["model"],
            donate_argnums=(0,),
            static_argnums=(2,),
        )
    else:
        jmaterialize_lora = lambda x, y, z: x
        jdematerialize_lora = lambda x, y, z: x

    def eval_loop(dataloader):
        eval_metrics = []

        for batch in tqdm(dataloader, desc="Running PPL evaluation..."):
            batch = sharding.sync_across_devices(batch)
            batch = sharding.to_global_array(batch, batch_shardings)
            step_metrics = jeval_step(state, batch)
            eval_metrics.append(step_metrics)

        eval_metrics = jax.tree.map(np.mean, common_utils.stack_forest(eval_metrics))
        return eval_metrics

    diter = iter(train_dataloader)

    if args.do_cost_analysis:
        first_batch = next(iter(train_dataloader))
        compiled_train_step_fn = jtrain_step.lower(
            state, first_batch, first_batch
        ).compile()
        flops_per_step = compiled_train_step_fn.cost_analysis()["flops"]
        memory_per_step = (
            compiled_train_step_fn.memory_analysis().output_size_in_bytes
            + compiled_train_step_fn.memory_analysis().temp_size_in_bytes
        )
        logger.info("TFLOPs per step:", flops_per_step / (10**12))
        logger.info("Memory (MB) per step:", memory_per_step / (1024**2))
        sys.exit()

    train_metrics = []
    start_time = time.time()

    upload_executor = None
    upload_name = args.name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")

    grad_acc_steps = args.optimizer.get("grad_acc_steps") or 1
    assert args.data.batch_size % grad_acc_steps == 0
    local_batch_size = args.data.batch_size // grad_acc_steps

    for step in tqdm(range(args.steps)):
        try:
            batch = next(diter)
        except StopIteration:
            diter = iter(train_dataloader)
            batch = next(diter)

        batch = sharding.sync_across_devices(batch)
        global_batch = sharding.to_global_array(batch, batch_shardings)

        if args.dry_run:
            continue

        for grad_acc_step in range(grad_acc_steps):
            if grad_acc_steps > 1:
                start, end = (
                    grad_acc_step * local_batch_size,
                    (grad_acc_step + 1) * local_batch_size,
                )

                def split_local(arr):
                    if arr.shape[0] == args.data.batch_size:
                        return arr[start:end]
                    else:
                        assert len(arr.shape) == 1  # otherwise ambiguous
                        return arr

                local_batch = jax.tree.map(split_local, batch)
            else:
                local_batch = batch

            local_batch = sharding.to_global_array(local_batch, batch_shardings)

            state, step_metrics = jtrain_step(
                state.replace(step=step), local_batch, global_batch
            )

            train_metrics.append(step_metrics)

        if (step + 1) % args.sync_interval == 0:
            stacked_train_metrics = jax.tree.map(
                jax.device_get, common_utils.stack_forest(train_metrics)
            )
            end_step = step + 1
            start_step = end_step - args.sync_interval
            for i in range(start_step, end_step, args.log_interval):
                for key, values in stacked_train_metrics.items():
                    avg_value = values[
                        (i - start_step)
                        * grad_acc_steps : (i - start_step + args.log_interval)
                        * grad_acc_steps
                    ].mean()
                    utils.log({f"train/{key}": avg_value}, step=i + args.log_interval)

            utils.log({"step": end_step}, step=i + args.log_interval)
            train_metrics = []

        if (step + 1) % args.eval_interval == 0 or (
            step == 0 and args.eval_at_step_zero
        ):
            # TODO: probably extract into eval function doing everything here
            logger.info("PPL Eval:")
            ppl_metrics = eval_loop(ppl_eval_dataloader)
            ppl_metrics = {f"eval_{k}": v for k, v in ppl_metrics.items()}
            utils.log(ppl_metrics, step=step + 1)

            if not args.skip_lm_eval:
                original_vocab = tokenizer_student_original.get_vocab()

                predicted_embeddings = jpredict_embeddings(state.params)
                model_params_with_embeddings = param.assign_embeddings(
                    jmaterialize_lora(
                        state.params["model"],
                        state.params.get("model_lora"),
                        args.model_lora_alpha,
                    ),
                    predicted_embeddings,
                    config=student_config,
                )
                model_params_with_embeddings_shardings = sharding.get_sharding_fn(
                    shard_patterns, mesh
                )({"params": model_params_with_embeddings})["params"]

                @partial(
                    jax.jit,
                    static_argnames=("model_fn", "atol"),
                    in_shardings=(
                        model_params_with_embeddings_shardings,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                    out_shardings=(None, None),
                )
                def jaxlm_inner_score_fn(
                    model_fn,
                    params,
                    input_ids,
                    expanded_input_ids,
                    labels,
                    suffix_mask,
                    space_mask,
                    logit_mask,
                    atol=eval.ATOL,
                ):
                    inputs_embeds = compute_inputs_embeds(
                        params,
                        input_ids,
                        expanded_input_ids,
                    )
                    return eval.score(
                        model_fn,
                        params,
                        (None, inputs_embeds),
                        labels=labels,
                        suffix_mask=suffix_mask,
                        space_mask=space_mask,
                        logit_mask=logit_mask,
                        atol=atol,
                    )

                def jaxlm_score_fn(model_fn, params, model_args, *pargs):
                    (input_ids,) = model_args
                    if args.add_expanded_input_ids:
                        expanded_input_ids = utils.expand_input_ids(
                            input_ids,
                            tokenizer=target_tokenizer,
                            original_vocab=original_vocab,
                            use_heuristic=True,
                        )
                    else:
                        expanded_input_ids = None
                    return jaxlm_inner_score_fn(
                        model_fn,
                        params,
                        input_ids,
                        expanded_input_ids,
                        *pargs,
                    )

                lm_eval_metrics, post_eval_params_buffer = eval.evaluate(
                    model=new_model,
                    config=student_config,
                    params=model_params_with_embeddings,
                    tokenizer=target_tokenizer,
                    logit_mask=state.logit_mask_new == 0,
                    output=output_dir / f"step_{step + 1}" / "lm_eval",
                    jaxlm_kwargs={"score_fn": jaxlm_score_fn},
                    **OmegaConf.to_object(args.eval),
                )
                state.params["model"] = jdematerialize_lora(
                    param.unassign_embeddings(post_eval_params_buffer, student_config),
                    state.params.get("model_lora"),
                    args.model_lora_alpha,
                )

                logger.info("LM Eval:")
                lm_eval_metrics = {
                    "_".join(k): v
                    for k, v in traverse_util.flatten_dict(lm_eval_metrics).items()
                }
                lm_eval_metrics = {
                    f"lm_eval_{k}": v for k, v in lm_eval_metrics.items()
                }

                utils.log(lm_eval_metrics, step=step + 1)

        if (step + 1) % args.save_interval == 0 or (
            step == 0 and args.save_at_step_zero
        ):
            if upload_executor is not None:
                upload_executor.shutdown(wait=True)
            multihost_utils.sync_global_devices("uploaded previous checkpoint")

            del student_config.mesh
            student_config.save_pretrained(output_dir)
            student_config.mesh = mesh
            target_tokenizer.save_pretrained(output_dir)
            checkpoint.save(
                output_dir / "params.msgpack",
                state.params,
                state_shardings.params,
                mesh,
                train_mask,
                keys_to_keep={"hypernet", "new_embeddings"},
            )

            if jax.process_index() == 0 and args.export_to_gcs_bucket is not None:
                upload_executor = gcs_utils.upload_directory_to_gcs(
                    args.export_to_gcs_bucket,
                    output_dir,
                    os.path.join(upload_name, f"step_{step + 1}"),
                )

        if (step + 1) % args.sync_interval == 0:
            if jax.process_index() == 0:
                utils.log(
                    {
                        "step": step + 1,
                        "time": time.time() - start_time,
                        "epoch": step / len(train_dataloader),
                    },
                    step=step + 1,
                    commit=True,
                )

    if upload_executor is not None:
        upload_executor.shutdown(wait=True)
    multihost_utils.sync_global_devices("uploaded final checkpoint")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "100"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = (
        True  # careful about this, required for lm_eval
    )

    my_app()
