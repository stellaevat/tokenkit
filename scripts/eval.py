"""
Example Usage:

python3 scripts/eval.py \
    +main.pretrained_model_name_or_path=\'benjamin/gemma-2-2b-it-flax\' \
    +main.tokenizer_name=\'google/gemma-2-2b-it:source=Gemma2\'

To evaluate a model with expanded input ids (byte-level models with n-gram embeddings), use:

python3 scripts/eval.py \
    +main.pretrained_model_name_or_path=<model_name> \
    +main.tokenizer_name=<model_name> \
    +expand.pretrained_model_name_or_path=benjamin/gemma-2-2b-it-flax \
    +expand.tokenizer_name=google/gemma-2-2b-it:source=Gemma2
"""

import logging
import os
from functools import partial
from pathlib import Path
from pprint import pformat, pprint

import datasets
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, FlaxAutoModelForCausalLM

from tokenkit import utils
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.eval import ATOL, evaluate, score
from tokenkit.models import param, sharding

logger = logging.getLogger(__name__)

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = (
    True  # careful about this, required for lm_eval
)


def pad_embeddings(embeddings, tokenizer):
    n_embed_diff = len(tokenizer) - len(embeddings)

    embeddings_mean = embeddings.mean(0)
    embeddings_std = embeddings.std(0)

    return np.concatenate(
        [
            embeddings,
            np.random.normal(
                size=(n_embed_diff, *embeddings.shape[1:]),
            )
            * embeddings_std[None]
            + embeddings_mean[None],
        ]
    )


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def my_app(args: DictConfig) -> None:
    logger.info(pformat(OmegaConf.to_object(args)))

    model_kwargs = OmegaConf.to_object(args.main)
    eval_kwargs = OmegaConf.to_object(args.eval)

    if args.use_cpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        mesh = sharding.get_mesh(devices=jax.devices("cpu"))
    else:
        mesh = sharding.get_mesh()

    if eval_kwargs["output"] is not None:
        output_dir = Path(eval_kwargs["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=args, f=output_dir / "args.yaml", resolve=True)

    config = AutoConfig.from_pretrained(**model_kwargs)
    config.max_length = eval_kwargs["lengths"][-1]
    config.mesh = mesh

    tokenizer = load_byteify_tokenizer(model_kwargs.pop("tokenizer_name"))

    if args.ignore_normalizer:
        tokenizer.backend_tokenizer.normalizer = None

    model = FlaxAutoModelForCausalLM.from_config(config, _do_init=False)

    params = param.load_params(**model_kwargs)

    if args.expand_input_ids:
        expand_model_kwargs = OmegaConf.to_object(args.expand)

        expand_config = AutoConfig.from_pretrained(**expand_model_kwargs)
        expand_tokenizer = load_byteify_tokenizer(
            expand_model_kwargs.pop("tokenizer_name")
        )
        expand_vocab = expand_tokenizer.get_vocab()

        expand_input_ids_model_params = param.load_params(**expand_model_kwargs)
        expand_input_ids_embeddings = param.get(
            expand_input_ids_model_params,
            param.get_input_embedding_path(expand_config.model_type),
        )

        n_overflow = expand_input_ids_embeddings.shape[0] % args.pad_to_multiple_of
        if n_overflow > 0:
            n_pad = args.pad_to_multiple_of - n_overflow
        else:
            n_pad = 0

        expand_input_ids_embeddings = np.pad(
            expand_input_ids_embeddings,
            ((0, n_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        expand_tokenizer = None
        expand_vocab = None
        expand_input_ids_embeddings = None

    input_embeddings = param.get(
        params, param.get_input_embedding_path(config.model_type)
    )
    input_embeddings = input_embeddings[: len(tokenizer)]

    if len(input_embeddings) < len(tokenizer):
        print("Padding input embeddings...")
        input_embeddings = pad_embeddings(input_embeddings, tokenizer)

    if not config.tie_word_embeddings:
        output_embeddings = param.get(
            params, param.get_output_embedding_path(config.model_type)
        )
        output_embeddings = output_embeddings[:, : len(tokenizer)]
        print("Padding output embeddings...")
        output_embeddings = pad_embeddings(output_embeddings.T, tokenizer).T
    else:
        output_embeddings = None

    n_overflow = input_embeddings.shape[0] % args.pad_to_multiple_of
    if n_overflow > 0:
        n_pad = args.pad_to_multiple_of - n_overflow
    else:
        n_pad = 0

    input_embeddings = np.pad(
        input_embeddings,
        ((0, n_pad), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    if output_embeddings is not None:
        output_embeddings = np.pad(
            output_embeddings,
            ((0, 0), (0, n_pad)),
            mode="constant",
            constant_values=0,
        )
    logit_mask = np.zeros((input_embeddings.shape[0],), dtype=bool)
    logit_mask[: model.config.vocab_size] = True

    model.config.vocab_size = input_embeddings.shape[0]

    params = param.put(
        params, param.get_input_embedding_path(config.model_type), input_embeddings
    )
    if output_embeddings is not None:
        params = param.put(
            params,
            param.get_output_embedding_path(config.model_type),
            output_embeddings,
        )

    shard_patterns = sharding.get_shard_patterns(config.model_type)
    param_shardings = sharding.get_sharding_fn(shard_patterns, mesh)(
        {"params": params}
    )["params"]
    params = sharding.to_devices(params, param_shardings, dtype=jnp.float32)

    multihost_utils.sync_global_devices("loaded weights")

    jaxlm_kwargs = {"precompile": not args.use_cpu}

    if args.expand_input_ids:
        # TODO: move elsewhere, probably into jaxlm
        def compute_inputs_embeds(model_params, input_ids, expanded_input_ids):
            input_embeddings = param.get(
                model_params, param.get_input_embedding_path(config.model_type)
            )

            standard_inputs_embeds = jnp.take(
                input_embeddings,
                input_ids,
                axis=0,
            )
            expanded_inputs_embeds = jnp.take(
                expand_input_ids_embeddings,
                expanded_input_ids,
                axis=0,
            )

            inputs_embeds = standard_inputs_embeds + expanded_inputs_embeds

            return inputs_embeds

        @partial(
            jax.jit,
            static_argnames=("model_fn", "atol"),
            in_shardings=(
                param_shardings,
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P()),
            ),
            out_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P())),
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
            atol=ATOL,
        ):
            inputs_embeds = compute_inputs_embeds(
                params,
                input_ids,
                expanded_input_ids,
            )
            return score(
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

            expanded_input_ids = utils.expand_input_ids(
                input_ids,
                tokenizer=tokenizer,
                original_vocab=expand_vocab,
                use_heuristic=True,
                maxlen=16,
            )

            return jaxlm_inner_score_fn(
                model_fn,
                params,
                input_ids,
                expanded_input_ids,
                *pargs,
            )

        jaxlm_kwargs["expand_input_ids"] = True
        jaxlm_kwargs["expand_input_ids_vocab"] = expand_vocab
        jaxlm_kwargs["expand_input_ids_embeddings"] = expand_input_ids_embeddings
        jaxlm_kwargs["score_fn"] = jaxlm_score_fn

    results, _ = evaluate(
        model=model,
        config=config,
        params=params,
        tokenizer=tokenizer,
        logit_mask=logit_mask,
        **eval_kwargs,
        jaxlm_kwargs=jaxlm_kwargs,
    )

    if jax.process_index() == 0:
        pprint(results)


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    my_app()
