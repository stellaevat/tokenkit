from dataclasses import dataclass
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
)
from omegaconf import OmegaConf
from flax import serialization, traverse_util
from pathlib import Path
import jax
import jax.numpy as jnp
from tokenkit.models.hypernet import Hypernet
from tokenkit.models import param, lora, sharding
from tokenkit import gcs_utils
import json
from pprint import pformat
import logging

logger = logging.getLogger(__name__)


@dataclass
class Args:
    checkpoint: str = "outputs/patch"
    output: str = "outputs/export"
    use_cpu: bool = False
    tmp_save_dir: str = "/tmp/tokenkit/"
    overwrite_args: str | None = None


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    tmp_checkpoint_dir = Path(args.tmp_save_dir) / "checkpoint"
    tmp_output_dir = Path(args.tmp_save_dir) / "output"

    tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_cpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        mesh = sharding.get_mesh(devices=jax.devices("cpu"))
    else:
        mesh = sharding.get_mesh()

    if gcs_utils.is_gcs_path(args.checkpoint):
        checkpoint_bucket, checkpoint_blob = gcs_utils.parse_gcs_path(args.checkpoint)
        checkpoint_dir = tmp_checkpoint_dir

        for filename in ["args.yaml", "params.msgpack", "config.json", "tokenizer.json", "tokenizer_config.json"]:
            gcs_utils.download_from_gcs(checkpoint_bucket, f"{checkpoint_blob}/{filename}", checkpoint_dir / filename)
    else:
        checkpoint_dir = Path(args.checkpoint)

    ckpt_args = OmegaConf.load(checkpoint_dir / "args.yaml")
    if args.overwrite_args is not None:
        ckpt_args = OmegaConf.merge(
            ckpt_args, OmegaConf.create(json.loads(args.overwrite_args))
        )

    logger.info("Using checkpoint args:")
    logger.info(pformat(ckpt_args))

    params = serialization.msgpack_restore(
        open(checkpoint_dir / "params.msgpack", "rb").read()
    )

    config = AutoConfig.from_pretrained(checkpoint_dir)
    config.mesh = mesh
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    dtype = getattr(jnp, ckpt_args.dtype)

    n_embd = params["new_embeddings"].shape[-1]

    hypernet = Hypernet(
        dtype=dtype,
        hidden_size=n_embd,
        num_embeddings=1 if config.tie_word_embeddings else 2,
        max_seq_length=1,
        vocab_size=config.vocab_size,
        **ckpt_args.hypernet,
    )
    model_kwargs = OmegaConf.to_object(ckpt_args.student)
    model = FlaxAutoModelForCausalLM.from_config(config, dtype=dtype, _do_init=False)

    if "model" in params:
        model_params = params["model"]
        original_model_params = param.load_params(**model_kwargs)
    else:
        model_params = original_model_params = param.load_params(**model_kwargs)

    # model params may be partial at this point e.g. if trained with LoRA, merge them
    flat_merged_model_params = traverse_util.flatten_dict(original_model_params)
    flat_model_params = traverse_util.flatten_dict(model_params)

    for key in flat_model_params.keys():
        flat_merged_model_params[key] = flat_model_params[key]

    merged_model_params = traverse_util.unflatten_dict(flat_merged_model_params)
    # assigned later
    merged_model_params = param.unassign_embeddings(merged_model_params, config=config)

    if "model_lora" in params:
        logger.info("Materializing LoRA parameters...")
        merged_model_params = lora.materialize_lora(
            merged_model_params,
            params["model_lora"],
            ckpt_args.model_lora_alpha,
        )

    hypernet_fn = hypernet.apply

    def predict_embeddings(params):  # TODO: add indices for subsampling
        embeddings = params["new_embeddings"]

        predicted_embeddings = hypernet_fn(
            params["hypernet"],
            embeddings[:, None, :, :],
            jnp.ones((embeddings.shape[0], 1), dtype=bool),
            jnp.arange(embeddings.shape[0], dtype=jnp.int32),
        )

        return predicted_embeddings

    embeddings = jax.device_get(predict_embeddings(params))
    embeddings = embeddings.copy()  # not writeable otherwise

    # remove padding
    config.vocab_size = len(tokenizer)
    embeddings = embeddings[: len(tokenizer)]  # remove padding

    merged_model_params = param.assign_embeddings(merged_model_params, embeddings, config=config)

    model_to_save = FlaxAutoModelForCausalLM.from_config(config)
    if gcs_utils.is_gcs_path(args.output):
        output_dir = tmp_output_dir
    else:
        output_dir = Path(args.output)

    del config.mesh

    # from_flax does not work with multiple shards so it is more convenient to save the model as a single shard
    model_to_save.save_pretrained(
        output_dir, params=merged_model_params, max_shard_size="100GB"
    )
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    if gcs_utils.is_gcs_path(args.output):
        output_bucket, output_blob = gcs_utils.parse_gcs_path(args.output)
        for filename in ["config.json", "flax_model.msgpack", "tokenizer.json", "tokenizer_config.json"]:
            gcs_utils.upload_to_gcs(output_bucket, output_dir / filename, f"{output_blob}/{filename}")