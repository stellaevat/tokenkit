import logging
from pprint import pformat

import hydra
import jax
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, FlaxAutoModelForCausalLM

from tokenkit import utils
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.models import param, sharding

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="zett")
def my_app(args: DictConfig) -> None:
    logger.info(pformat(OmegaConf.to_object(args)))

    # Load the model & tokenizer
    source_tokenizer = load_byteify_tokenizer(args.source_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(args.target_tokenizer_name)

    mesh = sharding.get_mesh(devices=jax.devices("cpu"))
    config = AutoConfig.from_pretrained(args.source_model_pretrained_name_or_path)
    config.mesh = mesh

    model = FlaxAutoModelForCausalLM.from_config(
        config,
        _do_init=False,
        input_shape=(1, 128),
    )
    del model.config.mesh

    model_params = param.load_params(
        pretrained_model_name_or_path=args.source_model_pretrained_name_or_path
    )

    embeddings, model_params = param.stack_embeddings(
        model_params,
        config,
        pop_embeddings=True,
    )

    diff_embeddings, original_to_new_indices, diff_indices = utils.fvt(
        source_tokenizer,
        target_tokenizer,
        embeddings,
    )
    new_embeddings = embeddings[original_to_new_indices]
    if len(diff_indices) > 0:
        new_embeddings[diff_indices] = diff_embeddings

    model_params = param.assign_embeddings(model_params, new_embeddings, config)

    model.save_pretrained(args.output, params=model_params)
    config.save_pretrained(args.output)
    target_tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    my_app()
