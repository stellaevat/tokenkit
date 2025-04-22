import logging
from pprint import pformat

import jax
from dataclasses import dataclass, asdict
from transformers import FlaxAutoModelForCausalLM

from tokenkit.hf import get_config
from tokenkit import utils
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.models import param, sharding
from tokenkit import parse_args

logger = logging.getLogger(__name__)

@dataclass
class ZettArgs:
    source_model: parse_args.ModelArgs
    target_tokenizer_name: str
    output: str

def main(args: ZettArgs) -> None:
    logger.info(pformat(args))

    # Load the model & tokenizer
    source_tokenizer = load_byteify_tokenizer(args.source_model.tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(args.target_tokenizer_name)

    mesh = sharding.get_mesh(devices=jax.devices("cpu"))
    config = get_config(args.source_model.pretrained_model_name_or_path)
    config.mesh = mesh

    model = FlaxAutoModelForCausalLM.from_config(
        config,
        _do_init=False,
        input_shape=(1, 128),
    )
    del model.config.mesh

    model_params = param.load_params(**asdict(args.source_model))
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
    main(parse_args.parse_args(ZettArgs))