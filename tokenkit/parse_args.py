from dataclasses import dataclass, fields, is_dataclass
from transformers import HfArgumentParser
from argparse import ArgumentParser
import json


@dataclass
class DataArgs:
    batch_size: int
    num_workers: int
    kind: str
    mix_languages: bool
    streaming: bool
    shuffle_buffer_size: int | str
    dataset_configs: list[dict]

@dataclass
class HypernetArgs:
    architecture: str
    num_layers: int
    residual: bool
    residual_alpha: float
    use_attention: bool

@dataclass
class OptimizerArgs:
    type: str
    weight_decay: float
    b1: float
    b2: float
    eps: float
    grad_acc_steps: int | None
    learning_rate: float
    max_grad_norm: float | None
    param_groups: list[dict]

@dataclass
class EvalArgs:
    tasks: list[str]
    lengths: list[int]
    tokens_per_batch: int
    add_bos: bool
    chat_template_mode: str
    confirm_run_unsafe_code: bool

@dataclass
class ModelArgs:
    pretrained_model_name_or_path: str
    tokenizer_name: str

def restore_dataclasses(args, cls):
    for field in fields(cls):
        if is_dataclass(field.type):
            setattr(
                args,
                field.name,
                restore_dataclasses(getattr(args, field.name), field.type),
            )
        elif isinstance(field.type, list) and is_dataclass(field.type.__args__[0]):
            setattr(
                args,
                field.name,
                [
                    restore_dataclasses(item, field.type.__args__[0])
                    for item in getattr(args, field.name)
                ],
            )
        elif isinstance(field.type, dict):
            setattr(
                args,
                field.name,
                {
                    k: restore_dataclasses(v, field.type.__args__[1])
                    for k, v in getattr(args, field.name).items()
                },
            )

    if not isinstance(args, cls):
        return cls(**args) if args is not None else None

    return args


def parse_args(cls):
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", type=str, nargs="*")
    meta_args = parser.parse_args()

    (args,) = HfArgumentParser([cls]).parse_yaml_file(meta_args.config)

    for overrides in meta_args.overrides:
        for override in overrides.split():
            first_equals = override.find("=")
            key = override[:first_equals].split(".")
            try:
                value = json.loads(override[first_equals + 1 :])
            except json.JSONDecodeError:
                value = override[first_equals + 1 :]

            current = args
            for k in key[:-1]:
                if isinstance(current, list):
                    current = current[int(k)]
                elif isinstance(current, dict):
                    current = current[k]
                else:
                    current = getattr(current, k)

            if isinstance(current, list):
                current[int(key[-1])] = value
            elif isinstance(current, dict):
                current[key[-1]] = value
            else:
                setattr(current, key[-1], value)

    return restore_dataclasses(args, cls)
