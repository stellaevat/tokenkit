from pprint import pprint

import optax
import regex as re
from flax import traverse_util


def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)

    # TODO: this is somewhat hacky but (almost) always accurate
    flat_mask = {
        path: not (
            path[-1] in {"bias", "b"}
            or any(
                ln_name in ".".join(path[-2:])
                for ln_name in {"layernorm", "layer_norm", "ln"}
            )
        )
        for path in flat_params
    }
    return traverse_util.unflatten_dict(flat_mask)


def get_optimizer(train_mask, learning_rate_fn, **optimizer_kwargs):
    transforms = []

    opt_type = optimizer_kwargs.pop("type")
    grad_acc_steps = optimizer_kwargs.pop("grad_acc_steps", None)
    max_grad_norm = optimizer_kwargs.pop("max_grad_norm", None)

    if max_grad_norm is not None:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))

    if opt_type == "adamw":
        opt_fn = optax.adamw
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    flat_param_group_labels = {}
    flat_train_mask = traverse_util.flatten_dict(train_mask)
    param_groups = optimizer_kwargs.pop("param_groups", [])
    optimizers = {
        "_default": opt_fn(
            mask=decay_mask_fn, learning_rate=learning_rate_fn, **optimizer_kwargs
        ),
        "_do_not_train": optax.set_to_zero(),
    }

    for group in param_groups:
        for key, trainable in flat_train_mask.items():
            if not trainable:
                flat_param_group_labels[key] = "_do_not_train"
            elif re.match(group["pattern"], ".".join(key)):
                flat_param_group_labels[key] = group["pattern"]
                if group["pattern"] not in optimizers:
                    optimizers[group["pattern"]] = opt_fn(
                        mask=decay_mask_fn,
                        learning_rate=lambda count: learning_rate_fn(count)
                        * group["lr_scale"],
                        **optimizer_kwargs,
                    )

    for key, trainable in flat_train_mask.items():
        if key not in flat_param_group_labels:
            if trainable:
                flat_param_group_labels[key] = "_default"
            else:
                flat_param_group_labels[key] = "_do_not_train"

    print("Special parameter groups:")
    pprint(
        {
            k: v
            for k, v in flat_param_group_labels.items()
            if v not in {"_default", "_do_not_train"}
        }
    )

    opt_fn = optax.multi_transform(
        optimizers,
        traverse_util.unflatten_dict(flat_param_group_labels),
    )

    transforms.append(opt_fn)

    opt = optax.chain(*transforms)

    if grad_acc_steps is not None:
        opt = optax.MultiSteps(opt=opt, every_k_schedule=grad_acc_steps)

    return opt
