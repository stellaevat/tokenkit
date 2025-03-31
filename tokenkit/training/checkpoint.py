import math

import jax
from flax import serialization, traverse_util
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def save(
    path,
    params,
    param_shardings,
    mesh,
    train_mask,
    keys_to_keep={
        "hypernet",
    },
    batch_size=16,
):
    flat_keys_to_save = [
        k
        for k, trainable in traverse_util.flatten_dict(train_mask).items()
        if trainable or k[0] in keys_to_keep
    ]
    flat_params = traverse_util.flatten_dict(params)
    flat_shardings = traverse_util.flatten_dict(param_shardings)

    flat_params_to_save = {k: flat_params[k] for k in flat_keys_to_save}
    shardings_to_save = {k: flat_shardings[k] for k in flat_keys_to_save}

    none_shardings_to_save = jax.tree.map(
        lambda _: NamedSharding(mesh, P()), shardings_to_save
    )

    keys = list(flat_params_to_save.keys())
    n_batches = math.ceil(len(keys) / batch_size)

    all_flat_out_params = {}

    for i in range(n_batches):
        batch_keys = keys[i * batch_size : (i + 1) * batch_size]

        flat_device_params = jax.jit(
            lambda x: x,
            in_shardings=([shardings_to_save[k] for k in batch_keys],),
            out_shardings=[none_shardings_to_save[k] for k in batch_keys],
        )([flat_params_to_save[k] for k in batch_keys])

        for key, value in zip(batch_keys, flat_device_params):
            all_flat_out_params[key] = jax.device_get(value)
            value.delete()

    if jax.process_index() == 0:
        open(path, "wb").write(
            serialization.msgpack_serialize(
                traverse_util.unflatten_dict(all_flat_out_params), in_place=True
            )
        )

    multihost_utils.sync_global_devices("saved checkpoint")
