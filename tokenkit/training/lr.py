import optax


def linear_warmup_linear_decay_with_linear_prefix(
    lr, steps, warmup_steps, prefix_steps=0, prefix_lr=0.0
):
    """Returns a linear warmup, linear decay learning rate function."""

    prefix_fn = optax.linear_schedule(
        init_value=0.0, end_value=prefix_lr, transition_steps=prefix_steps
    )

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=warmup_steps,
    )

    decay_fn = optax.linear_schedule(
        init_value=lr,
        end_value=0.0,
        transition_steps=steps - warmup_steps - prefix_steps,
    )

    fn = optax.join_schedules(
        schedules=[prefix_fn, warmup_fn, decay_fn],
        boundaries=[
            prefix_steps,
            prefix_steps + warmup_steps,
        ],
    )

    return fn


def linear_warmup_cosine_decay_with_linear_prefix(
    lr, steps, warmup_steps, alpha=0.0, prefix_steps=0, prefix_lr=0.0
):
    """Returns a linear warmup, cosine decay learning rate function."""

    prefix_fn = optax.linear_schedule(
        init_value=0.0, end_value=prefix_lr, transition_steps=prefix_steps
    )

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=warmup_steps,
    )

    decay_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=steps - warmup_steps - prefix_steps,
        alpha=alpha,
    )

    fn = optax.join_schedules(
        schedules=[prefix_fn, warmup_fn, decay_fn],
        boundaries=[
            prefix_steps,
            prefix_steps + warmup_steps,
        ],
    )

    return fn
