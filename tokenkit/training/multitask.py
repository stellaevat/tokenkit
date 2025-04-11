import jax
import jax.numpy as jnp
from typing import Any


def pcgrad(task_grads: Any) -> Any:
    """
    Implements PCGrad (Project Conflicting Gradients) in JAX.

    Args:
        task_grads: A pytree containing gradients where the first dimension of each
                   array represents different tasks. Shape: (n_tasks, ...)

    Returns:
        Modified gradients after applying PCGrad, with the same structure as the input.
    """
    # Get the structure of the input pytree and convert to flat representation
    flat_task_grads, treedef = jax.tree.flatten(task_grads)

    # Check if all elements are arrays and get number of tasks
    n_tasks = None
    for grad in flat_task_grads:
        if grad is not None:
            n_tasks = grad.shape[0]
            break

    if n_tasks is None:
        # No valid gradients found
        return task_grads

    # Create a new flat list to store the modified gradients
    modified_flat_grads = []

    # Process each element in the flat list
    for grad in flat_task_grads:
        if grad is None:
            modified_flat_grads.append(None)
            continue

        # Ensure the first dimension matches the number of tasks
        assert (
            grad.shape[0] == n_tasks
        ), f"Expected first dimension to be {n_tasks}, got {grad.shape[0]}"

        # Extract shape for reshaping later
        original_shape = grad.shape
        # Reshape to (n_tasks, -1) for easier processing
        reshaped_grad = grad.reshape(n_tasks, -1)

        # Initialize modified gradients with copies of the original
        modified_grad = reshaped_grad.copy()

        # Apply PCGrad for each task
        for i in range(n_tasks):
            # Project task i's gradient onto normal plane of other tasks' gradients if they conflict
            grad_i = modified_grad[i]

            for j in range(n_tasks):
                if i == j:
                    continue

                grad_j = reshaped_grad[j]

                # Calculate dot product to check for conflict
                dot_product = jnp.sum(grad_i * grad_j)

                # If dot product is negative, project gradient
                def project(g_i, g_j, dot):
                    g_j_norm_squared = jnp.sum(g_j * g_j)
                    # Avoid division by zero
                    safe_norm_squared = jnp.maximum(g_j_norm_squared, 1e-8)
                    # Project g_i onto normal plane of g_j
                    return g_i - jnp.minimum(0.0, dot) * g_j / safe_norm_squared

                modified_grad = modified_grad.at[i].set(
                    project(modified_grad[i], grad_j, dot_product)
                )

        # Reshape back to original shape and add to the modified list
        modified_flat_grads.append(modified_grad.reshape(original_shape))

    # Reconstruct the pytree with modified gradients
    return jax.tree.unflatten(treedef, modified_flat_grads)


def gradmag(task_grads: Any, epsilon: float = 1e-8) -> Any:
    """
    Normalizes gradients of all tasks to have the same magnitude.

    Args:
        task_grads: A pytree containing gradients where the first dimension of each
                   array represents different tasks. Shape: (n_tasks, ...)
        epsilon: Small constant to avoid division by zero

    Returns:
        Modified gradients after normalization, with the same structure as the input.
    """
    global_grad_norms = compute_global_grad_norm(task_grads) + epsilon
    return jax.tree.map(
        lambda grad: grad
        / jnp.reshape(global_grad_norms, (-1,) + (1,) * (grad.ndim - 1)),
        task_grads,
    )


def gradclip(task_grads: Any, max_norm: float) -> Any:
    """
    Clips gradients of all tasks to have the same magnitude.

    Args:
        task_grads: A pytree containing gradients where the first dimension of each
                   array represents different tasks. Shape: (n_tasks, ...)
        max_norm: Maximum allowed norm for gradients

    Returns:
        Modified gradients after clipping, with the same structure as the input.
    """
    global_grad_norms = compute_global_grad_norm(task_grads)
    denominators = jnp.maximum(global_grad_norms, max_norm)
    return jax.tree.map(lambda grad: grad / jnp.reshape(denominators, (-1,) + (1,) * (grad.ndim - 1)) * max_norm, task_grads)


def compute_global_grad_norm(task_grads: Any) -> jnp.ndarray:
    """
    Computes the global gradient norm for a pytree of task gradients.

    Args:
        task_grads: A pytree containing gradients where the first dimension of each
                   array represents different tasks. Shape: (n_tasks, ...)

    Returns:
        Global gradient norm, a scalar.
    """
    global_grad_norms = jnp.sqrt(
        jax.tree.reduce(
            lambda x, y: x + y,
            jax.tree.map(
                lambda x: jnp.square(x).reshape(x.shape[0], -1).sum(axis=1),
                task_grads,
            ),
        )
    )
    return global_grad_norms
