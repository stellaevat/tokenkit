import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple


def pcgrad(grads: Dict[str, Any], epsilon: float = 1e-8) -> Dict[str, Any]:
    """
    Implements PCGrad (Project Conflicting Gradients) in JAX.

    PCGrad projects each task's gradients onto the normal plane of any
    other task's gradients if the inner product is negative (i.e., they conflict).

    Args:
        grads: A dictionary or pytree of gradients for each task.
              Each key represents a task, and value contains gradients for that task.

    Returns:
        Modified gradients after applying PCGrad.
    """
    # Convert the gradient pytree to a flat list for easier processing
    grad_tasks = list(grads.items())
    num_tasks = len(grad_tasks)

    # Flatten gradients for each task
    flat_grads = []
    treedef = []
    for task, grad in grad_tasks:
        flat_grad, tree = jax.tree_util.tree_flatten(grad)
        flat_grad = [g.reshape(-1) if g is not None else None for g in flat_grad]
        flat_grads.append(flat_grad)
        treedef.append(tree)

    # Process each task's gradients
    modified_flat_grads = []
    for i in range(num_tasks):
        task_grad = flat_grads[i]
        modified_grad = [g.copy() if g is not None else None for g in task_grad]

        # Project gradients onto the normal plane of each other task's gradients
        for j in range(num_tasks):
            if i == j:
                continue

            other_grad = flat_grads[j]
            for k in range(len(task_grad)):
                if task_grad[k] is None or other_grad[k] is None:
                    continue

                # Calculate dot product to check for conflict
                dot_product = jnp.sum(task_grad[k] * other_grad[k])

                # If dot product is negative, project gradient
                def project(g_i, g_j, dot):
                    g_j_norm_squared = jnp.sum(g_j * g_j)
                    # Avoid division by zero
                    safe_norm_squared = jnp.maximum(g_j_norm_squared, epsilon)
                    # Project g_i onto normal plane of g_j
                    return g_i - jnp.minimum(0.0, dot) * g_j / safe_norm_squared

                modified_grad[k] = jax.lax.cond(
                    dot_product < 0,
                    lambda: project(modified_grad[k], other_grad[k], dot_product),
                    lambda: modified_grad[k],
                )

        # Unflatten the modified gradients back to their original structure
        unflat_shapes = [g.shape if g is not None else None for g in flat_grads[i]]
        reshaped_grads = []

        for k, g in enumerate(modified_grad):
            if g is None:
                reshaped_grads.append(None)
            else:
                reshaped_grads.append(g.reshape(unflat_shapes[k]))

        modified_flat_grads.append(reshaped_grads)

    # Convert back to original pytree structure
    result = {}
    for i, (task, _) in enumerate(grad_tasks):
        result[task] = jax.tree_util.tree_unflatten(treedef[i], modified_flat_grads[i])

    return result


def gradmag(grads, epsilon: float = 1e-8):
    """Normalize gradients of all tasks to have the same magnitude."""
    normalized_grads = {}

    for task, grad in grads.items():
        # Compute the norm of the gradient
        flat_grad, treedef = jax.tree_util.tree_flatten(grad)
        flat_grad = [g for g in flat_grad if g is not None]

        # Flatten gradients for norm computation
        squared_sum = 0
        for g in flat_grad:
            squared_sum += jnp.sum(g**2)

        grad_norm = jnp.maximum(jnp.sqrt(squared_sum), epsilon)

        # Normalize the gradient
        normalized_grad = jax.tree_map(
            lambda g: g / grad_norm if g is not None else None, grad
        )

        normalized_grads[task] = normalized_grad

    return normalized_grads
