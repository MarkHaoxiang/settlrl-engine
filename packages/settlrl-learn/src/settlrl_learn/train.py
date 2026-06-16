"""Minimal training loop: full-batch SGD over a params pytree.

Enough to verify gradients flow end-to-end and to fit small models; the
self-play data pipeline and a real optimiser arrive with Stage 1.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from settlrl_learn.model import MLPParams, mlp

Loss = Callable[
    [MLPParams, Float[Array, "n features"], Float[Array, "n"]], Float[Array, ""]
]


def value_loss(
    params: MLPParams, x: Float[Array, "n features"], y: Float[Array, "n"]
) -> Float[Array, ""]:
    """Logistic loss of the scalar head against win labels ``y`` in {0, 1}
    (the head is a win-probability logit, matching the search's
    ``tanh(v / s) = 2P - 1`` reading)."""
    v = jax.vmap(mlp, in_axes=(None, 0))(params, x)[:, 0]
    return jnp.mean(jax.nn.softplus(v) - y * v)


def fit(
    params: MLPParams,
    x: Float[Array, "n features"],
    y: Float[Array, "n"],
    *,
    loss: Loss = value_loss,
    steps: int = 200,
    lr: float = 1e-2,
) -> tuple[MLPParams, Float[Array, ""]]:
    """Full-batch SGD; returns the fitted params and the final loss."""

    @jax.jit
    def step(p: MLPParams) -> tuple[MLPParams, Float[Array, ""]]:
        val, grads = jax.value_and_grad(loss)(p, x, y)
        return jax.tree.map(lambda a, g: a - lr * g, p, grads), val

    final = jnp.asarray(0.0)
    for _ in range(steps):
        params, final = step(params)
    return params, final
