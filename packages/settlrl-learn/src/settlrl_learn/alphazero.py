"""AlphaZero training: a flashbax replay buffer + the policy/value loss step.

Consumes :class:`~settlrl_learn.selfplay.SelfPlaySamples` and improves an
:class:`~settlrl_learn.model.AZParams` net by imitating the search's policy
(cross-entropy) and predicting the game outcome (value logistic). Composable:
the experiment owns the loop (self-play -> add -> train -> repeat); these are
the pieces.

A training-side module (optax/flashbax): not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float
from settlrl_engine.env import N_FLAT

from settlrl_learn.model import AZParams, az_forward
from settlrl_learn.selfplay import SelfPlaySamples


class Batch(NamedTuple):
    """Training items: a position's features, the search's improved-policy
    target, and the game-outcome value label. A leading sample axis when
    batched."""

    features: Float[Array, "*b feat"]
    policy: Float[Array, "*b flat"]
    value: Float[Array, "*b"]


TrainStep = Callable[
    [AZParams, optax.OptState, Batch],
    tuple[AZParams, optax.OptState, dict[str, Float[Array, ""]]],
]


def az_loss(
    params: AZParams, batch: Batch, value_weight: float
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """AlphaZero loss: policy cross-entropy (against the search target) plus
    ``value_weight`` x the value logistic loss (win/loss)."""
    vs, logits = jax.vmap(az_forward, in_axes=(None, 0))(params, batch.features)
    logp = jax.nn.log_softmax(logits, axis=-1)
    policy_loss = -jnp.mean(jnp.sum(batch.policy * logp, axis=-1))
    value_loss = jnp.mean(jax.nn.softplus(vs) - batch.value * vs)
    total = policy_loss + value_weight * value_loss
    return total, {"policy_loss": policy_loss, "value_loss": value_loss}


def make_train_step(
    optimizer: optax.GradientTransformation, value_weight: float = 1.0
) -> TrainStep:
    """A jitted adamw-style update over ``AZParams`` for one minibatch."""
    grad_fn = jax.value_and_grad(az_loss, has_aux=True)

    @jax.jit
    def step(
        params: AZParams, opt_state: optax.OptState, batch: Batch
    ) -> tuple[AZParams, optax.OptState, dict[str, Float[Array, ""]]]:
        (total, aux), grads = grad_fn(params, batch, value_weight)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = cast(AZParams, optax.apply_updates(params, updates))
        return params, opt_state, {"loss": total, **aux}

    return step


# --- replay buffer (flashbax item buffer; samples are independent) ---


def replay_buffer(*, max_size: int, min_size: int, batch_size: int) -> Any:
    """A flashbax item buffer sized for AlphaZero replay (add whole self-play
    batches, sample independent minibatches of ``batch_size``)."""
    return fbx.make_item_buffer(
        max_length=max_size,
        min_length=min_size,
        sample_batch_size=batch_size,
        add_batches=True,
    )


def init_replay(buffer: Any, feature_dim: int) -> Any:
    """An empty buffer state shaped by one zero item."""
    empty = Batch(
        jnp.zeros((feature_dim,), jnp.float32),
        jnp.zeros((N_FLAT,), jnp.float32),
        jnp.float32(0.0),
    )
    return buffer.init(empty)


def add_samples(buffer: Any, state: Any, samples: SelfPlaySamples) -> Any:
    """Add a self-play batch (the oldest items age out at ``max_size``)."""
    batch = Batch(
        jnp.asarray(samples.features, jnp.float32),
        jnp.asarray(samples.policy, jnp.float32),
        jnp.asarray(samples.value, jnp.float32),
    )
    return buffer.add(state, batch)


def sample_batch(buffer: Any, state: Any, key: jax.Array) -> Batch:
    """A training minibatch from the buffer."""
    return cast(Batch, buffer.sample(state, key).experience)


def train(
    params: AZParams,
    opt_state: optax.OptState,
    buffer: Any,
    state: Any,
    *,
    step: TrainStep,
    n_steps: int,
    key: jax.Array,
) -> tuple[AZParams, optax.OptState, dict[str, Float[Array, ""]]]:
    """Run ``n_steps`` minibatch updates; return the params, optimiser state, and
    the last step's metrics."""
    metrics: dict[str, Float[Array, ""]] = {}
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        params, opt_state, metrics = step(
            params, opt_state, sample_batch(buffer, state, k)
        )
    return params, opt_state, metrics
