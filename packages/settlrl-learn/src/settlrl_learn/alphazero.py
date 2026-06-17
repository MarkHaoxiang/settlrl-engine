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
from pathlib import Path
from typing import Any, NamedTuple, cast

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float
from settlrl_agents import POLICIES, BeliefSpec, evaluate
from settlrl_agents.search import make_search
from settlrl_engine.env import N_FLAT

from settlrl_learn.features import FEATURE_DIM
from settlrl_learn.model import AZParams, az_forward, make_az
from settlrl_learn.selfplay import SelfPlaySamples, self_play
from settlrl_learn.train_state import TrainState, load_train_state, save_train_state


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


# --- arena + the iteration loop (the experiment composes these) ---


def arena(
    params: AZParams,
    *,
    n_games: int = 40,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    batch_size: int = 16,
    seed: int = 0,
) -> float:
    """The net's win rate vs. ``lookahead(heuristic)``, seat-swapped at 2p — the
    Stage-1 gate (a learned value worth shipping beats the hand-tuned one)."""
    value_fn, prior_fn = make_az(params)
    net = make_search(
        value_fn,
        prior=prior_fn,
        value_scale=2.0,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
    )
    net_spec = BeliefSpec(lambda: net, frozenset((2,)))
    base = POLICIES["lookahead"]
    half = max(1, n_games // 2)
    r1 = evaluate([net_spec, base], n_episodes=half, batch_size=batch_size, seed=seed)
    r2 = evaluate(
        [base, net_spec], n_episodes=half, batch_size=batch_size, seed=seed + 1
    )
    episodes = int(r1.episodes + r2.episodes)
    return float(r1.wins[0] + r2.wins[1]) / max(episodes, 1)


def learn(
    params: AZParams,
    *,
    n_iterations: int,
    selfplay_samples: int,
    selfplay_batch: int = 16,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    temperature: float = 1.0,
    buffer_max: int = 50_000,
    buffer_min: int = 256,
    batch_size: int = 256,
    train_steps: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    value_weight: float = 1.0,
    arena_games: int = 0,
    arena_every: int = 1,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 1,
    resume_from: str | Path | None = None,
    on_iter: Callable[[int, dict[str, float], TrainState], None] | None = None,
) -> TrainState:
    """One full AlphaZero loop: each iteration self-plays, buffers, and trains;
    every ``arena_every`` iterations (when ``arena_games`` > 0) it scores the net
    vs. ``lookahead(heuristic)``. Per-iteration RNG derives from ``seed`` and the
    iteration index, so ``resume_from`` (a prior ``trainstate`` checkpoint)
    continues the run bit-exactly; the full-state checkpoint is written to
    ``checkpoint_dir`` every ``checkpoint_every`` iterations.
    ``on_iter(i, metrics, state)`` runs after each iteration. Returns the final
    :class:`TrainState`."""
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    buffer = replay_buffer(
        max_size=buffer_max, min_size=buffer_min, batch_size=batch_size
    )
    step = make_train_step(optimizer, value_weight)

    fresh = TrainState(
        params=params,
        opt_state=optimizer.init(params),
        buffer_state=init_replay(buffer, FEATURE_DIM),
        iteration=jnp.int32(0),
        best=jnp.float32(-1.0),
    )
    ckpt = Path(checkpoint_dir) / "trainstate" if checkpoint_dir is not None else None
    state = load_train_state(resume_from, fresh) if resume_from is not None else fresh
    params, opt_state, buf_state = state.params, state.opt_state, state.buffer_state
    best = float(state.best)

    for i in range(int(state.iteration), n_iterations):
        samples = self_play(
            params,
            n_samples=selfplay_samples,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            batch_size=selfplay_batch,
            temperature=temperature,
            seed=seed + 1 + i,
        )
        buf_state = add_samples(buffer, buf_state, samples)
        metrics: dict[str, float] = {"samples": float(samples.value.shape[0])}
        if bool(buffer.can_sample(buf_state)):
            params, opt_state, m = train(
                params,
                opt_state,
                buffer,
                buf_state,
                step=step,
                n_steps=train_steps,
                key=jax.random.key(seed + 10_000 + i),
            )
            metrics.update({k: float(v) for k, v in m.items()})
        if arena_games and (i + 1) % arena_every == 0:
            winrate = arena(
                params,
                n_games=arena_games,
                num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
                seed=seed + 20_000 + i,
            )
            metrics["arena_winrate"] = winrate
            best = max(best, winrate)
        state = TrainState(
            params, opt_state, buf_state, jnp.int32(i + 1), jnp.float32(best)
        )
        if ckpt is not None and (i + 1) % checkpoint_every == 0:
            save_train_state(ckpt, state)
        if on_iter is not None:
            on_iter(i, metrics, state)
    return state
