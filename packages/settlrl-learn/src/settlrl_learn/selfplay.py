"""Self-play data generation for the AlphaZero loop.

Drives batched n-player self-play with the current net guiding the
re-determinizing search and records, per acting move, the seat's true-board
features, the search's improved-policy target, and the eventual game outcome.
Features are on the *true* board (no hidden state in the net's inputs), so the
net learns the belief-averaged value; determinization stays inside the search.

A training-side module: not imported by ``settlrl_learn``'s package root, so the
shipped-model path stays import-light.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from settlrl_agents.search import make_search_weights
from settlrl_engine.belief import belief_view
from settlrl_engine.env import BatchedSettlrlEnv, flat_to_action

from settlrl_learn.features import features
from settlrl_learn.model import AZParams, make_az


class SelfPlaySamples(NamedTuple):
    """A batch of training positions, stacked on a leading sample axis."""

    features: np.ndarray  # (n, FEATURE_DIM)
    policy: np.ndarray  # (n, N_FLAT) — the search's improved policy
    value: np.ndarray  # (n,) — 1.0 if the acting seat won that game, else 0.0


def _sample_moves(
    key: jax.Array,
    weights: Float[Array, "batch flat"],
    mask: Int[Array, "batch flat"],
    temperature: float,
) -> Int[Array, "batch"]:
    """One legal move per lane from the masked improved policy (argmax at
    ``temperature`` 0, else a tempered categorical draw)."""
    if temperature <= 0.0:
        return jnp.argmax(jnp.where(mask, weights, -jnp.inf), axis=-1)
    logits = jnp.where(mask, jnp.log(jnp.clip(weights, 1e-8)) / temperature, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1)


def self_play(
    params: AZParams,
    *,
    n_samples: int,
    n_players: int = 2,
    num_simulations: int = 64,
    num_trees: int = 1,
    max_num_considered_actions: int = 16,
    batch_size: int = 16,
    temperature: float = 1.0,
    seed: int = 0,
) -> SelfPlaySamples:
    """Collect at least ``n_samples`` self-play positions under the net ``params``.

    The net guides a re-determinizing search (``value_scale=2`` for the win-logit
    value head); each move is drawn from the improved policy at ``temperature``.
    Positions from finished games are credited with the acting seat's win (1) or
    loss (0); positions in games still running at the budget are discarded.
    """
    value_fn, prior_fn = make_az(params)
    weights_fn = make_search_weights(
        value_fn,
        prior=prior_fn,
        value_scale=2.0,
        num_simulations=num_simulations,
        num_trees=num_trees,
        max_num_considered_actions=max_num_considered_actions,
    )
    search = jax.jit(jax.vmap(weights_fn, in_axes=(0, 0, 0, 0, 0)))
    view_of = jax.jit(jax.vmap(belief_view, in_axes=(0, 0, 0)))
    feat_of = jax.jit(jax.vmap(features, in_axes=(0, 0, 0)))

    env = BatchedSettlrlEnv(
        batch_size=batch_size,
        seed=seed,
        reward="sparse",
        n_players=n_players,
        track_beliefs=True,
    )
    # Per-lane (features, policy, acting seat) awaiting its game's outcome.
    pending: list[list[tuple[np.ndarray, np.ndarray, int]]] = [
        [] for _ in range(batch_size)
    ]
    feats: list[np.ndarray] = []
    pols: list[np.ndarray] = []
    vals: list[float] = []
    key = jax.random.key(seed)

    while len(vals) < n_samples:
        layout, state = env.board
        beliefs = env.beliefs
        assert beliefs is not None  # track_beliefs=True
        sel = jnp.asarray(env.agent_selection)
        mask = env.flat_mask()
        view = view_of(state, beliefs, sel)
        key, k_search, k_move = jax.random.split(key, 3)
        weights = search(
            jax.random.split(k_search, batch_size), layout, view, sel, mask
        )
        move = _sample_moves(k_move, weights, mask, temperature)

        f_np = np.asarray(feat_of(layout, state, sel))
        w_np = np.asarray(weights)
        sel_np = np.asarray(sel)
        for lane in range(batch_size):
            pending[lane].append((f_np[lane], w_np[lane], int(sel_np[lane])))

        env.step(*flat_to_action(move))
        rewards = np.asarray(env.rewards)
        done = np.asarray(env.terminations).any(axis=1)
        for lane in np.flatnonzero(done).tolist():
            for f_l, w_l, seat in pending[lane]:
                feats.append(f_l)
                pols.append(w_l)
                vals.append(float(rewards[lane, seat] > 0))
            pending[lane] = []

    return SelfPlaySamples(
        features=np.stack(feats),
        policy=np.stack(pols),
        value=np.asarray(vals, np.float32),
    )
