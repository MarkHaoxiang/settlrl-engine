"""Net-agnostic self-play data generation for the training loop.

Drives batched n-player self-play with a search ``weights_fn`` (the net's, or a
fixed teacher's) and records, per acting move, the backend's observation of the
*true* board, the search's improved-policy target, the legality mask, and the
eventual game outcome. Features are on the true board (no hidden state in the
net's inputs), so the net learns the belief-averaged value; determinization
stays inside the search.

``setup_fn`` (when given) plays the **setup phase** (initial placements) with a
fixed policy instead of the net, and those positions are *not recorded* -- so the
net's value/policy only ever train on (and act in) the main game loop. The setup
placements are rare, high-leverage, and structurally distinct; handing them to a
strong fixed policy keeps a weak net's bad opening from dooming every game.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from settlrl_agents.policy import BeliefPolicy
from settlrl_agents.search import PolicyWeights
from settlrl_engine.belief import belief_view
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, GamePhase
from settlrl_engine.env import BatchedSettlrlEnv, flat_to_action

ObserveFn = Callable[[BoardLayout, BoardState, Array], dict[str, Array]]
Samples = dict[str, np.ndarray]
"""A batch of training positions: the backend's observation keys plus ``policy``,
``mask``, and ``value``, each stacked on a leading sample axis."""


def _sample_moves(key: Array, weights: Array, mask: Array, temperature: float) -> Array:
    """One legal move per lane from the masked improved policy (argmax at
    ``temperature`` 0, else a tempered categorical draw)."""
    if temperature <= 0.0:
        return jnp.argmax(jnp.where(mask, weights, -jnp.inf), axis=-1)
    logits = jnp.where(mask, jnp.log(jnp.clip(weights, 1e-8)) / temperature, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1)


def self_play(
    weights_fn: PolicyWeights,
    observe: ObserveFn,
    *,
    n_samples: int,
    setup_fn: BeliefPolicy | None = None,
    n_players: int = 2,
    batch_size: int = 16,
    temperature: float = 1.0,
    seed: int = 0,
    max_steps: int = 100_000,
    max_game_len: int = 800,
) -> Samples:
    """Collect >= ``n_samples`` self-play positions, the moves and policy targets
    drawn from ``weights_fn``. Positions from finished games are credited with the
    acting seat's win (1) / loss (0); unfinished games are discarded.

    ``max_steps`` caps the env-step budget and ``max_game_len`` each lane's
    retained pending positions -- a cold/degenerate net can drag a game out
    indefinitely, so without these the pending buffer grows unbounded. A capped
    lane keeps its most recent positions."""
    search = jax.jit(jax.vmap(weights_fn, in_axes=(0, 0, 0, 0, 0)))
    setup_search = (
        jax.jit(jax.vmap(setup_fn, in_axes=(0, 0, 0, 0, 0)))
        if setup_fn is not None
        else None
    )
    view_of = jax.jit(jax.vmap(belief_view, in_axes=(0, 0, 0)))
    observe_of = jax.jit(jax.vmap(observe, in_axes=(0, 0, 0)))

    env = BatchedSettlrlEnv(
        batch_size=batch_size, seed=seed, reward="sparse",
        n_players=n_players, track_beliefs=True,
    )  # fmt: skip
    pending: list[list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]]] = [
        [] for _ in range(batch_size)
    ]
    out: dict[str, list[np.ndarray]] = {}
    trailing: dict[str, tuple[int, ...]] = {}
    vals: list[float] = []
    key = jax.random.key(seed)

    for _step in range(max_steps):
        if len(vals) >= n_samples:
            break
        layout, state = env.board
        beliefs = env.beliefs
        assert beliefs is not None  # track_beliefs=True
        sel = jnp.asarray(env.agent_selection)
        mask = env.flat_mask()
        view = view_of(state, beliefs, sel)
        key, k_search, k_move, k_setup = jax.random.split(key, 4)
        weights = search(
            jax.random.split(k_search, batch_size), layout, view, sel, mask
        )
        move = _sample_moves(k_move, weights, mask, temperature)
        # Setup-phase lanes play (unrecorded) via the fixed setup policy.
        is_setup = (
            np.asarray(state.phase <= int(GamePhase.SETUP_ROAD))
            if setup_search is not None
            else np.zeros(batch_size, bool)
        )
        if setup_search is not None and is_setup.any():
            setup_move = setup_search(
                jax.random.split(k_setup, batch_size), layout, view, sel, mask
            )
            move = jnp.where(jnp.asarray(is_setup), setup_move, move)

        obs = {k: np.asarray(v) for k, v in observe_of(layout, state, sel).items()}
        w_np, sel_np, m_np = np.asarray(weights), np.asarray(sel), np.asarray(mask)
        if not trailing:  # capture per-key trailing shapes once, for the empty case
            trailing = {k: v.shape[1:] for k, v in obs.items()}
            trailing["policy"], trailing["mask"] = w_np.shape[1:], m_np.shape[1:]
            out = {k: [] for k in (*trailing,)}
        for lane in range(batch_size):
            if is_setup[lane]:  # the net does not train on setup positions
                continue
            row = (
                {k: obs[k][lane] for k in obs},
                w_np[lane],
                m_np[lane],
                int(sel_np[lane]),
            )
            pending[lane].append(row)
            if len(pending[lane]) > max_game_len:
                del pending[lane][:-max_game_len]

        env.step(*flat_to_action(move))
        rewards = np.asarray(env.rewards)
        for lane in np.flatnonzero(np.asarray(env.terminations).any(axis=1)).tolist():
            for obs_l, pol_l, mask_l, seat in pending[lane]:
                for k, v in obs_l.items():
                    out[k].append(v)
                out["policy"].append(pol_l)
                out["mask"].append(mask_l)
                vals.append(float(rewards[lane, seat] > 0))
            pending[lane] = []

    if not vals:  # no game finished within the budget (a degenerate cold net)
        empty: Samples = {k: np.zeros((0, *trailing[k]), np.float32) for k in trailing}
        empty["value"] = np.zeros((0,), np.float32)
        return empty
    samples: Samples = {k: np.stack(out[k]) for k in out}
    samples["value"] = np.asarray(vals, np.float32)
    return samples


def index(samples: Samples, idx: np.ndarray) -> Samples:
    return {k: v[idx] for k, v in samples.items()}


def concat(a: Samples, b: Samples, cap: int) -> Samples:
    return {k: np.concatenate([a[k], b[k]])[-cap:] for k in a}
