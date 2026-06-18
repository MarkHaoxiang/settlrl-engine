"""Behaviour contracts for the Single-Observer ISMCTS search (``search.ismcts``).

Correctness invariants only, at tiny budgets: the move is always legal, the
returned weights are a distribution supported on the legal set, the search is
reproducible from its key, and a self-played game reaches a terminal. The
per-determinization legality property -- the search only ever returns an action
legal in the true position -- is what the legality/support tests pin (an illegal
return would mean the descent leaked an action illegal under the real board).

Strength vs. the mctx search is a *match*, not a unit test: see
``experiments``/the comparison harness, gated at n the SE can support.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from settlrl_agents.search.ismcts import ismcts_move, ismcts_weights
from settlrl_agents.value import heuristic_value
from settlrl_engine.belief import BeliefView
from settlrl_engine.env import BatchedSettlrlEnv, flat_to_action


def _position(seed: int, steps: int, n_players: int = 2) -> tuple:
    """A single-game mid-game position with the acting seat's belief view."""
    env = BatchedSettlrlEnv(
        batch_size=1, seed=seed, n_players=n_players, track_beliefs=True
    )
    env.rollout(jax.random.key(seed), steps)
    layout = jax.tree.map(lambda x: x[0], env.board[0])
    p = int(env.agent_selection[0])
    view: BeliefView = jax.tree.map(lambda x: x[0], env.belief_view(p))
    mask = np.asarray(env.flat_mask()[0])
    return layout, view, p, mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_move_is_legal(seed: int) -> None:
    layout, view, p, mask = _position(seed, steps=100 + seed * 20)
    if mask.sum() == 0:
        pytest.skip("no legal move (stalled lane)")
    a = ismcts_move(
        jax.random.key(seed), layout, view, jnp.int32(p), jnp.asarray(mask),
        value=heuristic_value, num_simulations=12,
    )  # fmt: skip
    assert mask[a] > 0


def test_weights_are_a_legal_distribution() -> None:
    layout, view, p, mask = _position(7, steps=130)
    w = ismcts_weights(
        jax.random.key(1), layout, view, jnp.int32(p), jnp.asarray(mask),
        value=heuristic_value, num_simulations=16,
    )  # fmt: skip
    assert np.all(w >= 0.0)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert float(w[mask == 0].sum()) == 0.0  # support is exactly the legal set


def test_reproducible_from_key() -> None:
    layout, view, p, mask = _position(3, steps=110)
    a1 = ismcts_move(
        jax.random.key(9), layout, view, jnp.int32(p), jnp.asarray(mask),
        value=heuristic_value, num_simulations=12,
    )  # fmt: skip
    a2 = ismcts_move(
        jax.random.key(9), layout, view, jnp.int32(p), jnp.asarray(mask),
        value=heuristic_value, num_simulations=12,
    )  # fmt: skip
    assert a1 == a2


def test_visits_concentrate_above_uniform() -> None:
    # A healthy search is neither degenerate (all mass on one action) nor a
    # round-robin: the top action takes clearly more than a uniform share of the
    # visits, while more than one action is explored.
    layout, view, p, mask = _position(5, steps=120)
    n_legal = int(mask.sum())
    if n_legal < 4:
        pytest.skip("trivial decision")
    w = ismcts_weights(
        jax.random.key(2), layout, view, jnp.int32(p), jnp.asarray(mask),
        value=heuristic_value, num_simulations=48,
    )  # fmt: skip
    assert float(w.max()) > 1.5 / n_legal  # concentrates above uniform
    assert int((w > 0).sum()) > 1  # but explores more than one action


@pytest.mark.slow
def test_self_play_completes_a_game() -> None:
    env = BatchedSettlrlEnv(batch_size=1, seed=4, n_players=2, track_beliefs=True)
    key = jax.random.key(0)
    for _ in range(400):
        if bool(env.terminations[0].any()):
            break
        layout = jax.tree.map(lambda x: x[0], env.board[0])
        p = int(env.agent_selection[0])
        view = jax.tree.map(lambda x: x[0], env.belief_view(p))
        mask = np.asarray(env.flat_mask()[0])
        if mask.sum() == 0:
            break
        key, k = jax.random.split(key)
        mv = ismcts_move(
            k, layout, view, jnp.int32(p), jnp.asarray(mask),
            value=heuristic_value, num_simulations=8,
        )  # fmt: skip
        assert mask[mv] > 0
        env.step(*flat_to_action(jnp.asarray([mv])))
    assert bool(env.terminations[0].any())
