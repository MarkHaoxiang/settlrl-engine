"""Symmetry contracts for the board architectures.

Two symmetries leave a position's *meaning* unchanged, so a sound representation
must score it identically:

- **player relabeling** -- swap who is who (and the perspective); the
  player-relative featurization is exactly invariant, so any model is too;
- **board rotation/reflection** -- the 12 hexagon automorphisms; a structure-
  aware readout (GNN message passing, DeepSet pooling) is invariant, while the
  structure-blind flat MLP is not (it reads nodes in fixed vertex order).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _symmetry import apply_symmetry, board_symmetries, relabel_players
from settlrl_engine.board import Board
from settlrl_engine.env import BatchedSettlrlEnv
from settlrl_learn.architectures import DeepSetModel, GNNModel, MLPModel
from settlrl_learn.graph import board_sample

_OUT, _W = 4, 8


def _mid_game(n_players: int, steps: int = 150, seed: int = 7) -> Board:
    """A single-game position with real ownership (past setup), random play."""
    env = BatchedSettlrlEnv(
        batch_size=1, n_players=n_players, seed=seed, auto_reset=False
    )
    key = jax.random.key(seed)
    for _ in range(steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    layout = jax.tree.map(lambda x: x[0], env.board[0])
    state = jax.tree.map(lambda x: x[0], env.board[1])
    return layout, state


def _gnn() -> GNNModel:
    return GNNModel(jax.random.key(0), out_dim=_OUT, width=_W, depth=1, layers=2)


def test_board_symmetries_form_the_order_6_group() -> None:
    # The harbors are 3-fold symmetric, so the full board's group is order 6.
    syms = board_symmetries()
    assert len(syms) == 6
    keys = {s.vertices.tobytes() for s in syms}
    assert len(keys) == 6  # all distinct
    assert any(np.array_equal(s.vertices, np.arange(s.vertices.shape[0])) for s in syms)


def test_player_relabel_leaves_features_and_gnn_invariant() -> None:
    layout, state = _mid_game(4)
    gnn = _gnn()
    perms = [np.array([1, 0, 2, 3]), np.array([1, 2, 3, 0]), np.array([3, 2, 1, 0])]
    for p in range(4):
        base = board_sample(layout, state, jnp.int32(p))
        base_out = gnn(base)
        for perm in perms:
            other = board_sample(
                layout, relabel_players(state, perm), jnp.int32(perm[p])
            )
            # the featurization is exactly relabeling-invariant ...
            for a, b in zip(base, other, strict=True):
                assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
            # ... so the model is too.
            assert np.allclose(np.asarray(base_out), np.asarray(gnn(other)), atol=1e-4)


def test_board_symmetry_leaves_structured_models_invariant() -> None:
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    base = board_sample(layout, state, p)
    key = jax.random.key(1)
    structured = (
        _gnn(),
        DeepSetModel(key, out_dim=_OUT, width=_W, depth=1),
    )
    rotated = [apply_symmetry(layout, state, sym) for sym in board_symmetries()]
    for model in structured:
        base_out = np.asarray(model(base))
        for l2, s2 in rotated:
            out = np.asarray(model(board_sample(l2, s2, p)))
            assert np.allclose(base_out, out, atol=1e-4)


def test_flat_mlp_is_not_symmetry_invariant() -> None:
    # The contrast that motivates structure: reordering nodes moves the flat
    # input vector, so the structure-blind MLP cannot be rotation-invariant.
    layout, state = _mid_game(2)
    p = jnp.int32(0)
    flat = MLPModel(
        jax.random.key(0), out_dim=_OUT, width=_W, depth=1, engineered=False
    )
    base = np.asarray(flat(board_sample(layout, state, p)))
    moved = max(
        float(np.abs(np.asarray(flat(board_sample(l2, s2, p))) - base).max())
        for l2, s2 in (apply_symmetry(layout, state, sym) for sym in board_symmetries())
    )
    assert moved > 1e-3
