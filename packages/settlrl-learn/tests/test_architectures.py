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
import pytest
from _symmetry import (
    action_permutation,
    apply_symmetry,
    board_symmetries,
    relabel_players,
)
from settlrl_agents.internal.rows import ROW_TYPE
from settlrl_engine.board import Board
from settlrl_engine.env import N_FLAT, ActionType, BatchedSettlrlEnv
from settlrl_learn.architectures import DeepSetModel, GNNModel, MLPModel
from settlrl_learn.azgnn import AZGraphNet, az_gnn_loss
from settlrl_learn.graph import board_sample
from settlrl_learn.graphnet import PRESETS, GraphNet

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


@pytest.mark.parametrize("preset", ["gn_multi", "gn_graphnorm", "gn_gat", "gn_full"])
def test_graphnet_presets_are_invariant(preset: str) -> None:
    # The configurable GraphNet keeps both invariances across every lever
    # (attention, GraphNorm spanning the node axis, the global node, JK) -- it
    # uses only symmetric aggregations and relative features, no absolute PE.
    layout, state = _mid_game(4)
    cfg = PRESETS[preset]._replace(width=8, layers=2, head_depth=1)
    model = GraphNet(jax.random.key(0), out_dim=_OUT, cfg=cfg)
    base = np.asarray(model(board_sample(layout, state, jnp.int32(0))))
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        rot = np.asarray(model(board_sample(l2, s2, jnp.int32(0))))
        assert np.allclose(base, rot, atol=1e-3)
    perm = np.array([1, 2, 3, 0])
    relabeled = board_sample(layout, relabel_players(state, perm), jnp.int32(perm[0]))
    assert np.allclose(base, np.asarray(model(relabeled)), atol=1e-3)


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


def _aznet() -> AZGraphNet:
    cfg = PRESETS["gn_global"]._replace(width=16, layers=2, head_depth=1)
    return AZGraphNet(jax.random.key(0), cfg)


def test_aznet_value_invariant_policy_equivariant_under_board_symmetry() -> None:
    # The factored value+policy net: the value is invariant under a board
    # symmetry, and the policy is *equivariant* -- a settlement-at-v action maps
    # to settlement-at-(sigma v), road-at-e to road-at-(sigma e), robber-tile-t
    # to sigma(t) -- so policy(sigma . board)[action_permutation] == policy(board).
    layout, state = _mid_game(4)
    net = _aznet()
    p = jnp.int32(0)
    vv, pp = net(board_sample(layout, state, p))
    v0, pol0 = np.asarray(vv), np.asarray(pp)
    for sym in board_symmetries():
        l2, s2 = apply_symmetry(layout, state, sym)
        v, pol = net(board_sample(l2, s2, p))
        assert np.allclose(np.asarray(v), v0, atol=1e-3)  # value invariant
        perm = action_permutation(sym)
        assert np.allclose(np.asarray(pol)[perm], pol0, atol=1e-3)  # policy equivariant


def test_aznet_value_and_policy_invariant_under_player_relabel() -> None:
    layout, state = _mid_game(4)
    net = _aznet()
    vv, pp = net(board_sample(layout, state, jnp.int32(0)))
    v0, pol0 = np.asarray(vv), np.asarray(pp)
    perm = np.array([1, 2, 3, 0])
    relabeled = board_sample(layout, relabel_players(state, perm), jnp.int32(perm[0]))
    v, pol = net(relabeled)
    assert np.allclose(np.asarray(v), v0, atol=1e-3)  # value invariant
    # Spatial (vertex/edge/tile, with the robber victim collapsed to no-steal vs
    # steal) and non-player-indexed actions are relabel-invariant; PROPOSE_TRADE's
    # partner index inherits the opponent-collapse limitation, so it is excluded.
    keep = np.asarray(ROW_TYPE) != int(ActionType.PROPOSE_TRADE)
    assert np.allclose(np.asarray(pol)[keep], pol0[keep], atol=1e-3)


def test_aznet_runs_on_random_play_boards() -> None:
    # Fast net check (no MCTS): drive the board with a random policy and run the
    # net forward -- correct shapes, finite values.
    net = _aznet()
    env = BatchedSettlrlEnv(batch_size=8, n_players=2, seed=0)
    fwd = jax.jit(jax.vmap(lambda lo, st, p: net(board_sample(lo, st, p))))
    key = jax.random.key(0)
    for _ in range(40):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
        lo, st = env.board
        v, pol = fwd(lo, st, env.agent_selection)
        assert v.shape == (8,) and pol.shape == (8, N_FLAT)
        assert bool(jnp.isfinite(v).all()) and bool(jnp.isfinite(pol).all())


def test_az_gnn_loss_masked_is_finite() -> None:
    # The masked policy CE must stay finite (no 0 * -inf on illegal slots) for a
    # legal-supported target -- checked on real random-play boards + their masks.
    net = _aznet()
    env = BatchedSettlrlEnv(batch_size=4, n_players=2, seed=1)
    key = jax.random.key(0)
    for _ in range(25):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    lo, st = env.board
    mask = jnp.asarray(env.flat_mask(), jnp.float32)  # (4, N_FLAT)
    samples = jax.vmap(board_sample)(lo, st, env.agent_selection)
    target = mask / jnp.clip(mask.sum(-1, keepdims=True), 1.0)  # uniform over legal
    loss, aux = az_gnn_loss(net, samples, target, jnp.zeros(4), mask)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.isfinite(v)) for v in aux.values())
