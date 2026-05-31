"""Equivalence tests: the traceable placement-legality rules must match the
``catan-reference`` oracle (via ``tests.conversion``) across randomized board
occupancy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.mechanics import placement
from catan_engine.board.layout import N_EDGES, N_VERTICES
from catan_engine.board.resources import N_PLAYERS
from catan_engine.board.state import BoardState, make_board_state
from tests import conversion as reference

_dist = jax.jit(placement.distance_rule_ok)
_conn = jax.jit(placement.settlement_connected)
_road = jax.jit(placement.road_placeable)


def _state(seed: int) -> tuple[BoardState, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    edge_road = rng.choice(
        [0, 1, 2, 3, 4], size=N_EDGES, p=[0.55, 0.15, 0.12, 0.1, 0.08]
    ).astype(np.uint8)
    vertex_owner = rng.choice(
        [0, 1, 2, 3, 4], size=N_VERTICES, p=[0.7, 0.1, 0.08, 0.07, 0.05]
    ).astype(np.uint8)
    state = make_board_state(1, key=jax.random.key(0))._replace(
        edge_road=jnp.asarray(edge_road)[None],
        vertex_owner=jnp.asarray(vertex_owner)[None],
    )
    return state, edge_road, vertex_owner


def test_distance_rule_matches_reference() -> None:
    for seed in range(8):
        state, _, vertex_owner = _state(seed)
        vo = jnp.asarray(vertex_owner)
        for v in range(N_VERTICES):
            ref = reference.distance_rule_ok(state, v, 0)
            got = bool(_dist(vo, jnp.int32(v)))
            assert got == ref, f"seed={seed} v={v}: vec={got} ref={ref}"


def test_settlement_connected_matches_reference() -> None:
    for seed in range(8):
        state, edge_road, _ = _state(seed)
        er = jnp.asarray(edge_road)
        for p in range(N_PLAYERS):
            for v in range(N_VERTICES):
                ref = reference.settlement_connected(state, p, v, 0)
                got = bool(_conn(er, jnp.int32(p), jnp.int32(v)))
                assert got == ref, f"seed={seed} p={p} v={v}: vec={got} ref={ref}"


def test_road_placeable_matches_reference() -> None:
    for seed in range(8):
        state, edge_road, vertex_owner = _state(seed)
        er, vo = jnp.asarray(edge_road), jnp.asarray(vertex_owner)
        for p in range(N_PLAYERS):
            for e in range(N_EDGES):
                ref = reference.road_placeable(state, p, e, 0)
                got = bool(_road(er, vo, jnp.int32(p), jnp.int32(e)))
                assert got == ref, f"seed={seed} p={p} e={e}: vec={got} ref={ref}"
