"""Shared fixtures for the per-action tests.

Composes the board.py construction helpers with the vectorized actions to build
legal mid-game positions. Boards are batch=1 (single game) so success cases read
out of lane 0; the vmapped actions accept them directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import BuildRoad
from catan_engine.board import (
    Board,
    give,
    make_board,
    place_road,
    place_settlement,
    replicate,
    to_main,
)
from catan_engine.layout import NO_INDEX, N_EDGES
from catan_engine.rules_vec import EDGE_V, V_NBR

_EDGE_V = np.asarray(EDGE_V)
_V_NBR = np.asarray(V_NBR)


def code(result: jax.Array, lane: int = 0) -> str:
    """Human-readable ActionResult for lane ``lane`` (OK / INVALID / DONE)."""
    return str(ActionResult(int(result[lane])))


def fmt(result: jax.Array, **fields: object) -> str:
    """A stable one-field-per-line snapshot for expect tests."""
    lines = [f"result={code(result)}"]
    lines += [f"{k}={v}" for k, v in fields.items()]
    return "\n".join(lines)


def _edge_between(a: int, b: int) -> int:
    for e in range(N_EDGES):
        if {int(_EDGE_V[e, 0]), int(_EDGE_V[e, 1])} == {a, b}:
            return e
    raise AssertionError(f"no edge between {a} and {b}")


def first_legal_edge(board: Board) -> int:
    """Lowest-index edge where BuildRoad is currently legal (single game)."""
    batched = replicate(board, N_EDGES)
    avail = np.asarray(
        BuildRoad().is_available(batched, jnp.arange(N_EDGES, dtype=jnp.int32))
    )
    idx = np.where(avail)[0]
    assert idx.size, "no legal road edge available"
    return int(idx[0])


def road_fixture() -> tuple[Board, int]:
    """MAIN board where player 0 owns a settlement; returns (board, target edge)."""
    board = to_main(make_board())
    board = place_settlement(board, 0, 0)
    board = give(board, 0, [0, 0, 1, 1, 0])  # exactly one road's worth
    return board, first_legal_edge(board)


def settlement_fixture() -> tuple[Board, int]:
    """MAIN board with a 2-road spur off vertex 0; returns (board, legal vertex).

    Vertices adjacent to a settlement are distance-blocked, so the legal target
    sits two edges away (v0 -> w -> x), connected by the player's road network.
    """
    board = to_main(make_board())
    v0 = 0
    w = int(_V_NBR[v0][0])
    x = next(int(n) for n in _V_NBR[w] if n != NO_INDEX and int(n) != v0)
    board = place_settlement(board, 0, v0)
    board = place_road(board, 0, _edge_between(v0, w))
    board = place_road(board, 0, _edge_between(w, x))
    board = give(board, 0, [1, 1, 1, 1, 0])  # one settlement's worth
    return board, x
