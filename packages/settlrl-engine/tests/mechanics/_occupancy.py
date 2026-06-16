"""Shared helpers for the mechanics equivalence tests.

The rule-equivalence tests all need (a) a random ``(edge_road, vertex_owner)``
board occupancy to sweep the legality / award rules over, and (b) the single-game
slicing util that pulls lane 0 out of a batched pytree. Both were duplicated
across ``test_rules`` / ``test_placement`` / ``test_robber`` / ``test_awards``;
they live here once, parameterized by the per-test probability weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, cast

import jax
import numpy as np
from settlrl_engine.board.layout import N_EDGES, N_VERTICES
from settlrl_engine.board.state import MAX_ROADS

_T = TypeVar("_T")

_VALUES = [0, 1, 2, 3, 4]  # 0 = empty, 1..4 = player + 1


def single(tree: _T) -> _T:
    """Slice lane 0 out of a batched pytree (the single-game view)."""
    return cast(_T, jax.tree_util.tree_map(lambda x: x[0], tree))


def random_occupancy(
    seed: int,
    *,
    edge_p: Sequence[float],
    vertex_p: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Random ``(edge_road, vertex_owner)`` arrays.

    Values are 0 (empty) or 1..4 (player + 1), drawn from the categorical
    weights ``edge_p`` / ``vertex_p`` (each length 5). The biases keep road
    networks realistically small, and each player is capped at ``MAX_ROADS``
    edges (excess randomly cleared): the longest-road bitmask and stack bounds
    rely on that rule invariant, so states beyond it are out of contract.
    """
    rng = np.random.default_rng(seed)
    edge_road = rng.choice(_VALUES, size=N_EDGES, p=list(edge_p)).astype(np.uint8)
    for code in range(1, 5):
        owned = np.where(edge_road == code)[0]
        if len(owned) > MAX_ROADS:
            drop = rng.choice(owned, size=len(owned) - MAX_ROADS, replace=False)
            edge_road[drop] = 0
    vertex_owner = rng.choice(_VALUES, size=N_VERTICES, p=list(vertex_p)).astype(
        np.uint8
    )
    return edge_road, vertex_owner
