"""Shared helpers for the mechanics equivalence tests.

The rule-equivalence tests all need (a) a random ``(edge_road, vertex_owner)``
board occupancy to sweep the legality / award rules over, and (b) the single-game
slicing util that pulls lane 0 out of a batched pytree. Both were duplicated
across ``test_rules`` / ``test_placement`` / ``test_robber`` / ``test_awards``;
they live here once, parameterized by the per-test probability weights.
"""

from __future__ import annotations

from typing import Sequence, TypeVar, cast

import jax
import numpy as np

from catan_engine.board.layout import N_EDGES, N_VERTICES

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
    networks realistically small.
    """
    rng = np.random.default_rng(seed)
    edge_road = rng.choice(_VALUES, size=N_EDGES, p=list(edge_p)).astype(np.uint8)
    vertex_owner = rng.choice(_VALUES, size=N_VERTICES, p=list(vertex_p)).astype(
        np.uint8
    )
    return edge_road, vertex_owner
