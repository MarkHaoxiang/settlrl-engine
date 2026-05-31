"""Tests for awards.py award reassignment: Largest Army and Longest Road holder
selection (threshold + tie-to-holder), checked against the NumPy oracle.

(The longest-road *length* DFS itself is covered in test_rules.py.)
"""

from __future__ import annotations

from typing import TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.mechanics import awards
from catan_engine.board.layout import N_EDGES, N_VERTICES
from catan_engine.board.resources import N_PLAYERS
from catan_engine.board.state import NO_INDEX, BoardState, make_board_state
from tests import reference

_T = TypeVar("_T")


def _single(tree: _T) -> _T:
    return cast(_T, jax.tree_util.tree_map(lambda x: x[0], tree))


def _army_state(knights: list[int], owner: int) -> BoardState:
    return make_board_state(1, key=jax.random.key(0))._replace(
        knights_played=jnp.asarray(knights, jnp.uint8)[None],
        largest_army_owner=jnp.asarray([owner], jnp.uint8),
    )


class TestLargestArmy:
    def test_no_qualifier_is_unclaimed(self) -> None:
        out = awards.recompute_largest_army(_single(_army_state([2, 2, 0, 1], NO_INDEX)))
        assert int(out.largest_army_owner) == NO_INDEX

    def test_awarded_at_three(self) -> None:
        out = awards.recompute_largest_army(_single(_army_state([0, 3, 0, 0], NO_INDEX)))
        assert int(out.largest_army_owner) == 1

    def test_tie_keeps_current_holder(self) -> None:
        # Player 0 already holds it; player 2 ties at 3 -> holder keeps it.
        out = awards.recompute_largest_army(_single(_army_state([3, 0, 3, 0], 0)))
        assert int(out.largest_army_owner) == 0

    def test_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            knights = rng.integers(0, 5, size=N_PLAYERS).tolist()
            owner = int(rng.choice([NO_INDEX, 0, 1, 2, 3]))
            state = _army_state(knights, owner)
            got = awards.recompute_largest_army(_single(state))
            ref = reference.recompute_largest_army(state, 0)
            assert int(got.largest_army_owner) == int(ref.largest_army_owner[0]), (
                f"knights={knights} owner={owner}"
            )


def _road_state(seed: int) -> BoardState:
    rng = np.random.default_rng(seed)
    edge_road = rng.choice(
        [0, 1, 2, 3, 4], size=N_EDGES, p=[0.4, 0.2, 0.16, 0.14, 0.1]
    ).astype(np.uint8)
    vertex_owner = rng.choice(
        [0, 1, 2, 3, 4], size=N_VERTICES, p=[0.7, 0.1, 0.08, 0.07, 0.05]
    ).astype(np.uint8)
    owner = int(rng.choice([NO_INDEX, 0, 1, 2, 3]))
    return make_board_state(1, key=jax.random.key(0))._replace(
        edge_road=jnp.asarray(edge_road)[None],
        vertex_owner=jnp.asarray(vertex_owner)[None],
        longest_road_owner=jnp.asarray([owner], jnp.uint8),
    )


class TestLongestRoadAward:
    def test_recompute_matches_reference(self) -> None:
        for seed in range(40):
            state = _road_state(seed)
            got = awards.recompute_longest_road(_single(state))
            ref = reference.recompute_longest_road(state, 0)
            assert int(got.longest_road_owner) == int(ref.longest_road_owner[0]), (
                f"seed={seed}: owner"
            )
            assert int(got.longest_road_len) == int(ref.longest_road_len[0]), (
                f"seed={seed}: len"
            )
