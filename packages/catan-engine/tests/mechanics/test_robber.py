"""Tests for robber.py: the victim mask (equivalence vs the NumPy oracle) and
the conservation properties of the random steal."""

from __future__ import annotations

from typing import TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.mechanics import robber
from catan_engine.board.layout import N_TILES, N_VERTICES
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import BoardState, make_board_state
from tests import reference

_T = TypeVar("_T")


def _single(tree: _T) -> _T:
    return cast(_T, jax.tree_util.tree_map(lambda x: x[0], tree))


def _state(seed: int) -> BoardState:
    rng = np.random.default_rng(seed)
    owner = rng.choice(
        [0, 1, 2, 3, 4], size=N_VERTICES, p=[0.6, 0.12, 0.11, 0.1, 0.07]
    ).astype(np.uint8)
    pr = rng.integers(0, 3, size=(N_PLAYERS, N_RESOURCES)).astype(np.uint8)
    return make_board_state(1, key=jax.random.key(seed))._replace(
        vertex_owner=jnp.asarray(owner)[None],
        player_resources=jnp.asarray(pr)[None],
    )


def test_victim_mask_matches_reference() -> None:
    for seed in range(30):
        state = _state(seed)
        single = _single(state)
        for tile in range(N_TILES):
            for current in range(N_PLAYERS):
                mask = robber.robber_victim_mask(
                    single, jnp.int32(tile), jnp.int32(current)
                )
                got = sorted(int(p) for p in np.where(np.asarray(mask))[0])
                ref = reference.robber_victims(state, tile, current, 0)
                assert got == ref, f"seed={seed} tile={tile} cur={current}"


class TestSteal:
    def _board(self, victim_hand: list[int]) -> BoardState:
        state = make_board_state(1, key=jax.random.key(3))
        pr = np.zeros((N_PLAYERS, N_RESOURCES), np.uint8)
        pr[2] = victim_hand
        return _single(state._replace(player_resources=jnp.asarray(pr)[None]))

    def test_moves_exactly_one_card(self) -> None:
        before = self._board([2, 0, 1, 0, 0])
        after = robber.steal(before, jnp.int32(0), jnp.int32(2))
        b = np.asarray(before.player_resources).astype(int)
        a = np.asarray(after.player_resources).astype(int)
        diff = a - b
        # Victim (2) loses one of a single resource; thief (0) gains that same one.
        assert diff.sum() == 0
        assert a[2].sum() == b[2].sum() - 1
        assert a[0].sum() == 1
        moved = np.where(diff[0] == 1)[0]
        assert moved.size == 1 and diff[2][moved[0]] == -1

    def test_empty_victim_is_noop_on_resources(self) -> None:
        before = self._board([0, 0, 0, 0, 0])
        after = robber.steal(before, jnp.int32(0), jnp.int32(2))
        assert np.array_equal(
            np.asarray(after.player_resources), np.asarray(before.player_resources)
        )

    def test_advances_key(self) -> None:
        before = self._board([1, 0, 0, 0, 0])
        after = robber.steal(before, jnp.int32(0), jnp.int32(2))
        assert not np.array_equal(
            np.asarray(jax.random.key_data(after.key)),
            np.asarray(jax.random.key_data(before.key)),
        )
