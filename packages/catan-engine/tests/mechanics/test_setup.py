"""Tests for setup.py: the snake placement order and the 2nd-settlement grant
(equivalence vs the `catan-reference` oracle on a full-bank setup board)."""

from __future__ import annotations

from typing import TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from catan_engine.board import make_board
from catan_engine.board.layout import N_VERTICES
from catan_engine.board.resources import N_PLAYERS
from catan_engine.mechanics import setup

from tests import conversion as reference

_T = TypeVar("_T")


def _single(tree: _T) -> _T:
    return cast(_T, jax.tree_util.tree_map(lambda x: x[0], tree))


def test_setup_order_is_snake() -> None:
    assert setup.setup_order(4) == [0, 1, 2, 3, 3, 2, 1, 0]
    assert setup.setup_order(2) == [0, 1, 1, 0]
    # The traceable snake (`_setup_player`) agrees with the host-side order.
    for n in range(2, N_PLAYERS + 1):
        got = [int(setup._setup_player(jnp.int32(i), n)) for i in range(2 * n)]
        assert got == setup.setup_order(n)


def test_grant_setup_resources_matches_reference() -> None:
    # Fresh boards keep the bank full, so the per-tile and min(demand, bank)
    # payouts coincide; sweep vertices and players against the oracle.
    for seed in range(10):
        layout, state = make_board(1, seed=seed)
        layout1, state1 = _single(layout), _single(state)
        for player in range(N_PLAYERS):
            for vertex in range(0, N_VERTICES, 3):  # stride keeps the sweep quick
                ref = reference.grant_setup_resources(layout, state, vertex, player, 0)
                got = setup.grant_setup_resources(
                    layout1, state1, jnp.int32(vertex), jnp.int32(player)
                )
                assert np.array_equal(
                    np.asarray(got.player_resources),
                    np.asarray(ref.player_resources[0]),
                ), f"seed={seed} player={player} vertex={vertex}"
