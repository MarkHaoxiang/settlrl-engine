"""Tests for dice.py: the two-die roll and a sanity check on production payout.

(The full distribute_resources / bank-cap equivalence sweep lives in
test_rules.py; here we cover roll_dice and the empty-board no-op.)
"""

from __future__ import annotations

from typing import TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.mechanics import dice
from catan_engine.board import make_board
from catan_engine.board.layout import TILE_V, N_TILES, BoardLayout
from catan_engine.board.resources import BANK_INITIAL, N_PLAYERS, N_RESOURCES
from catan_engine.board.state import CITY, SETTLEMENT, BoardState
from catan_engine.board.tile import Tile

_T = TypeVar("_T")
_TILE_V = np.asarray(TILE_V)


def _single(tree: _T) -> _T:
    return cast(_T, jax.tree_util.tree_map(lambda x: x[0], tree))


class TestRollDice:
    def test_sum_in_range(self) -> None:
        for s in range(200):
            _, roll = dice.roll_dice(jax.random.key(s))
            assert 2 <= int(roll) <= 12

    def test_advances_key(self) -> None:
        k_in = jax.random.key(0)
        k_out, _ = dice.roll_dice(k_in)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(k_out)),
            np.asarray(jax.random.key_data(k_in)),
        )


def test_distribute_on_empty_board_is_noop() -> None:
    layout, state = make_board(1, seed=0)
    single = jax.tree_util.tree_map(lambda x: x[0], state)
    layout1 = jax.tree_util.tree_map(lambda x: x[0], layout)
    for roll in range(2, 13):
        out = dice.distribute_resources(layout1, single, jnp.int32(roll))
        assert int(np.asarray(out.player_resources).sum()) == 0


# ---------------------------------------------------------------------------
# Bank cap / depletion (dice.py:53-58)
# ---------------------------------------------------------------------------

_ROLL = 8  # the production roll the cap tests trigger


def _capped_board(
    settlement_owners: list[int], hands: np.ndarray
) -> tuple[BoardLayout, BoardState]:
    """Single-game (layout, state) where every named player owns a SHEEP-producing
    settlement on tile 0 (rolls ``_ROLL``), with ``hands`` the starting holdings.

    ``hands`` (shape (N_PLAYERS, N_RESOURCES)) sets the bank stock implicitly:
    bank[r] = BANK_INITIAL - hands[:, r].sum(). The robber sits on a different
    tile so tile 0 always produces.
    """
    layout, state = _single(make_board(1, seed=0))
    # Tile 0 produces SHEEP (resource index 0) on `_ROLL`; robber elsewhere.
    tile_resource = layout.tile_resource.at[0].set(int(Tile.SHEEP))
    tile_number = layout.tile_number.at[0].set(_ROLL)
    layout = layout._replace(tile_resource=tile_resource, tile_number=tile_number)

    owner = np.zeros(int(state.vertex_owner.shape[0]), np.uint8)
    vtype = np.zeros(int(state.vertex_type.shape[0]), np.uint8)
    corners = [int(c) for c in _TILE_V[0]]
    for i, p in enumerate(settlement_owners):
        v = corners[i]
        owner[v] = p + 1
        vtype[v] = SETTLEMENT

    robber = next(t for t in range(N_TILES) if t != 0)
    state = state._replace(
        vertex_owner=jnp.asarray(owner),
        vertex_type=jnp.asarray(vtype),
        player_resources=jnp.asarray(hands.astype(np.uint8)),
        robber=jnp.asarray(np.uint8(robber)),
    )
    return layout, state


def test_distribute_multi_claimant_over_bank_gives_nobody() -> None:
    # Two players each owed 2 sheep (4 total) but the bank holds only 3 ->
    # multiple claimants over stock -> nobody gets any sheep.
    hands = np.zeros((N_PLAYERS, N_RESOURCES), np.int64)
    # Deplete the sheep bank to 3 via player 3's holdings (no settlement).
    hands[3, 0] = BANK_INITIAL - 3  # bank sheep == 3
    layout, state = _capped_board([0, 1], hands)
    # Make players 0 and 1's buildings cities so each is owed 2 sheep (demand 4).
    vtype = np.asarray(state.vertex_type).copy()
    for v in (int(_TILE_V[0, 0]), int(_TILE_V[0, 1])):
        vtype[v] = CITY
    state = state._replace(vertex_type=jnp.asarray(vtype.astype(np.uint8)))

    out = dice.distribute_resources(layout, state, jnp.int32(_ROLL))
    res = np.asarray(out.player_resources).astype(int)
    assert res[0, 0] == 0, "claimant 0 should receive no sheep"
    assert res[1, 0] == 0, "claimant 1 should receive no sheep"
    # The depleting holder is untouched.
    assert res[3, 0] == BANK_INITIAL - 3


def test_distribute_single_claimant_capped_at_bank() -> None:
    # One player owed 2 sheep (a city) but the bank holds only 1 ->
    # single claimant -> receives exactly the remaining stock (1).
    hands = np.zeros((N_PLAYERS, N_RESOURCES), np.int64)
    hands[3, 0] = BANK_INITIAL - 1  # bank sheep == 1
    layout, state = _capped_board([0], hands)
    v = int(_TILE_V[0, 0])
    vtype = np.asarray(state.vertex_type).copy()
    vtype[v] = CITY  # city -> demand 2
    state = state._replace(vertex_type=jnp.asarray(vtype.astype(np.uint8)))

    out = dice.distribute_resources(layout, state, jnp.int32(_ROLL))
    res = np.asarray(out.player_resources).astype(int)
    assert res[0, 0] == 1, "single claimant gets min(demand=2, bank=1) == 1"
