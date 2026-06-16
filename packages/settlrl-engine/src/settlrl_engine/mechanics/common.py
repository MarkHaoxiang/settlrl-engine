"""Shared action vocabulary for the rule modules (single-game, traceable).

Holds what every per-action core needs -- result codes, phase predicates,
economy helpers, the action-layer jaxtyping aliases -- so the topical rule
modules stay leaves importing only ``board.*`` and ``common``, with no cycle
through ``action``.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from settlrl_engine.board.dev_cards import DEV_CARD_COST, DevCard
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.resources import (
    CITY_COST,
    ROAD_COST,
    SETTLEMENT_COST,
)
from settlrl_engine.board.state import (
    CITY,
    MAX_ROADS,
    SETTLEMENT,
    BoardState,
    GamePhase,
    to_u8,
)

# ---------------------------------------------------------------------------
# jaxtyping aliases for the action layer.
#
# Following the BoardState convention (state.py), every array carries a leading
# variable `batch` axis; `jax.vmap` over the single-game cores strips it, and
# all other axes are fixed game constants. `*_avail` cores yield a per-game
# `Mask`; applying an action yields `(BoardState, ResultCode)`.
# ---------------------------------------------------------------------------
Mask = Bool[Array, "batch"]  # per-game legality / win flag
ResultCode = Int[Array, "batch"]  # ActionResult codes
ActionTypeArray = Int[Array, "batch"]  # ActionType codes (unified dispatch)
IndexParam = Int[Array, "batch"]  # vertex / edge / tile / resource / player / victim
TwoIndexParams = tuple[IndexParam, IndexParam]  # e.g. (tile, victim), (give, receive)

# The cores in the rule modules run one game at a time, so they annotate their
# params/results with the single-game ``IntScalar`` / ``BoolScalar`` (from
# ``board.state``); the batched ``IndexParam`` / ``Mask`` / ``ResultCode``
# describe the wrapped views.

# Single-game legality-core signatures, keyed by their native param shape. Used
# by the flattening / enumeration helpers in ``action`` that close over a board
# and map a core over a static parameter domain.
IndexAvail = Callable[[BoardLayout, BoardState, jax.Array], jax.Array]
PairAvail = Callable[[BoardLayout, BoardState, tuple[jax.Array, jax.Array]], jax.Array]
NoneAvail = Callable[[BoardLayout, BoardState, None], jax.Array]


class ActionResult(IntEnum):
    """Outcome of attempting to apply an action."""

    SUCCESS = 0  # Legal and applied; play continues.
    INVALID = 1  # Not legal in the current state; board left unchanged.
    GAME_COMPLETE = 2  # Applied and ended the game (a player reached 10 VP).

    def __str__(self) -> str:
        return ("OK", "INVALID", "DONE")[self]


SUCCESS = jnp.int32(ActionResult.SUCCESS.value)
INVALID = jnp.int32(ActionResult.INVALID.value)
GAME_COMPLETE = jnp.int32(ActionResult.GAME_COMPLETE.value)


def main_after_roll(state: BoardState) -> jax.Array:
    """MAIN phase with the dice already rolled: the build / bank-trade window."""
    return (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)


def dev_play_window(state: BoardState) -> jax.Array:
    """ROLL (before rolling) or MAIN, with no development card played yet:
    the rulebook's "any time during your turn" window for the one dev card."""
    in_turn = (state.phase == GamePhase.ROLL) | (state.phase == GamePhase.MAIN)
    return in_turn & (state.dev_played == 0)


# ===========================================================================
# Economy helpers
#
# Single-game, traceable cost / affordability / payment / building-count /
# victory-point helpers over the BoardState arrays, used throughout the action
# cores. (Generic uint8 saturation lives on state.to_u8; bank stock on
# resources.bank_stock.)
# ===========================================================================

# Build-cost vectors in resource order [sheep, wheat, wood, brick, ore].
ROAD_COST_ARR = jnp.array(ROAD_COST, dtype=jnp.int32)
SETTLEMENT_COST_ARR = jnp.array(SETTLEMENT_COST, dtype=jnp.int32)
CITY_COST_ARR = jnp.array(CITY_COST, dtype=jnp.int32)
DEV_CARD_COST_ARR = jnp.array(DEV_CARD_COST, dtype=jnp.int32)


def roads_left(edge_road: jax.Array, player: jax.Array) -> jax.Array:
    built = jnp.sum(edge_road == player + 1).astype(jnp.int32)
    return MAX_ROADS - built


def count_settlements(
    vertex_owner: jax.Array, vertex_type: jax.Array, player: jax.Array
) -> jax.Array:
    return jnp.sum((vertex_owner == player + 1) & (vertex_type == SETTLEMENT)).astype(
        jnp.int32
    )


def count_cities(
    vertex_owner: jax.Array, vertex_type: jax.Array, player: jax.Array
) -> jax.Array:
    return jnp.sum((vertex_owner == player + 1) & (vertex_type == CITY)).astype(
        jnp.int32
    )


def can_afford(resources_row: jax.Array, cost_arr: jax.Array) -> jax.Array:
    """True if a single player's resource row covers ``cost_arr``."""
    return jnp.all(resources_row.astype(jnp.int32) >= cost_arr)


def pay(
    player_resources: jax.Array, player: jax.Array, cost_arr: jax.Array
) -> jax.Array:
    """Subtract ``cost_arr`` from ``player``'s row (clipped at 0), returning uint8."""
    updated = player_resources.astype(jnp.int32).at[player].add(-cost_arr)
    return to_u8(updated)


def player_total_vp(state: BoardState, player: jax.Array) -> jax.Array:
    """Building VP + awards + hidden Victory Point cards for ``player``."""
    total = state.victory_points[player].astype(jnp.int32)
    total += jnp.where(state.longest_road_owner == player, 2, 0)
    total += jnp.where(state.largest_army_owner == player, 2, 0)
    total += state.dev_hand[player, DevCard.VICTORY_POINT].astype(jnp.int32)
    return total


def agent_selection_single(state: BoardState) -> jax.Array:
    """Acting player for one game: the discarder during DISCARD, the proposed-to
    partner during TRADE_RESPONSE, else current."""
    owes = state.pending_discard > 0
    discarder = jnp.argmax(owes).astype(jnp.int32)
    in_discard = state.phase == jnp.uint8(GamePhase.DISCARD)
    in_response = state.phase == jnp.uint8(GamePhase.TRADE_RESPONSE)
    partner = jnp.clip(state.trade_partner.astype(jnp.int32), 0, state.n_players - 1)
    return jnp.where(
        in_discard,
        discarder,
        jnp.where(in_response, partner, state.current_player.astype(jnp.int32)),
    )
