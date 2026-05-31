"""Vectorized (JAX-native) Catan actions -- the engine's action layer.

Each action's core is a pure transition on a **single, unbatched game** written
in traceable JAX, composed from the focused rule modules (``placement``,
``awards``, ``trade``, ``dice``, ``robber``, ``setup``, ``development``) and the
local economy helpers below, over the static geometry maps in ``layout``. The public
``is_available`` / ``__call__`` are
``jax.jit(jax.vmap(...))`` over those cores, so they operate on a whole batch at
once: ``params`` are batched arrays and the outcome is a ``(batch,)`` array of
``ActionResult`` codes.

Application is branchless: the candidate next state is always computed, then
selected against the current state with ``state.tree_select`` using the
``is_available`` mask. Illegal games are returned unchanged with ``INVALID``.

Player convention (see state.py): players are 0-indexed; ``vertex_owner`` /
``edge_road`` store ``player + 1`` (0 = empty). Multi-field action params are
passed as tuples of ``(batch,)`` arrays; ``victim`` parameters are 0-indexed
with ``-1`` meaning "steal from no one".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Generic, NamedTuple, TypeVar, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from catan_engine.mechanics import (
    awards,
    development,
    dice,
    placement,
    robber,
    setup,
    trade,
)
from catan_engine.board import Board
from catan_engine.board.dev_cards import DEV_CARD_COST, DevCard
from catan_engine.board.layout import N_EDGES, N_TILES, N_VERTICES, BoardLayout
from catan_engine.board.resources import (
    CITY_COST,
    N_PLAYERS,
    N_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
    bank_stock,
)
from catan_engine.board.state import (
    CITY,
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    SETTLEMENT,
    VICTORY_POINTS_TO_WIN,
    BoardState,
    GamePhase,
    to_u8,
    tree_select,
)

ParamsT = TypeVar("ParamsT")

# ---------------------------------------------------------------------------
# jaxtyping aliases for the action layer.
#
# Following the BoardState convention (state.py), every array carries a leading
# variable `batch` axis; `jax.vmap` over the single-game cores in this module
# strips it, and all other axes are fixed game constants. `is_available` yields
# a per-game `Mask`; applying an action yields `(BoardState, ResultCode)`.
# ---------------------------------------------------------------------------
Mask = Bool[Array, "batch"]  # per-game legality / win flag
ResultCode = Int[Array, "batch"]  # ActionResult codes
ActionTypeArray = Int[Array, "batch"]  # ActionType codes (unified dispatch)
IndexParam = Int[Array, "batch"]  # vertex / edge / tile / resource / player / victim
ResourceParam = Int[Array, f"batch resources={N_RESOURCES}"]  # per-resource counts
TwoIndexParams = tuple[IndexParam, IndexParam]  # e.g. (tile, victim), (give, receive)
DiscardParams = tuple[IndexParam, ResourceParam]  # (player, per-resource discard counts)


class ActionResult(IntEnum):
    """Outcome of attempting to apply an action."""

    SUCCESS = 0        # Legal and applied; play continues.
    INVALID = 1        # Not legal in the current state; board left unchanged.
    GAME_COMPLETE = 2  # Applied and ended the game (a player reached 10 VP).

    def __str__(self) -> str:
        return ("OK", "INVALID", "DONE")[self]


SUCCESS = jnp.int32(ActionResult.SUCCESS.value)
INVALID = jnp.int32(ActionResult.INVALID.value)
GAME_COMPLETE = jnp.int32(ActionResult.GAME_COMPLETE.value)


# ===========================================================================
# Economy helpers
#
# Single-game, traceable cost / affordability / payment / building-count /
# victory-point helpers operating on the BoardState arrays, used throughout the
# action cores below. (Generic uint8 saturation lives on state.to_u8; bank stock
# on resources.bank_stock.)
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


class VecAction(ABC, Generic[ParamsT]):
    """Base class for the batched actions: methods return JAX arrays.

    ``is_available`` returns a ``(batch,)`` bool mask; ``__call__`` returns the
    new batched ``BoardState`` and a ``(batch,)`` int array of ActionResult
    codes. Cores are single-game; the public methods are jit(vmap(...)).
    """

    @abstractmethod
    def is_available(self, board: Board, params: ParamsT) -> Mask: ...

    @abstractmethod
    def __call__(self, board: Board, params: ParamsT) -> tuple[BoardState, ResultCode]:
        ...


def _outcome(available: Mask, won: Mask) -> ResultCode:
    return jnp.where(available, jnp.where(won, GAME_COMPLETE, SUCCESS), INVALID)


def _main_after_roll(state: BoardState) -> jax.Array:
    """MAIN phase with the dice already rolled: the build / bank-trade window."""
    return (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)


def _main_no_dev_played(state: BoardState) -> jax.Array:
    """MAIN phase with no development card played yet this turn."""
    return (state.phase == GamePhase.MAIN) & (state.dev_played == 0)


# ===========================================================================
# BuildRoad
# ===========================================================================


def _build_road_avail(
    layout: BoardLayout, state: BoardState, edge: IndexParam
) -> Mask:
    in_range = (edge >= 0) & (edge < N_EDGES)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    main = _main_after_roll(state)
    has_road = roads_left(state.edge_road, player) > 0
    placeable = placement.road_placeable(state.edge_road, state.vertex_owner, player, e)
    free = state.free_roads > 0
    afford = can_afford(state.player_resources[player], ROAD_COST_ARR)
    return in_range & main & has_road & placeable & (free | afford)


def _build_road_apply(
    layout: BoardLayout, state: BoardState, edge: IndexParam
) -> tuple[BoardState, ResultCode]:
    available = _build_road_avail(layout, state, edge)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    use_free = state.free_roads > 0
    new_free = to_u8(
        jnp.where(
            use_free,
            state.free_roads.astype(jnp.int32) - 1,
            state.free_roads.astype(jnp.int32),
        )
    )
    paid = pay(state.player_resources, player, ROAD_COST_ARR)
    new_res = jnp.where(use_free, state.player_resources, paid)
    cand = state._replace(
        edge_road=state.edge_road.at[e].set((player + 1).astype(jnp.uint8)),
        free_roads=new_free,
        player_resources=new_res,
    )
    cand = awards.recompute_longest_road(cand)
    won = player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return tree_select(available, cand, state), _outcome(available, won)


_build_road_avail_b = jax.jit(jax.vmap(_build_road_avail))
_build_road_apply_b = jax.jit(jax.vmap(_build_road_apply))


class BuildRoad(VecAction[IndexParam]):
    """Build a road (params: edge indices, shape (batch,)). Free if free_roads > 0."""

    def is_available(self, board: Board, params: IndexParam) -> Mask:
        return cast(Mask, _build_road_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: IndexParam) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _build_road_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# BuildSettlement
# ===========================================================================


def _build_settlement_avail(
    layout: BoardLayout, state: BoardState, vertex: IndexParam
) -> Mask:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    main = _main_after_roll(state)
    under_max = (
        count_settlements(state.vertex_owner, state.vertex_type, player)
        < MAX_SETTLEMENTS
    )
    afford = can_afford(state.player_resources[player], SETTLEMENT_COST_ARR)
    dist = placement.distance_rule_ok(state.vertex_owner, v)
    conn = placement.settlement_connected(state.edge_road, player, v)
    return in_range & main & under_max & afford & dist & conn


def _build_settlement_apply(
    layout: BoardLayout, state: BoardState, vertex: IndexParam
) -> tuple[BoardState, ResultCode]:
    available = _build_settlement_avail(layout, state, vertex)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    cand = state._replace(
        player_resources=pay(
            state.player_resources, player, SETTLEMENT_COST_ARR
        ),
        vertex_owner=state.vertex_owner.at[v].set((player + 1).astype(jnp.uint8)),
        vertex_type=state.vertex_type.at[v].set(SETTLEMENT),
        victory_points=state.victory_points.at[player].add(1),
    )
    cand = awards.recompute_longest_road(cand)
    won = player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return tree_select(available, cand, state), _outcome(available, won)


_build_settlement_avail_b = jax.jit(jax.vmap(_build_settlement_avail))
_build_settlement_apply_b = jax.jit(jax.vmap(_build_settlement_apply))


class BuildSettlement(VecAction[IndexParam]):
    """Build a settlement (params: vertex indices, shape (batch,))."""

    def is_available(self, board: Board, params: IndexParam) -> Mask:
        return cast(Mask, _build_settlement_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: IndexParam) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _build_settlement_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# RollDice
# ===========================================================================


def _roll_avail(layout: BoardLayout, state: BoardState, params: None) -> Mask:
    return (state.phase == GamePhase.ROLL) & (state.has_rolled == 0)


def _roll_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, ResultCode]:
    available = _roll_avail(layout, state, params)
    key, roll = dice.roll_dice(state.key)
    is_seven = roll == 7

    hand = state.player_resources.astype(jnp.int32).sum(axis=1)  # (P,)
    owes = jnp.where(hand > 7, hand // 2, 0).astype(jnp.uint8)
    pending = jnp.where(is_seven, owes, jnp.zeros_like(owes))
    any_discard = jnp.sum(pending) > 0
    phase_seven = jnp.where(any_discard, GamePhase.DISCARD, GamePhase.MOVE_ROBBER)
    new_phase = jnp.where(is_seven, phase_seven, GamePhase.MAIN).astype(jnp.uint8)

    distributed = dice.distribute_resources(layout, state, roll)
    new_res = jnp.where(is_seven, state.player_resources, distributed.player_resources)

    cand = state._replace(
        key=key,
        dice_roll=roll.astype(jnp.uint8),
        has_rolled=jnp.uint8(1),
        phase=new_phase,
        pending_discard=pending,
        player_resources=new_res,
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_roll_avail_b = jax.jit(jax.vmap(_roll_avail, in_axes=(0, 0, None)))
_roll_apply_b = jax.jit(jax.vmap(_roll_apply, in_axes=(0, 0, None)))


class RollDice(VecAction[None]):
    """Roll the dice (params: None). Distributes resources or triggers a 7."""

    def is_available(self, board: Board, params: None = None) -> Mask:
        return cast(Mask, _roll_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]", _roll_apply_b(board[0], board[1], None)
        )


# ===========================================================================
# BuildCity
# ===========================================================================


def _build_city_avail(
    layout: BoardLayout, state: BoardState, vertex: IndexParam
) -> Mask:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    main = _main_after_roll(state)
    under_max = (
        count_cities(state.vertex_owner, state.vertex_type, player) < MAX_CITIES
    )
    owns_settlement = (state.vertex_owner[v] == (player + 1).astype(jnp.uint8)) & (
        state.vertex_type[v] == SETTLEMENT
    )
    afford = can_afford(state.player_resources[player], CITY_COST_ARR)
    return in_range & main & under_max & owns_settlement & afford


def _build_city_apply(
    layout: BoardLayout, state: BoardState, vertex: IndexParam
) -> tuple[BoardState, ResultCode]:
    available = _build_city_avail(layout, state, vertex)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    cand = state._replace(
        player_resources=pay(state.player_resources, player, CITY_COST_ARR),
        vertex_type=state.vertex_type.at[v].set(CITY),
        victory_points=state.victory_points.at[player].add(1),
    )
    won = player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return tree_select(available, cand, state), _outcome(available, won)


_build_city_avail_b = jax.jit(jax.vmap(_build_city_avail))
_build_city_apply_b = jax.jit(jax.vmap(_build_city_apply))


class BuildCity(VecAction[IndexParam]):
    """Upgrade an own settlement to a city (params: vertex indices, shape (batch,))."""

    def is_available(self, board: Board, params: IndexParam) -> Mask:
        return cast(Mask, _build_city_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: IndexParam) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _build_city_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# BuyDevelopmentCard
# ===========================================================================


def _buy_dev_avail(layout: BoardLayout, state: BoardState, params: None) -> Mask:
    player = state.current_player.astype(jnp.int32)
    main = _main_after_roll(state)
    deck_nonempty = state.dev_deck.astype(jnp.int32).sum() > 0
    afford = can_afford(state.player_resources[player], DEV_CARD_COST_ARR)
    return main & deck_nonempty & afford


def _buy_dev_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, ResultCode]:
    available = _buy_dev_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    key, card = development.draw_dev_card(state.key, state.dev_deck)
    new_deck = state.dev_deck.astype(jnp.int32).at[card].add(-1)
    new_hand = state.dev_hand.astype(jnp.int32).at[player, card].add(1)
    new_bought = state.dev_bought.astype(jnp.int32).at[card].add(1)
    cand = state._replace(
        player_resources=pay(
            state.player_resources, player, DEV_CARD_COST_ARR
        ),
        dev_deck=to_u8(new_deck),
        dev_hand=to_u8(new_hand),
        dev_bought=to_u8(new_bought),
        key=key,
    )
    won = player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return tree_select(available, cand, state), _outcome(available, won)


_buy_dev_avail_b = jax.jit(jax.vmap(_buy_dev_avail, in_axes=(0, 0, None)))
_buy_dev_apply_b = jax.jit(jax.vmap(_buy_dev_apply, in_axes=(0, 0, None)))


class BuyDevelopmentCard(VecAction[None]):
    """Buy a development card (params: None). Draws from ``state.dev_deck``."""

    def is_available(self, board: Board, params: None = None) -> Mask:
        return cast(Mask, _buy_dev_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]", _buy_dev_apply_b(board[0], board[1], None)
        )


# ===========================================================================
# EndTurn
# ===========================================================================


def _end_turn_avail(layout: BoardLayout, state: BoardState, params: None) -> Mask:
    return _main_after_roll(state)


def _end_turn_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, ResultCode]:
    available = _end_turn_avail(layout, state, params)
    nxt = (state.current_player.astype(jnp.int32) + 1) % N_PLAYERS
    cand = state._replace(
        dice_roll=jnp.uint8(0),
        has_rolled=jnp.uint8(0),
        dev_played=jnp.uint8(0),
        dev_bought=jnp.zeros_like(state.dev_bought),
        free_roads=jnp.uint8(0),
        current_player=nxt.astype(state.current_player.dtype),
        phase=jnp.uint8(GamePhase.ROLL),
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_end_turn_avail_b = jax.jit(jax.vmap(_end_turn_avail, in_axes=(0, 0, None)))
_end_turn_apply_b = jax.jit(jax.vmap(_end_turn_apply, in_axes=(0, 0, None)))


class EndTurn(VecAction[None]):
    """End the current player's turn (params: None). Advances to the next player."""

    def is_available(self, board: Board, params: None = None) -> Mask:
        return cast(Mask, _end_turn_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]", _end_turn_apply_b(board[0], board[1], None)
        )


# ===========================================================================
# MaritimeTrade
# ===========================================================================


def _maritime_avail(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> Mask:
    give, receive = params
    player = state.current_player.astype(jnp.int32)
    g = jnp.clip(give, 0, N_RESOURCES - 1)
    r = jnp.clip(receive, 0, N_RESOURCES - 1)
    main = _main_after_roll(state)
    give_ok = (give >= 0) & (give < N_RESOURCES)
    recv_ok = (receive >= 0) & (receive < N_RESOURCES)
    distinct = give != receive
    ratio = trade.port_ratio(state.vertex_owner, layout.port_allocation, player, g)
    has_give = state.player_resources[player, g].astype(jnp.int32) >= ratio
    bank_ok = bank_stock(state.player_resources, r) >= 1
    return main & give_ok & recv_ok & distinct & has_give & bank_ok


def _maritime_apply(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> tuple[BoardState, ResultCode]:
    give, receive = params
    available = _maritime_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    g = jnp.clip(give, 0, N_RESOURCES - 1)
    r = jnp.clip(receive, 0, N_RESOURCES - 1)
    ratio = trade.port_ratio(state.vertex_owner, layout.port_allocation, player, g)
    res = state.player_resources.astype(jnp.int32)
    res = res.at[player, g].add(-ratio)
    res = res.at[player, r].add(1)
    cand = state._replace(
        player_resources=to_u8(res)
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_maritime_avail_b = jax.jit(jax.vmap(_maritime_avail))
_maritime_apply_b = jax.jit(jax.vmap(_maritime_apply))


class MaritimeTrade(VecAction["TwoIndexParams"]):
    """Trade with the bank at the best available ratio (params: (give, receive))."""

    def is_available(
        self, board: Board, params: TwoIndexParams
    ) -> Mask:
        return cast(Mask, _maritime_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: TwoIndexParams
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _maritime_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# PlayMonopoly
# ===========================================================================


def _monopoly_avail(
    layout: BoardLayout, state: BoardState, resource: IndexParam
) -> Mask:
    player = state.current_player.astype(jnp.int32)
    main = _main_no_dev_played(state)
    in_range = (resource >= 0) & (resource < N_RESOURCES)
    has_card = development.playable_dev(state, player, DevCard.MONOPOLY)
    return main & in_range & has_card


def _monopoly_apply(
    layout: BoardLayout, state: BoardState, resource: IndexParam
) -> tuple[BoardState, ResultCode]:
    available = _monopoly_avail(layout, state, resource)
    player = state.current_player.astype(jnp.int32)
    r = jnp.clip(resource, 0, N_RESOURCES - 1)
    res = state.player_resources.astype(jnp.int32)  # (N_PLAYERS, N_RESOURCES)
    taken = res[:, r].sum() - res[player, r]
    col = jnp.zeros((N_PLAYERS,), jnp.int32).at[player].set(res[player, r] + taken)
    res = res.at[:, r].set(col)
    new_hand = state.dev_hand.astype(jnp.int32).at[player, DevCard.MONOPOLY].add(-1)
    cand = state._replace(
        dev_played=jnp.uint8(1),
        dev_hand=to_u8(new_hand),
        player_resources=to_u8(res),
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_monopoly_avail_b = jax.jit(jax.vmap(_monopoly_avail))
_monopoly_apply_b = jax.jit(jax.vmap(_monopoly_apply))


class PlayMonopoly(VecAction[IndexParam]):
    """Play Monopoly (params: resource indices, shape (batch,)).

    Takes all of one resource from every other player.
    """

    def is_available(self, board: Board, params: IndexParam) -> Mask:
        return cast(Mask, _monopoly_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: IndexParam) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _monopoly_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# PlayYearOfPlenty
# ===========================================================================


def _yop_avail(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> Mask:
    resource_a, resource_b = params
    player = state.current_player.astype(jnp.int32)
    ca = jnp.clip(resource_a, 0, N_RESOURCES - 1)
    cb = jnp.clip(resource_b, 0, N_RESOURCES - 1)
    main = _main_no_dev_played(state)
    has_card = development.playable_dev(state, player, DevCard.YEAR_OF_PLENTY)
    a_ok = (resource_a >= 0) & (resource_a < N_RESOURCES)
    b_ok = (resource_b >= 0) & (resource_b < N_RESOURCES)
    same = resource_a == resource_b
    need_a = 1 + same.astype(jnp.int32)
    bank_a = bank_stock(state.player_resources, ca) >= need_a
    bank_b = same | (bank_stock(state.player_resources, cb) >= 1)
    return main & has_card & a_ok & b_ok & bank_a & bank_b


def _yop_apply(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> tuple[BoardState, ResultCode]:
    resource_a, resource_b = params
    available = _yop_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    ca = jnp.clip(resource_a, 0, N_RESOURCES - 1)
    cb = jnp.clip(resource_b, 0, N_RESOURCES - 1)
    new_hand = (
        state.dev_hand.astype(jnp.int32).at[player, DevCard.YEAR_OF_PLENTY].add(-1)
    )
    res = state.player_resources.astype(jnp.int32)
    res = res.at[player, ca].add(1)
    res = res.at[player, cb].add(1)
    cand = state._replace(
        dev_played=jnp.uint8(1),
        dev_hand=to_u8(new_hand),
        player_resources=to_u8(res),
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_yop_avail_b = jax.jit(jax.vmap(_yop_avail))
_yop_apply_b = jax.jit(jax.vmap(_yop_apply))


class PlayYearOfPlenty(VecAction["TwoIndexParams"]):
    """Play Year of Plenty (params: (resource_a, resource_b)).

    Takes two resource cards from the bank; ``a == b`` draws two of one kind.
    """

    def is_available(
        self, board: Board, params: TwoIndexParams
    ) -> Mask:
        return cast(Mask, _yop_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: TwoIndexParams
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _yop_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# PlayRoadBuilding
# ===========================================================================


def _road_building_avail(
    layout: BoardLayout, state: BoardState, params: None
) -> Mask:
    player = state.current_player.astype(jnp.int32)
    main = _main_no_dev_played(state)
    has_card = development.playable_dev(state, player, DevCard.ROAD_BUILDING)
    return main & has_card


def _road_building_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, ResultCode]:
    available = _road_building_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    grant = jnp.minimum(2, roads_left(state.edge_road, player))
    new_hand = (
        state.dev_hand.astype(jnp.int32).at[player, DevCard.ROAD_BUILDING].add(-1)
    )
    new_free = state.free_roads.astype(jnp.int32) + grant
    cand = state._replace(
        dev_played=jnp.uint8(1),
        dev_hand=to_u8(new_hand),
        free_roads=to_u8(new_free),
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_road_building_avail_b = jax.jit(jax.vmap(_road_building_avail, in_axes=(0, 0, None)))
_road_building_apply_b = jax.jit(jax.vmap(_road_building_apply, in_axes=(0, 0, None)))


class PlayRoadBuilding(VecAction[None]):
    """Play Road Building (params: None). Grants up to 2 free roads."""

    def is_available(self, board: Board, params: None = None) -> Mask:
        return cast(Mask, _road_building_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _road_building_apply_b(board[0], board[1], None),
        )


# ===========================================================================
# Robber helpers (shared by PlayKnight and MoveRobber)
# ===========================================================================


def _valid_robber_victim(
    state: BoardState, tile: jax.Array, player: jax.Array, victim: IndexParam
) -> Mask:
    """Victim choice is legal for a robber move onto ``tile`` by ``player``.

    If any opponent can be robbed on ``tile``, ``victim`` must name one of them;
    otherwise the only legal choice is ``-1`` ("steal from no one").
    """
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    mask = robber.robber_victim_mask(state, tile, player)
    victims_exist = jnp.any(mask)
    return jnp.where(
        victims_exist,
        (victim >= 0) & (victim < N_PLAYERS) & mask[vc],
        victim == -1,
    )


def _apply_steal(state: BoardState, player: jax.Array, victim: IndexParam) -> BoardState:
    """Steal a random card from ``victim`` when ``victim >= 0``; else leave state."""
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    stolen = robber.steal(state, player, vc)
    return tree_select(victim >= 0, stolen, state)


# ===========================================================================
# PlayKnight
# ===========================================================================


def _knight_avail(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> Mask:
    tile, victim = params
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    phase_ok = (state.phase == GamePhase.ROLL) | (state.phase == GamePhase.MAIN)
    not_played = state.dev_played == 0
    has_card = development.playable_dev(state, player, DevCard.KNIGHT)
    tile_in_range = (tile >= 0) & (tile < N_TILES)
    tile_moves = tile != state.robber
    valid_victim = _valid_robber_victim(state, t, player, victim)
    return (
        phase_ok
        & not_played
        & has_card
        & tile_in_range
        & tile_moves
        & valid_victim
    )


def _knight_apply(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> tuple[BoardState, ResultCode]:
    tile, victim = params
    available = _knight_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    new_hand = state.dev_hand.astype(jnp.int32).at[player, DevCard.KNIGHT].add(-1)
    new_knights = state.knights_played.astype(jnp.int32).at[player].add(1)
    cand = state._replace(
        dev_played=jnp.uint8(1),
        dev_hand=to_u8(new_hand),
        knights_played=to_u8(new_knights),
        robber=t.astype(state.robber.dtype),
    )
    cand = awards.recompute_largest_army(cand)
    cand = _apply_steal(cand, player, victim)
    won = player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return tree_select(available, cand, state), _outcome(available, won)


_knight_avail_b = jax.jit(jax.vmap(_knight_avail))
_knight_apply_b = jax.jit(jax.vmap(_knight_apply))


class PlayKnight(VecAction["TwoIndexParams"]):
    """Play a Knight (params: (tile, victim)).

    Moves the robber to ``tile`` and steals from ``victim`` (``victim == -1``
    steals from no one). Can win via the Largest Army award (+2 VP).
    """

    def is_available(
        self, board: Board, params: TwoIndexParams
    ) -> Mask:
        return cast(Mask, _knight_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: TwoIndexParams
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _knight_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# MoveRobber
# ===========================================================================


def _move_robber_avail(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> Mask:
    tile, victim = params
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    phase_ok = state.phase == GamePhase.MOVE_ROBBER
    tile_in_range = (tile >= 0) & (tile < N_TILES)
    tile_moves = tile != state.robber
    valid_victim = _valid_robber_victim(state, t, player, victim)
    return phase_ok & tile_in_range & tile_moves & valid_victim


def _move_robber_apply(
    layout: BoardLayout,
    state: BoardState,
    params: TwoIndexParams,
) -> tuple[BoardState, ResultCode]:
    tile, victim = params
    available = _move_robber_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    # Knight-before-roll resumes ROLL; the post-7 robber move resumes MAIN.
    new_phase = jnp.where(
        state.has_rolled != 0, GamePhase.MAIN, GamePhase.ROLL
    ).astype(jnp.uint8)
    cand = state._replace(
        robber=t.astype(state.robber.dtype),
        phase=new_phase,
    )
    cand = _apply_steal(cand, player, victim)
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_move_robber_avail_b = jax.jit(jax.vmap(_move_robber_avail))
_move_robber_apply_b = jax.jit(jax.vmap(_move_robber_apply))


class MoveRobber(VecAction["TwoIndexParams"]):
    """Move the robber and steal (params: (tile, victim)).

    Moves the robber to ``tile`` and steals from ``victim`` (``victim == -1``
    steals from no one). Resolves the post-7 (or knight-before-roll) robber
    move; never wins, so it always returns SUCCESS / INVALID.
    """

    def is_available(
        self, board: Board, params: TwoIndexParams
    ) -> Mask:
        return cast(Mask, _move_robber_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: TwoIndexParams
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _move_robber_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# Discard
# ===========================================================================


def _discard_avail(
    layout: BoardLayout,
    state: BoardState,
    params: DiscardParams,
) -> Mask:
    player, resources = params
    p = jnp.clip(player, 0, N_PLAYERS - 1)
    req = resources.astype(jnp.int32)  # (N_RESOURCES,)
    phase_ok = state.phase == GamePhase.DISCARD
    player_ok = (player >= 0) & (player < N_PLAYERS)
    nonneg = jnp.all(req >= 0)
    owed = state.pending_discard[p].astype(jnp.int32)
    owes = owed != 0
    count_ok = req.sum() == owed
    held = state.player_resources[p].astype(jnp.int32)
    within_hand = jnp.all(req <= held)
    return phase_ok & player_ok & nonneg & owes & count_ok & within_hand


def _discard_apply(
    layout: BoardLayout,
    state: BoardState,
    params: DiscardParams,
) -> tuple[BoardState, ResultCode]:
    player, resources = params
    available = _discard_avail(layout, state, params)
    p = jnp.clip(player, 0, N_PLAYERS - 1)
    req = resources.astype(jnp.int32)
    new_row = jnp.clip(state.player_resources[p].astype(jnp.int32) - req, 0, 255).astype(
        jnp.uint8
    )
    new_resources = state.player_resources.at[p].set(new_row)
    updated_pending = state.pending_discard.at[p].set(jnp.uint8(0))
    new_phase = jnp.where(
        updated_pending.astype(jnp.int32).sum() == 0,
        GamePhase.MOVE_ROBBER,
        GamePhase.DISCARD,
    ).astype(state.phase.dtype)
    cand = state._replace(
        player_resources=new_resources,
        pending_discard=updated_pending,
        phase=new_phase,
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_discard_avail_b = jax.jit(jax.vmap(_discard_avail))
_discard_apply_b = jax.jit(jax.vmap(_discard_apply))


class Discard(VecAction["DiscardParams"]):
    """Discard half your hand after a 7 (params: (player, resources)).

    ``player`` is a ``(batch,)`` int array; ``resources`` is a
    ``(batch, N_RESOURCES)`` int array of per-resource discard counts. When
    every player has finished discarding, the phase advances to MOVE_ROBBER.
    Never wins, so it always returns SUCCESS / INVALID.
    """

    def is_available(
        self, board: Board, params: DiscardParams
    ) -> Mask:
        return cast(Mask, _discard_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: DiscardParams
    ) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _discard_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# SetupSettlement
# ===========================================================================


def _setup_settlement_avail(
    layout: BoardLayout, state: BoardState, vertex: IndexParam
) -> Mask:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    phase_ok = state.phase == GamePhase.SETUP_SETTLEMENT
    dist = placement.distance_rule_ok(state.vertex_owner, v)
    return in_range & phase_ok & dist


def _setup_settlement_apply(
    layout: BoardLayout, state: BoardState, vertex: IndexParam
) -> tuple[BoardState, ResultCode]:
    available = _setup_settlement_avail(layout, state, vertex)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    placed = state._replace(
        vertex_owner=state.vertex_owner.at[v].set((player + 1).astype(jnp.uint8)),
        vertex_type=state.vertex_type.at[v].set(SETTLEMENT),
        victory_points=state.victory_points.at[player].add(1),
    )
    # The second settlement (placed in the reverse pass) grants resources.
    granted = setup.grant_setup_resources(layout, placed, v, player)
    placed = tree_select(
        state.setup_index.astype(jnp.int32) >= N_PLAYERS, granted, placed
    )
    cand = placed._replace(phase=jnp.uint8(GamePhase.SETUP_ROAD))
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_setup_settlement_avail_b = jax.jit(jax.vmap(_setup_settlement_avail))
_setup_settlement_apply_b = jax.jit(jax.vmap(_setup_settlement_apply))


class SetupSettlement(VecAction[IndexParam]):
    """Place a free starting settlement (params: vertex indices, shape (batch,)).

    The second settlement (placed in the reverse setup pass, when
    ``setup_index >= N_PLAYERS``) grants one resource per adjacent tile. Always
    advances to SETUP_ROAD; never wins, so it returns SUCCESS / INVALID.
    """

    def is_available(self, board: Board, params: IndexParam) -> Mask:
        return cast(Mask, _setup_settlement_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: IndexParam) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _setup_settlement_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# SetupRoad
# ===========================================================================


def _setup_road_avail(
    layout: BoardLayout, state: BoardState, edge: IndexParam
) -> Mask:
    in_range = (edge >= 0) & (edge < N_EDGES)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    phase_ok = state.phase == GamePhase.SETUP_ROAD
    empty = state.edge_road[e] == 0
    touches = placement.setup_road_placeable(
        state.edge_road, state.vertex_owner, player, e
    )
    return in_range & phase_ok & empty & touches


def _setup_road_apply(
    layout: BoardLayout, state: BoardState, edge: IndexParam
) -> tuple[BoardState, ResultCode]:
    available = _setup_road_avail(layout, state, edge)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    new_index = state.setup_index.astype(jnp.int32) + 1
    setup_continues = new_index < setup.N_SETUP
    next_player = jnp.where(
        setup_continues,
        setup.SETUP_ORDER_ARR[jnp.clip(new_index, 0, setup.N_SETUP - 1)],
        0,
    )
    next_phase = jnp.where(
        setup_continues, GamePhase.SETUP_SETTLEMENT, GamePhase.ROLL
    )
    cand = state._replace(
        edge_road=state.edge_road.at[e].set((player + 1).astype(jnp.uint8)),
        setup_index=new_index.astype(state.setup_index.dtype),
        phase=next_phase.astype(state.phase.dtype),
        current_player=next_player.astype(state.current_player.dtype),
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_setup_road_avail_b = jax.jit(jax.vmap(_setup_road_avail))
_setup_road_apply_b = jax.jit(jax.vmap(_setup_road_apply))


class SetupRoad(VecAction[IndexParam]):
    """Place the road next to the just-placed setup settlement (params: edge).

    The edge must be empty and touch a setup settlement the player owns that has
    no incident road yet. Advances the snake setup order: the next settlement
    placement, or ROLL with player 0 once setup is complete. Never wins, so it
    returns SUCCESS / INVALID.
    """

    def is_available(self, board: Board, params: IndexParam) -> Mask:
        return cast(Mask, _setup_road_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: IndexParam) -> tuple[BoardState, ResultCode]:
        return cast(
            "tuple[BoardState, ResultCode]",
            _setup_road_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# Unified action dispatch
# ===========================================================================
#
# A single ``(action_type, params)`` interface over all 15 actions, dispatched
# with ``jax.lax.switch`` so the whole thing stays traceable and vmappable. The
# heterogeneous per-action params are packed into one ``ActionParams`` struct;
# each branch unpacks the fields it needs and ignores the rest. Wrap
# ``apply_action`` / ``action_available`` in ``jax.vmap`` to run a batch (see
# ``catan_engine.env``).


class ActionType(IntEnum):
    """Index into the unified action set (the order of the dispatch branches)."""

    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL_DICE = 2
    DISCARD = 3
    MOVE_ROBBER = 4
    BUILD_ROAD = 5
    BUILD_SETTLEMENT = 6
    BUILD_CITY = 7
    BUY_DEVELOPMENT_CARD = 8
    PLAY_KNIGHT = 9
    PLAY_ROAD_BUILDING = 10
    PLAY_YEAR_OF_PLENTY = 11
    PLAY_MONOPOLY = 12
    MARITIME_TRADE = 13
    END_TURN = 14


N_ACTION_TYPES = len(ActionType)


class ActionParams(NamedTuple):
    """Packed parameters for the unified ``apply_action`` / ``action_available``.

    Single game: all fields are 0-d / 1-d arrays (batch by vmapping). Each action
    reads only what it needs; unused fields are ignored. By action:

    - vertex/edge/tile/resource/give (single index)  -> ``idx``
    - victim / receive / second resource (Year of Plenty) -> ``target``
    - Discard: ``idx`` = player, ``resources`` = per-resource discard counts
    - parameterless actions (RollDice, BuyDevelopmentCard, PlayRoadBuilding,
      EndTurn) read nothing.

    ``target`` follows the ``victim == -1`` ("steal from no one") convention for
    the robber actions. (``idx`` rather than ``index`` because ``NamedTuple``
    reserves the ``index`` attribute.)
    """

    idx: IndexParam  # primary index / player
    target: IndexParam  # secondary index (victim / receive / resource_b)
    resources: ResourceParam  # per-resource counts — Discard only


# Branch adapters in ActionType order: each maps the packed ActionParams onto the
# single-game core's native param shape.
_APPLY_BRANCHES = (
    lambda lay, st, pp: _setup_settlement_apply(lay, st, pp.idx),
    lambda lay, st, pp: _setup_road_apply(lay, st, pp.idx),
    lambda lay, st, pp: _roll_apply(lay, st, None),
    lambda lay, st, pp: _discard_apply(lay, st, (pp.idx, pp.resources)),
    lambda lay, st, pp: _move_robber_apply(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _build_road_apply(lay, st, pp.idx),
    lambda lay, st, pp: _build_settlement_apply(lay, st, pp.idx),
    lambda lay, st, pp: _build_city_apply(lay, st, pp.idx),
    lambda lay, st, pp: _buy_dev_apply(lay, st, None),
    lambda lay, st, pp: _knight_apply(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _road_building_apply(lay, st, None),
    lambda lay, st, pp: _yop_apply(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _monopoly_apply(lay, st, pp.idx),
    lambda lay, st, pp: _maritime_apply(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _end_turn_apply(lay, st, None),
)

_AVAIL_BRANCHES = (
    lambda lay, st, pp: _setup_settlement_avail(lay, st, pp.idx),
    lambda lay, st, pp: _setup_road_avail(lay, st, pp.idx),
    lambda lay, st, pp: _roll_avail(lay, st, None),
    lambda lay, st, pp: _discard_avail(lay, st, (pp.idx, pp.resources)),
    lambda lay, st, pp: _move_robber_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _build_road_avail(lay, st, pp.idx),
    lambda lay, st, pp: _build_settlement_avail(lay, st, pp.idx),
    lambda lay, st, pp: _build_city_avail(lay, st, pp.idx),
    lambda lay, st, pp: _buy_dev_avail(lay, st, None),
    lambda lay, st, pp: _knight_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _road_building_avail(lay, st, None),
    lambda lay, st, pp: _yop_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _monopoly_avail(lay, st, pp.idx),
    lambda lay, st, pp: _maritime_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _end_turn_avail(lay, st, None),
)


def apply_action(
    layout: BoardLayout,
    state: BoardState,
    action_type: ActionTypeArray,
    params: ActionParams,
) -> tuple[BoardState, ResultCode]:
    """Apply ``action_type`` (single game) and return (new state, ActionResult code)."""
    return cast(
        "tuple[BoardState, ResultCode]",
        jax.lax.switch(action_type, _APPLY_BRANCHES, layout, state, params),
    )


def action_available(
    layout: BoardLayout,
    state: BoardState,
    action_type: ActionTypeArray,
    params: ActionParams,
) -> Mask:
    """Legality of ``action_type`` with ``params`` (single game) as a scalar bool."""
    return cast(
        Mask,
        jax.lax.switch(action_type, _AVAIL_BRANCHES, layout, state, params),
    )
