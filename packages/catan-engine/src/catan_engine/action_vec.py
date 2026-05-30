"""Vectorized (JAX-native) Catan actions -- the engine's action layer.

Each action's core is a pure transition on a **single, unbatched game** written
in traceable JAX (see ``catan_engine.rules_vec``). The public ``is_available`` /
``__call__`` are ``jax.jit(jax.vmap(...))`` over those cores, so they operate on
a whole batch at once: ``params`` are batched arrays and the outcome is a
``(batch,)`` array of ``ActionResult`` codes.

Application is branchless: the candidate next state is always computed, then
selected against the current state with ``rules_vec.tree_select`` using the
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

from catan_engine import rules_vec as rv
from catan_engine.board import Board
from catan_engine.dev_cards import DevCard
from catan_engine.layout import BoardLayout, N_EDGES, N_TILES, N_VERTICES
from catan_engine.resources import N_PLAYERS, N_RESOURCES
from catan_engine.state import (
    MAX_CITIES,
    MAX_SETTLEMENTS,
    VICTORY_POINTS_TO_WIN,
    BoardState,
    GamePhase,
)

ParamsT = TypeVar("ParamsT")


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


class VecAction(ABC, Generic[ParamsT]):
    """Batched analogue of ``action.Action``: methods return JAX arrays.

    ``is_available`` returns a ``(batch,)`` bool mask; ``__call__`` returns the
    new batched ``BoardState`` and a ``(batch,)`` int array of ActionResult
    codes. Cores are single-game; the public methods are jit(vmap(...)).
    """

    @abstractmethod
    def is_available(self, board: Board, params: ParamsT) -> jax.Array: ...

    @abstractmethod
    def __call__(self, board: Board, params: ParamsT) -> tuple[BoardState, jax.Array]:
        ...


def _outcome(available: jax.Array, won: jax.Array) -> jax.Array:
    return jnp.where(available, jnp.where(won, GAME_COMPLETE, SUCCESS), INVALID)


# ===========================================================================
# BuildRoad
# ===========================================================================


def _build_road_avail(
    layout: BoardLayout, state: BoardState, edge: jax.Array
) -> jax.Array:
    in_range = (edge >= 0) & (edge < N_EDGES)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    main = (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)
    has_road = rv.roads_left(state.edge_road, player) > 0
    placeable = rv.road_placeable(state.edge_road, state.vertex_owner, player, e)
    free = state.free_roads > 0
    afford = rv.can_afford(state.player_resources[player], rv.ROAD_COST_ARR)
    return in_range & main & has_road & placeable & (free | afford)


def _build_road_apply(
    layout: BoardLayout, state: BoardState, edge: jax.Array
) -> tuple[BoardState, jax.Array]:
    available = _build_road_avail(layout, state, edge)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    use_free = state.free_roads > 0
    new_free = jnp.where(use_free, state.free_roads.astype(jnp.int32) - 1, 0).astype(
        jnp.uint8
    )
    paid = rv.pay(state.player_resources, player, rv.ROAD_COST_ARR)
    new_res = jnp.where(use_free, state.player_resources, paid)
    cand = state._replace(
        edge_road=state.edge_road.at[e].set((player + 1).astype(jnp.uint8)),
        free_roads=jnp.where(use_free, new_free, state.free_roads),
        player_resources=new_res,
    )
    cand = rv.recompute_longest_road(cand)
    won = rv.player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return rv.tree_select(available, cand, state), _outcome(available, won)


_build_road_avail_b = jax.jit(jax.vmap(_build_road_avail))
_build_road_apply_b = jax.jit(jax.vmap(_build_road_apply))


class BuildRoad(VecAction[jax.Array]):
    """Build a road (params: edge indices, shape (batch,)). Free if free_roads > 0."""

    def is_available(self, board: Board, params: jax.Array) -> jax.Array:
        return cast(jax.Array, _build_road_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: jax.Array) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _build_road_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# BuildSettlement
# ===========================================================================


def _build_settlement_avail(
    layout: BoardLayout, state: BoardState, vertex: jax.Array
) -> jax.Array:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    main = (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)
    under_max = (
        rv.count_settlements(state.vertex_owner, state.vertex_type, player)
        < MAX_SETTLEMENTS
    )
    afford = rv.can_afford(state.player_resources[player], rv.SETTLEMENT_COST_ARR)
    dist = rv.distance_rule_ok(state.vertex_owner, v)
    conn = rv.settlement_connected(state.edge_road, player, v)
    return in_range & main & under_max & afford & dist & conn


def _build_settlement_apply(
    layout: BoardLayout, state: BoardState, vertex: jax.Array
) -> tuple[BoardState, jax.Array]:
    available = _build_settlement_avail(layout, state, vertex)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    cand = state._replace(
        player_resources=rv.pay(
            state.player_resources, player, rv.SETTLEMENT_COST_ARR
        ),
        vertex_owner=state.vertex_owner.at[v].set((player + 1).astype(jnp.uint8)),
        vertex_type=state.vertex_type.at[v].set(1),
        victory_points=state.victory_points.at[player].add(1),
    )
    cand = rv.recompute_longest_road(cand)
    won = rv.player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return rv.tree_select(available, cand, state), _outcome(available, won)


_build_settlement_avail_b = jax.jit(jax.vmap(_build_settlement_avail))
_build_settlement_apply_b = jax.jit(jax.vmap(_build_settlement_apply))


class BuildSettlement(VecAction[jax.Array]):
    """Build a settlement (params: vertex indices, shape (batch,))."""

    def is_available(self, board: Board, params: jax.Array) -> jax.Array:
        return cast(jax.Array, _build_settlement_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: jax.Array) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _build_settlement_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# RollDice
# ===========================================================================


def _roll_avail(layout: BoardLayout, state: BoardState, params: None) -> jax.Array:
    return (state.phase == GamePhase.ROLL) & (state.has_rolled == 0)


def _roll_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, jax.Array]:
    available = _roll_avail(layout, state, params)
    key, roll = rv.roll_dice(state.key)
    is_seven = roll == 7

    hand = state.player_resources.astype(jnp.int32).sum(axis=1)  # (P,)
    owes = jnp.where(hand > 7, hand // 2, 0).astype(jnp.uint8)
    pending = jnp.where(is_seven, owes, jnp.zeros_like(owes))
    any_discard = jnp.sum(pending) > 0
    phase_seven = jnp.where(any_discard, GamePhase.DISCARD, GamePhase.MOVE_ROBBER)
    new_phase = jnp.where(is_seven, phase_seven, GamePhase.MAIN).astype(jnp.uint8)

    distributed = rv.distribute_resources(layout, state, roll)
    new_res = jnp.where(is_seven, state.player_resources, distributed.player_resources)

    cand = state._replace(
        key=key,
        dice_roll=roll.astype(jnp.uint8),
        has_rolled=jnp.uint8(1),
        phase=new_phase,
        pending_discard=pending,
        player_resources=new_res,
    )
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_roll_avail_b = jax.jit(jax.vmap(_roll_avail, in_axes=(0, 0, None)))
_roll_apply_b = jax.jit(jax.vmap(_roll_apply, in_axes=(0, 0, None)))


class RollDice(VecAction[None]):
    """Roll the dice (params: None). Distributes resources or triggers a 7."""

    def is_available(self, board: Board, params: None = None) -> jax.Array:
        return cast(jax.Array, _roll_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]", _roll_apply_b(board[0], board[1], None)
        )


# ===========================================================================
# BuildCity
# ===========================================================================


def _build_city_avail(
    layout: BoardLayout, state: BoardState, vertex: jax.Array
) -> jax.Array:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    main = (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)
    under_max = (
        rv.count_cities(state.vertex_owner, state.vertex_type, player) < MAX_CITIES
    )
    owns_settlement = (state.vertex_owner[v] == (player + 1).astype(jnp.uint8)) & (
        state.vertex_type[v] == 1
    )
    afford = rv.can_afford(state.player_resources[player], rv.CITY_COST_ARR)
    return in_range & main & under_max & owns_settlement & afford


def _build_city_apply(
    layout: BoardLayout, state: BoardState, vertex: jax.Array
) -> tuple[BoardState, jax.Array]:
    available = _build_city_avail(layout, state, vertex)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    cand = state._replace(
        player_resources=rv.pay(state.player_resources, player, rv.CITY_COST_ARR),
        vertex_type=state.vertex_type.at[v].set(2),
        victory_points=state.victory_points.at[player].add(1),
    )
    won = rv.player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return rv.tree_select(available, cand, state), _outcome(available, won)


_build_city_avail_b = jax.jit(jax.vmap(_build_city_avail))
_build_city_apply_b = jax.jit(jax.vmap(_build_city_apply))


class BuildCity(VecAction[jax.Array]):
    """Upgrade an own settlement to a city (params: vertex indices, shape (batch,))."""

    def is_available(self, board: Board, params: jax.Array) -> jax.Array:
        return cast(jax.Array, _build_city_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: jax.Array) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _build_city_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# BuyDevelopmentCard
# ===========================================================================


def _buy_dev_avail(layout: BoardLayout, state: BoardState, params: None) -> jax.Array:
    player = state.current_player.astype(jnp.int32)
    main = (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)
    deck_nonempty = state.dev_deck.astype(jnp.int32).sum() > 0
    afford = rv.can_afford(state.player_resources[player], rv.DEV_CARD_COST_ARR)
    return main & deck_nonempty & afford


def _buy_dev_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, jax.Array]:
    available = _buy_dev_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    key, card = rv.draw_dev_card(state.key, state.dev_deck)
    new_deck = state.dev_deck.astype(jnp.int32).at[card].add(-1)
    new_hand = state.dev_hand.astype(jnp.int32).at[player, card].add(1)
    new_bought = state.dev_bought.astype(jnp.int32).at[card].add(1)
    cand = state._replace(
        player_resources=rv.pay(
            state.player_resources, player, rv.DEV_CARD_COST_ARR
        ),
        dev_deck=jnp.clip(new_deck, 0, 255).astype(jnp.uint8),
        dev_hand=jnp.clip(new_hand, 0, 255).astype(jnp.uint8),
        dev_bought=jnp.clip(new_bought, 0, 255).astype(jnp.uint8),
        key=key,
    )
    won = rv.player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return rv.tree_select(available, cand, state), _outcome(available, won)


_buy_dev_avail_b = jax.jit(jax.vmap(_buy_dev_avail, in_axes=(0, 0, None)))
_buy_dev_apply_b = jax.jit(jax.vmap(_buy_dev_apply, in_axes=(0, 0, None)))


class BuyDevelopmentCard(VecAction[None]):
    """Buy a development card (params: None). Draws from ``state.dev_deck``."""

    def is_available(self, board: Board, params: None = None) -> jax.Array:
        return cast(jax.Array, _buy_dev_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]", _buy_dev_apply_b(board[0], board[1], None)
        )


# ===========================================================================
# EndTurn
# ===========================================================================


def _end_turn_avail(layout: BoardLayout, state: BoardState, params: None) -> jax.Array:
    return (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)


def _end_turn_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, jax.Array]:
    available = _end_turn_avail(layout, state, params)
    nxt = (state.current_player.astype(jnp.int32) + 1) % N_PLAYERS
    cand = state._replace(
        dice_roll=jnp.uint8(0),
        has_rolled=jnp.uint8(0),
        dev_played=jnp.uint8(0),
        dev_bought=jnp.zeros_like(state.dev_bought),
        free_roads=jnp.uint8(0),
        current_player=nxt.astype(state.current_player.dtype),
        turn_number=state.turn_number + 1,
        phase=jnp.uint8(GamePhase.ROLL),
    )
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_end_turn_avail_b = jax.jit(jax.vmap(_end_turn_avail, in_axes=(0, 0, None)))
_end_turn_apply_b = jax.jit(jax.vmap(_end_turn_apply, in_axes=(0, 0, None)))


class EndTurn(VecAction[None]):
    """End the current player's turn (params: None). Advances to the next player."""

    def is_available(self, board: Board, params: None = None) -> jax.Array:
        return cast(jax.Array, _end_turn_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]", _end_turn_apply_b(board[0], board[1], None)
        )


# ===========================================================================
# MaritimeTrade
# ===========================================================================


def _maritime_avail(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> jax.Array:
    give, receive = params
    player = state.current_player.astype(jnp.int32)
    g = jnp.clip(give, 0, N_RESOURCES - 1)
    r = jnp.clip(receive, 0, N_RESOURCES - 1)
    main = (state.phase == GamePhase.MAIN) & (state.has_rolled != 0)
    give_ok = (give >= 0) & (give < N_RESOURCES)
    recv_ok = (receive >= 0) & (receive < N_RESOURCES)
    distinct = give != receive
    ratio = rv.port_ratio(state.vertex_owner, layout.port_allocation, player, g)
    has_give = state.player_resources[player, g].astype(jnp.int32) >= ratio
    bank_ok = rv.bank_stock(state.player_resources, r) >= 1
    return main & give_ok & recv_ok & distinct & has_give & bank_ok


def _maritime_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> tuple[BoardState, jax.Array]:
    give, receive = params
    available = _maritime_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    g = jnp.clip(give, 0, N_RESOURCES - 1)
    r = jnp.clip(receive, 0, N_RESOURCES - 1)
    ratio = rv.port_ratio(state.vertex_owner, layout.port_allocation, player, g)
    res = state.player_resources.astype(jnp.int32)
    res = res.at[player, g].add(-ratio)
    res = res.at[player, r].add(1)
    cand = state._replace(
        player_resources=jnp.clip(res, 0, 255).astype(jnp.uint8)
    )
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_maritime_avail_b = jax.jit(jax.vmap(_maritime_avail))
_maritime_apply_b = jax.jit(jax.vmap(_maritime_apply))


class MaritimeTrade(VecAction["tuple[jax.Array, jax.Array]"]):
    """Trade with the bank at the best available ratio (params: (give, receive))."""

    def is_available(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        return cast(jax.Array, _maritime_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _maritime_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# PlayMonopoly
# ===========================================================================


def _monopoly_avail(
    layout: BoardLayout, state: BoardState, resource: jax.Array
) -> jax.Array:
    player = state.current_player.astype(jnp.int32)
    main = (state.phase == GamePhase.MAIN) & (state.dev_played == 0)
    in_range = (resource >= 0) & (resource < N_RESOURCES)
    has_card = rv.playable_dev(state, player, DevCard.MONOPOLY)
    return main & in_range & has_card


def _monopoly_apply(
    layout: BoardLayout, state: BoardState, resource: jax.Array
) -> tuple[BoardState, jax.Array]:
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
        dev_hand=jnp.clip(new_hand, 0, 255).astype(jnp.uint8),
        player_resources=jnp.clip(res, 0, 255).astype(jnp.uint8),
    )
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_monopoly_avail_b = jax.jit(jax.vmap(_monopoly_avail))
_monopoly_apply_b = jax.jit(jax.vmap(_monopoly_apply))


class PlayMonopoly(VecAction[jax.Array]):
    """Play Monopoly (params: resource indices, shape (batch,)).

    Takes all of one resource from every other player.
    """

    def is_available(self, board: Board, params: jax.Array) -> jax.Array:
        return cast(jax.Array, _monopoly_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: jax.Array) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _monopoly_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# PlayYearOfPlenty
# ===========================================================================


def _yop_avail(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> jax.Array:
    resource_a, resource_b = params
    player = state.current_player.astype(jnp.int32)
    ca = jnp.clip(resource_a, 0, N_RESOURCES - 1)
    cb = jnp.clip(resource_b, 0, N_RESOURCES - 1)
    main = (state.phase == GamePhase.MAIN) & (state.dev_played == 0)
    has_card = rv.playable_dev(state, player, DevCard.YEAR_OF_PLENTY)
    a_ok = (resource_a >= 0) & (resource_a < N_RESOURCES)
    b_ok = (resource_b >= 0) & (resource_b < N_RESOURCES)
    same = resource_a == resource_b
    need_a = 1 + same.astype(jnp.int32)
    bank_a = rv.bank_stock(state.player_resources, ca) >= need_a
    bank_b = same | (rv.bank_stock(state.player_resources, cb) >= 1)
    return main & has_card & a_ok & b_ok & bank_a & bank_b


def _yop_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> tuple[BoardState, jax.Array]:
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
        dev_hand=jnp.clip(new_hand, 0, 255).astype(jnp.uint8),
        player_resources=jnp.clip(res, 0, 255).astype(jnp.uint8),
    )
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_yop_avail_b = jax.jit(jax.vmap(_yop_avail))
_yop_apply_b = jax.jit(jax.vmap(_yop_apply))


class PlayYearOfPlenty(VecAction["tuple[jax.Array, jax.Array]"]):
    """Play Year of Plenty (params: (resource_a, resource_b)).

    Takes two resource cards from the bank; ``a == b`` draws two of one kind.
    """

    def is_available(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        return cast(jax.Array, _yop_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _yop_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# PlayRoadBuilding
# ===========================================================================


def _road_building_avail(
    layout: BoardLayout, state: BoardState, params: None
) -> jax.Array:
    player = state.current_player.astype(jnp.int32)
    main = (state.phase == GamePhase.MAIN) & (state.dev_played == 0)
    has_card = rv.playable_dev(state, player, DevCard.ROAD_BUILDING)
    return main & has_card


def _road_building_apply(
    layout: BoardLayout, state: BoardState, params: None
) -> tuple[BoardState, jax.Array]:
    available = _road_building_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    grant = jnp.minimum(2, rv.roads_left(state.edge_road, player))
    new_hand = (
        state.dev_hand.astype(jnp.int32).at[player, DevCard.ROAD_BUILDING].add(-1)
    )
    new_free = state.free_roads.astype(jnp.int32) + grant
    cand = state._replace(
        dev_played=jnp.uint8(1),
        dev_hand=jnp.clip(new_hand, 0, 255).astype(jnp.uint8),
        free_roads=jnp.clip(new_free, 0, 255).astype(jnp.uint8),
    )
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_road_building_avail_b = jax.jit(jax.vmap(_road_building_avail, in_axes=(0, 0, None)))
_road_building_apply_b = jax.jit(jax.vmap(_road_building_apply, in_axes=(0, 0, None)))


class PlayRoadBuilding(VecAction[None]):
    """Play Road Building (params: None). Grants up to 2 free roads."""

    def is_available(self, board: Board, params: None = None) -> jax.Array:
        return cast(jax.Array, _road_building_avail_b(board[0], board[1], None))

    def __call__(
        self, board: Board, params: None = None
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _road_building_apply_b(board[0], board[1], None),
        )


# ===========================================================================
# PlayKnight
# ===========================================================================


def _knight_avail(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> jax.Array:
    tile, victim = params
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    phase_ok = (state.phase == GamePhase.ROLL) | (state.phase == GamePhase.MAIN)
    not_played = state.dev_played == 0
    has_card = rv.playable_dev(state, player, DevCard.KNIGHT)
    tile_in_range = (tile >= 0) & (tile < N_TILES)
    tile_moves = tile != state.robber
    mask = rv.robber_victim_mask(state, t, player)
    victims_exist = jnp.any(mask)
    valid_victim = jnp.where(
        victims_exist,
        (victim >= 0) & (victim < N_PLAYERS) & mask[vc],
        victim == -1,
    )
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
    params: tuple[jax.Array, jax.Array],
) -> tuple[BoardState, jax.Array]:
    tile, victim = params
    available = _knight_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    new_hand = state.dev_hand.astype(jnp.int32).at[player, DevCard.KNIGHT].add(-1)
    new_knights = state.knights_played.astype(jnp.int32).at[player].add(1)
    cand = state._replace(
        dev_played=jnp.uint8(1),
        dev_hand=jnp.clip(new_hand, 0, 255).astype(jnp.uint8),
        knights_played=jnp.clip(new_knights, 0, 255).astype(jnp.uint8),
        robber=t.astype(state.robber.dtype),
    )
    cand = rv.recompute_largest_army(cand)
    stolen = rv.steal(cand, player, vc)
    do_steal = victim >= 0
    cand = rv.tree_select(do_steal, stolen, cand)
    won = rv.player_total_vp(cand, player) >= VICTORY_POINTS_TO_WIN
    return rv.tree_select(available, cand, state), _outcome(available, won)


_knight_avail_b = jax.jit(jax.vmap(_knight_avail))
_knight_apply_b = jax.jit(jax.vmap(_knight_apply))


class PlayKnight(VecAction["tuple[jax.Array, jax.Array]"]):
    """Play a Knight (params: (tile, victim)).

    Moves the robber to ``tile`` and steals from ``victim`` (``victim == -1``
    steals from no one). Can win via the Largest Army award (+2 VP).
    """

    def is_available(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        return cast(jax.Array, _knight_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _knight_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# MoveRobber
# ===========================================================================


def _move_robber_avail(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> jax.Array:
    tile, victim = params
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    phase_ok = state.phase == GamePhase.MOVE_ROBBER
    tile_in_range = (tile >= 0) & (tile < N_TILES)
    tile_moves = tile != state.robber
    mask = rv.robber_victim_mask(state, t, player)
    victims_exist = jnp.any(mask)
    valid_victim = jnp.where(
        victims_exist,
        (victim >= 0) & (victim < N_PLAYERS) & mask[vc],
        victim == -1,
    )
    return phase_ok & tile_in_range & tile_moves & valid_victim


def _move_robber_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> tuple[BoardState, jax.Array]:
    tile, victim = params
    available = _move_robber_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    # Knight-before-roll resumes ROLL; the post-7 robber move resumes MAIN.
    new_phase = jnp.where(
        state.has_rolled != 0, GamePhase.MAIN, GamePhase.ROLL
    ).astype(jnp.uint8)
    cand = state._replace(
        robber=t.astype(state.robber.dtype),
        phase=new_phase,
    )
    stolen = rv.steal(cand, player, vc)
    do_steal = victim >= 0
    cand = rv.tree_select(do_steal, stolen, cand)
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_move_robber_avail_b = jax.jit(jax.vmap(_move_robber_avail))
_move_robber_apply_b = jax.jit(jax.vmap(_move_robber_apply))


class MoveRobber(VecAction["tuple[jax.Array, jax.Array]"]):
    """Move the robber and steal (params: (tile, victim)).

    Moves the robber to ``tile`` and steals from ``victim`` (``victim == -1``
    steals from no one). Resolves the post-7 (or knight-before-roll) robber
    move; never wins, so it always returns SUCCESS / INVALID.
    """

    def is_available(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        return cast(jax.Array, _move_robber_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _move_robber_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# Discard
# ===========================================================================


def _discard_avail(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[jax.Array, jax.Array],
) -> jax.Array:
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
    params: tuple[jax.Array, jax.Array],
) -> tuple[BoardState, jax.Array]:
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
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_discard_avail_b = jax.jit(jax.vmap(_discard_avail))
_discard_apply_b = jax.jit(jax.vmap(_discard_apply))


class Discard(VecAction["tuple[jax.Array, jax.Array]"]):
    """Discard half your hand after a 7 (params: (player, resources)).

    ``player`` is a ``(batch,)`` int array; ``resources`` is a
    ``(batch, N_RESOURCES)`` int array of per-resource discard counts. When
    every player has finished discarding, the phase advances to MOVE_ROBBER.
    Never wins, so it always returns SUCCESS / INVALID.
    """

    def is_available(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        return cast(jax.Array, _discard_avail_b(board[0], board[1], params))

    def __call__(
        self, board: Board, params: tuple[jax.Array, jax.Array]
    ) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _discard_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# SetupSettlement
# ===========================================================================


def _setup_settlement_avail(
    layout: BoardLayout, state: BoardState, vertex: jax.Array
) -> jax.Array:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    phase_ok = state.phase == GamePhase.SETUP_SETTLEMENT
    dist = rv.distance_rule_ok(state.vertex_owner, v)
    return in_range & phase_ok & dist


def _setup_settlement_apply(
    layout: BoardLayout, state: BoardState, vertex: jax.Array
) -> tuple[BoardState, jax.Array]:
    available = _setup_settlement_avail(layout, state, vertex)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    placed = state._replace(
        vertex_owner=state.vertex_owner.at[v].set((player + 1).astype(jnp.uint8)),
        vertex_type=state.vertex_type.at[v].set(1),
        victory_points=state.victory_points.at[player].add(1),
    )
    # The second settlement (placed in the reverse pass) grants resources.
    granted = rv.grant_setup_resources(layout, placed, v, player)
    placed = rv.tree_select(
        state.setup_index.astype(jnp.int32) >= N_PLAYERS, granted, placed
    )
    cand = placed._replace(phase=jnp.uint8(GamePhase.SETUP_ROAD))
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_setup_settlement_avail_b = jax.jit(jax.vmap(_setup_settlement_avail))
_setup_settlement_apply_b = jax.jit(jax.vmap(_setup_settlement_apply))


class SetupSettlement(VecAction[jax.Array]):
    """Place a free starting settlement (params: vertex indices, shape (batch,)).

    The second settlement (placed in the reverse setup pass, when
    ``setup_index >= N_PLAYERS``) grants one resource per adjacent tile. Always
    advances to SETUP_ROAD; never wins, so it returns SUCCESS / INVALID.
    """

    def is_available(self, board: Board, params: jax.Array) -> jax.Array:
        return cast(jax.Array, _setup_settlement_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: jax.Array) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
            _setup_settlement_apply_b(board[0], board[1], params),
        )


# ===========================================================================
# SetupRoad
# ===========================================================================


def _setup_road_avail(
    layout: BoardLayout, state: BoardState, edge: jax.Array
) -> jax.Array:
    in_range = (edge >= 0) & (edge < N_EDGES)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    target = (player + 1).astype(state.vertex_owner.dtype)
    phase_ok = state.phase == GamePhase.SETUP_ROAD
    empty = state.edge_road[e] == 0

    def endpoint_ok(v: jax.Array) -> jax.Array:
        owns_here = state.vertex_owner[v] == target
        e2 = rv.V_EDGES[v]  # (MAX_VERTEX_DEGREE,)
        valid = e2 != rv.NO_IDX
        roads = state.edge_road[jnp.where(valid, e2, 0)]
        has_own_road = jnp.any(valid & (roads == target))
        return owns_here & ~has_own_road

    touches = endpoint_ok(rv.EDGE_V[e, 0]) | endpoint_ok(rv.EDGE_V[e, 1])
    return in_range & phase_ok & empty & touches


def _setup_road_apply(
    layout: BoardLayout, state: BoardState, edge: jax.Array
) -> tuple[BoardState, jax.Array]:
    available = _setup_road_avail(layout, state, edge)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    new_index = state.setup_index.astype(jnp.int32) + 1
    setup_continues = new_index < rv.N_SETUP
    next_player = jnp.where(
        setup_continues,
        rv.SETUP_ORDER_ARR[jnp.clip(new_index, 0, rv.N_SETUP - 1)],
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
    return rv.tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_setup_road_avail_b = jax.jit(jax.vmap(_setup_road_avail))
_setup_road_apply_b = jax.jit(jax.vmap(_setup_road_apply))


class SetupRoad(VecAction[jax.Array]):
    """Place the road next to the just-placed setup settlement (params: edge).

    The edge must be empty and touch a setup settlement the player owns that has
    no incident road yet. Advances the snake setup order: the next settlement
    placement, or ROLL with player 0 once setup is complete. Never wins, so it
    returns SUCCESS / INVALID.
    """

    def is_available(self, board: Board, params: jax.Array) -> jax.Array:
        return cast(jax.Array, _setup_road_avail_b(board[0], board[1], params))

    def __call__(self, board: Board, params: jax.Array) -> tuple[BoardState, jax.Array]:
        return cast(
            "tuple[BoardState, jax.Array]",
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

    idx: jax.Array  # int32 primary index / player
    target: jax.Array  # int32 secondary index (victim / receive / resource_b)
    resources: jax.Array  # int32 (N_RESOURCES,) — Discard only


# Branch adapters in ActionType order: each maps the packed ActionParams onto the
# single-game core's native param shape.
_APPLY_BRANCHES = (
    lambda l, s, p: _setup_settlement_apply(l, s, p.idx),
    lambda l, s, p: _setup_road_apply(l, s, p.idx),
    lambda l, s, p: _roll_apply(l, s, None),
    lambda l, s, p: _discard_apply(l, s, (p.idx, p.resources)),
    lambda l, s, p: _move_robber_apply(l, s, (p.idx, p.target)),
    lambda l, s, p: _build_road_apply(l, s, p.idx),
    lambda l, s, p: _build_settlement_apply(l, s, p.idx),
    lambda l, s, p: _build_city_apply(l, s, p.idx),
    lambda l, s, p: _buy_dev_apply(l, s, None),
    lambda l, s, p: _knight_apply(l, s, (p.idx, p.target)),
    lambda l, s, p: _road_building_apply(l, s, None),
    lambda l, s, p: _yop_apply(l, s, (p.idx, p.target)),
    lambda l, s, p: _monopoly_apply(l, s, p.idx),
    lambda l, s, p: _maritime_apply(l, s, (p.idx, p.target)),
    lambda l, s, p: _end_turn_apply(l, s, None),
)

_AVAIL_BRANCHES = (
    lambda l, s, p: _setup_settlement_avail(l, s, p.idx),
    lambda l, s, p: _setup_road_avail(l, s, p.idx),
    lambda l, s, p: _roll_avail(l, s, None),
    lambda l, s, p: _discard_avail(l, s, (p.idx, p.resources)),
    lambda l, s, p: _move_robber_avail(l, s, (p.idx, p.target)),
    lambda l, s, p: _build_road_avail(l, s, p.idx),
    lambda l, s, p: _build_settlement_avail(l, s, p.idx),
    lambda l, s, p: _build_city_avail(l, s, p.idx),
    lambda l, s, p: _buy_dev_avail(l, s, None),
    lambda l, s, p: _knight_avail(l, s, (p.idx, p.target)),
    lambda l, s, p: _road_building_avail(l, s, None),
    lambda l, s, p: _yop_avail(l, s, (p.idx, p.target)),
    lambda l, s, p: _monopoly_avail(l, s, p.idx),
    lambda l, s, p: _maritime_avail(l, s, (p.idx, p.target)),
    lambda l, s, p: _end_turn_avail(l, s, None),
)


def apply_action(
    layout: BoardLayout,
    state: BoardState,
    action_type: jax.Array,
    params: ActionParams,
) -> tuple[BoardState, jax.Array]:
    """Apply ``action_type`` (single game) and return (new state, ActionResult code)."""
    return cast(
        "tuple[BoardState, jax.Array]",
        jax.lax.switch(action_type, _APPLY_BRANCHES, layout, state, params),
    )


def action_available(
    layout: BoardLayout,
    state: BoardState,
    action_type: jax.Array,
    params: ActionParams,
) -> jax.Array:
    """Legality of ``action_type`` with ``params`` (single game) as a scalar bool."""
    return cast(
        jax.Array,
        jax.lax.switch(action_type, _AVAIL_BRANCHES, layout, state, params),
    )
