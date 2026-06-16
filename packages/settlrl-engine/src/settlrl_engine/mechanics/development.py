"""Development-card rules and action cores (single-game, traceable).

The playability / weighted-draw primitives are separate from ``dev_cards.py``
(which holds the ``DevCard`` enum and deck counts) because they operate on
``BoardState`` and ``state`` already imports ``dev_cards`` -- colocating would
create an import cycle. The lower half holds the dev-card action cores:
``BuyDevelopmentCard``, ``PlayMonopoly``, ``PlayYearOfPlenty``,
``PlayRoadBuilding``, and ``PlayKnight`` (which composes the robber helpers in
``robber``). The cores apply only the core state change; the Largest Army award
and win check are resolved once per step by ``awards.resolve_step``.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from settlrl_engine.board import Board
from settlrl_engine.board.dev_cards import N_DEV_CARD_TYPES, DevCard, DevDeckVec
from settlrl_engine.board.layout import N_TILES, BoardLayout
from settlrl_engine.board.resources import N_RESOURCES, bank_stock
from settlrl_engine.board.state import (
    BoardState,
    BoolScalar,
    IntScalar,
    KeyScalar,
    to_u8,
    tree_select,
)
from settlrl_engine.mechanics import awards, robber
from settlrl_engine.mechanics.common import (
    DEV_CARD_COST_ARR,
    INVALID,
    SUCCESS,
    IndexParam,
    Mask,
    ResultCode,
    TwoIndexParams,
    can_afford,
    dev_play_window,
    main_after_roll,
    pay,
    roads_left,
)


def playable_dev(state: BoardState, player: IntScalar, card: int) -> BoolScalar:
    """True if ``player`` holds a playable copy of ``card`` (not bought this turn).

    Single-actor invariant: ``dev_hand`` is per-player ``(n_players, N_DEV_CARD_TYPES)``
    but ``dev_bought`` is per-GAME ``(N_DEV_CARD_TYPES,)``, so subtracting the
    game-global ``dev_bought[card]`` from a specific player's hand is only correct
    because exactly one player acts per turn (``current_player``) and ``dev_bought``
    resets on EndTurn. Do not make ``dev_bought`` per-player without auditing every
    caller -- this comparison relies on the acting player owning every "bought this
    turn" card.
    """
    held = state.dev_hand[player, card].astype(jnp.int32)
    bought = state.dev_bought[card].astype(jnp.int32)
    return held - bought > 0


def draw_dev_card(key: KeyScalar, dev_deck: DevDeckVec) -> tuple[KeyScalar, IntScalar]:
    """Draw one card type from ``dev_deck`` weighted by remaining counts.

    Returns ``(advanced key, card index)``. The probabilities fall back to
    uniform when the deck is empty so the draw is always well defined under a
    trace; callers gate the actual application on deck availability.
    """
    deck = dev_deck.astype(jnp.float32)
    total = deck.sum()
    probs = jnp.where(
        total > 0,
        deck / jnp.maximum(total, 1.0),
        jnp.full((N_DEV_CARD_TYPES,), 1.0 / N_DEV_CARD_TYPES),
    )
    key, sub = jax.random.split(key)
    card = jax.random.choice(sub, N_DEV_CARD_TYPES, p=probs)
    return key, card.astype(jnp.int32)


# ===========================================================================
# BuyDevelopmentCard
# ===========================================================================


def _buy_dev_avail(layout: BoardLayout, state: BoardState, params: None) -> BoolScalar:
    player = state.current_player.astype(jnp.int32)
    main = main_after_roll(state)
    deck_nonempty = state.dev_deck.astype(jnp.int32).sum() > 0
    afford = can_afford(state.player_resources[player], DEV_CARD_COST_ARR)
    return main & deck_nonempty & afford


def _buy_dev_apply(
    layout: BoardLayout, state: BoardState, params: IntScalar, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    """``params`` forces the drawn card type (1..N_DEV_CARD_TYPES, meaning
    type ``params - 1``); any other value (the flat table's 0) samples from
    the state key. Forcing an out-of-stock type is INVALID. The chance-node
    seam for stochastic search: the key advances identically either way."""
    player = state.current_player.astype(jnp.int32)
    key, sampled = draw_dev_card(state.key, state.dev_deck)
    forced = (params >= 1) & (params <= N_DEV_CARD_TYPES)
    card = jnp.where(forced, jnp.clip(params - 1, 0, N_DEV_CARD_TYPES - 1), sampled)
    available = available & (~forced | (state.dev_deck[card].astype(jnp.int32) > 0))
    new_deck = state.dev_deck.astype(jnp.int32).at[card].add(-1)
    new_hand = state.dev_hand.astype(jnp.int32).at[player, card].add(1)
    new_bought = state.dev_bought.astype(jnp.int32).at[card].add(1)
    cand = state._replace(
        player_resources=pay(state.player_resources, player, DEV_CARD_COST_ARR),
        dev_deck=to_u8(new_deck),
        dev_hand=to_u8(new_hand),
        dev_bought=to_u8(new_bought),
        key=key,
    )
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_buy_dev_avail_b = jax.jit(jax.vmap(_buy_dev_avail, in_axes=(0, 0, None)))
_buy_dev_apply_b = jax.jit(jax.vmap(_buy_dev_apply, in_axes=(0, 0, None, 0)))


def buy_development_card_available(board: Board, params: None = None) -> Mask:
    """``(batch,)`` legality of buying a development card (no state change)."""
    return cast(Mask, _buy_dev_avail_b(board[0], board[1], None))


def buy_development_card_step(
    board: Board, params: int | None = None
) -> tuple[BoardState, ResultCode]:
    """Buy a development card per game. Draws from ``state.dev_deck``.

    ``params`` forces the drawn type in every lane (1..N_DEV_CARD_TYPES,
    meaning type ``params - 1``); None samples. Resolves any win (a drawn
    Victory Point card can reach the threshold) via
    :func:`awards.resolve_step`.
    """
    available = _buy_dev_avail_b(board[0], board[1], None)
    state, result = _buy_dev_apply_b(
        board[0], board[1], jnp.int32(params or 0), available
    )
    return cast(
        "tuple[BoardState, ResultCode]",
        awards.resolve_step_b(state, result, jnp.zeros_like(result, jnp.bool_)),
    )


# ===========================================================================
# PlayMonopoly
# ===========================================================================


def _monopoly_avail(
    layout: BoardLayout, state: BoardState, resource: IntScalar
) -> BoolScalar:
    player = state.current_player.astype(jnp.int32)
    window = dev_play_window(state)
    in_range = (resource >= 0) & (resource < N_RESOURCES)
    has_card = playable_dev(state, player, DevCard.MONOPOLY)
    return window & in_range & has_card


def _monopoly_apply(
    layout: BoardLayout, state: BoardState, resource: IntScalar, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    player = state.current_player.astype(jnp.int32)
    r = jnp.clip(resource, 0, N_RESOURCES - 1)
    res = state.player_resources.astype(jnp.int32)  # (n_players, N_RESOURCES)
    taken = res[:, r].sum() - res[player, r]
    col = (
        jnp.zeros((state.n_players,), jnp.int32).at[player].set(res[player, r] + taken)
    )
    res = res.at[:, r].set(col)
    new_hand = state.dev_hand.astype(jnp.int32).at[player, DevCard.MONOPOLY].add(-1)
    cand = state._replace(
        dev_played=jnp.uint8(1),
        dev_hand=to_u8(new_hand),
        player_resources=to_u8(res),
    )
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_monopoly_avail_b = jax.jit(jax.vmap(_monopoly_avail))
_monopoly_apply_b = jax.jit(jax.vmap(_monopoly_apply))


def play_monopoly_available(board: Board, resource: IndexParam) -> Mask:
    """``(batch,)`` legality of playing Monopoly on ``resource`` (no state change)."""
    return cast(Mask, _monopoly_avail_b(board[0], board[1], resource))


def play_monopoly_step(
    board: Board, resource: IndexParam
) -> tuple[BoardState, ResultCode]:
    """Play Monopoly: take all of ``resource`` from every other player."""
    available = _monopoly_avail_b(board[0], board[1], resource)
    return cast(
        "tuple[BoardState, ResultCode]",
        _monopoly_apply_b(board[0], board[1], resource, available),
    )


# ===========================================================================
# PlayYearOfPlenty
# ===========================================================================


def _yop_avail(
    layout: BoardLayout, state: BoardState, params: tuple[IntScalar, IntScalar]
) -> BoolScalar:
    resource_a, resource_b = params
    player = state.current_player.astype(jnp.int32)
    ca = jnp.clip(resource_a, 0, N_RESOURCES - 1)
    cb = jnp.clip(resource_b, 0, N_RESOURCES - 1)
    window = dev_play_window(state)
    has_card = playable_dev(state, player, DevCard.YEAR_OF_PLENTY)
    a_ok = (resource_a >= 0) & (resource_a < N_RESOURCES)
    b_ok = (resource_b >= 0) & (resource_b < N_RESOURCES)
    same = resource_a == resource_b
    need_a = 1 + same.astype(jnp.int32)
    bank_a = bank_stock(state.player_resources, ca) >= need_a
    bank_b = same | (bank_stock(state.player_resources, cb) >= 1)
    return window & has_card & a_ok & b_ok & bank_a & bank_b


def _yop_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[IntScalar, IntScalar],
    available: BoolScalar,
) -> tuple[BoardState, IntScalar]:
    resource_a, resource_b = params
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
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_yop_avail_b = jax.jit(jax.vmap(_yop_avail))
_yop_apply_b = jax.jit(jax.vmap(_yop_apply))


def play_year_of_plenty_available(board: Board, params: TwoIndexParams) -> Mask:
    """``(batch,)`` legality of Year of Plenty (params: (a, b)) (no state change)."""
    return cast(Mask, _yop_avail_b(board[0], board[1], params))


def play_year_of_plenty_step(
    board: Board, params: TwoIndexParams
) -> tuple[BoardState, ResultCode]:
    """Play Year of Plenty (params: (a, b)); ``a == b`` draws two of one kind."""
    available = _yop_avail_b(board[0], board[1], params)
    return cast(
        "tuple[BoardState, ResultCode]",
        _yop_apply_b(board[0], board[1], params, available),
    )


# ===========================================================================
# PlayRoadBuilding
# ===========================================================================


def _road_building_avail(
    layout: BoardLayout, state: BoardState, params: None
) -> BoolScalar:
    player = state.current_player.astype(jnp.int32)
    window = dev_play_window(state)
    has_card = playable_dev(state, player, DevCard.ROAD_BUILDING)
    return window & has_card


def _road_building_apply(
    layout: BoardLayout, state: BoardState, params: None, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
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
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_road_building_avail_b = jax.jit(jax.vmap(_road_building_avail, in_axes=(0, 0, None)))
_road_building_apply_b = jax.jit(
    jax.vmap(_road_building_apply, in_axes=(0, 0, None, 0))
)


def play_road_building_available(board: Board, params: None = None) -> Mask:
    """``(batch,)`` legality of playing Road Building (no state change)."""
    return cast(Mask, _road_building_avail_b(board[0], board[1], None))


def play_road_building_step(
    board: Board, params: None = None
) -> tuple[BoardState, ResultCode]:
    """Play Road Building per game. Grants up to 2 free roads."""
    available = _road_building_avail_b(board[0], board[1], None)
    return cast(
        "tuple[BoardState, ResultCode]",
        _road_building_apply_b(board[0], board[1], None, available),
    )


# ===========================================================================
# PlayKnight
# ===========================================================================


def _knight_avail(
    layout: BoardLayout, state: BoardState, params: tuple[IntScalar, IntScalar]
) -> BoolScalar:
    tile, victim = params
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    window = dev_play_window(state)
    has_card = playable_dev(state, player, DevCard.KNIGHT)
    tile_in_range = (tile >= 0) & (tile < N_TILES)
    tile_moves = tile != state.robber
    valid_victim = robber.valid_robber_victim(state, t, player, victim)
    return window & has_card & tile_in_range & tile_moves & valid_victim


def _knight_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[IntScalar, IntScalar],
    available: BoolScalar,
) -> tuple[BoardState, IntScalar]:
    tile, victim = params
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
    cand = robber.apply_steal(cand, player, victim)
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_knight_avail_b = jax.jit(jax.vmap(_knight_avail))
_knight_apply_b = jax.jit(jax.vmap(_knight_apply))


def play_knight_available(board: Board, params: TwoIndexParams) -> Mask:
    """``(batch,)`` legality of a (tile, victim) Knight play (no state change)."""
    return cast(Mask, _knight_avail_b(board[0], board[1], params))


def play_knight_step(
    board: Board, params: TwoIndexParams
) -> tuple[BoardState, ResultCode]:
    """Play a Knight (params: (tile, victim)); ``victim == -1`` steals from no one.

    Moves the robber and steals; the Largest Army award and any win it brings are
    resolved via :func:`awards.resolve_step`.
    """
    available = _knight_avail_b(board[0], board[1], params)
    state, result = _knight_apply_b(board[0], board[1], params, available)
    return cast(
        "tuple[BoardState, ResultCode]",
        awards.resolve_step_b(state, result, jnp.zeros_like(result, jnp.bool_)),
    )
