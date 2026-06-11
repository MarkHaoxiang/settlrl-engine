"""Trade rules: the best bank/port exchange ratio for a resource, the
``MaritimeTrade`` action core, and the domestic-trade cores
(``ProposeTrade`` / ``AcceptTrade`` / ``RejectTrade``)."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.board import Board
from catan_engine.board.layout import N_VERTICES, PORT_V, BoardLayout, PortAllocVec
from catan_engine.board.port import Port
from catan_engine.board.resources import N_RESOURCES, bank_stock
from catan_engine.board.state import (
    NO_INDEX,
    BoardState,
    BoolScalar,
    GamePhase,
    IntScalar,
    VertexOwnerVec,
    to_u8,
    tree_select,
)
from catan_engine.mechanics.common import (
    INVALID,
    SUCCESS,
    Mask,
    ResultCode,
    TwoIndexParams,
    main_after_roll,
)

# Static port geometry, flattened to one slot per (port, vertex) pair.
_PORT_VERTS = PORT_V.reshape(-1)  # (2 * N_PORTS,) port-vertex ids
_PORT_SLOT = jnp.repeat(jnp.arange(PORT_V.shape[0]), 2)  # owning port per slot
_IS_PORT_VERTEX = jnp.zeros((N_VERTICES,), jnp.bool_).at[_PORT_VERTS].set(True)


def port_ratio(
    vertex_owner: VertexOwnerVec,
    port_allocation: PortAllocVec,
    player: IntScalar,
    give: IntScalar,
) -> IntScalar:
    """Best maritime ratio for giving ``give``: 4, or 3 (general), or 2 (match)."""
    # Scatter each port's (per-game) type onto its two vertices; the boolean
    # ``_IS_PORT_VERTEX`` mask, not a fill sentinel, marks non-port vertices.
    ptypes = port_allocation[_PORT_SLOT]  # (2 * N_PORTS,)
    vert_port = (
        jnp.zeros((N_VERTICES,), port_allocation.dtype).at[_PORT_VERTS].set(ptypes)
    )
    my_port = (vertex_owner == player + 1) & _IS_PORT_VERTEX
    general = jnp.any(my_port & (vert_port == Port.GENERAL))
    match = jnp.any(my_port & (vert_port == give))
    return jnp.where(match, 2, jnp.where(general, 3, 4)).astype(jnp.int32)


# ===========================================================================
# MaritimeTrade
# ===========================================================================


def _maritime_avail(
    layout: BoardLayout, state: BoardState, params: tuple[IntScalar, IntScalar]
) -> BoolScalar:
    give, receive = params
    player = state.current_player.astype(jnp.int32)
    g = jnp.clip(give, 0, N_RESOURCES - 1)
    r = jnp.clip(receive, 0, N_RESOURCES - 1)
    main = main_after_roll(state)
    give_ok = (give >= 0) & (give < N_RESOURCES)
    recv_ok = (receive >= 0) & (receive < N_RESOURCES)
    distinct = give != receive
    ratio = port_ratio(state.vertex_owner, layout.port_allocation, player, g)
    has_give = state.player_resources[player, g].astype(jnp.int32) >= ratio
    bank_ok = bank_stock(state.player_resources, r) >= 1
    return main & give_ok & recv_ok & distinct & has_give & bank_ok


def _maritime_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[IntScalar, IntScalar],
    available: BoolScalar,
) -> tuple[BoardState, IntScalar]:
    give, receive = params
    player = state.current_player.astype(jnp.int32)
    g = jnp.clip(give, 0, N_RESOURCES - 1)
    r = jnp.clip(receive, 0, N_RESOURCES - 1)
    ratio = port_ratio(state.vertex_owner, layout.port_allocation, player, g)
    res = state.player_resources.astype(jnp.int32)
    res = res.at[player, g].add(-ratio)
    res = res.at[player, r].add(1)
    cand = state._replace(player_resources=to_u8(res))
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_maritime_avail_b = jax.jit(jax.vmap(_maritime_avail))
_maritime_apply_b = jax.jit(jax.vmap(_maritime_apply))


def maritime_available(board: Board, params: TwoIndexParams) -> Mask:
    """``(batch,)`` legality of a (give, receive) maritime trade (no state change)."""
    return cast(Mask, _maritime_avail_b(board[0], board[1], params))


def maritime_step(
    board: Board, params: TwoIndexParams
) -> tuple[BoardState, ResultCode]:
    """Trade with the bank at the best available ratio (params: (give, receive))."""
    available = _maritime_avail_b(board[0], board[1], params)
    return cast(
        "tuple[BoardState, ResultCode]",
        _maritime_apply_b(board[0], board[1], params, available),
    )


# ===========================================================================
# Domestic trade: ProposeTrade -> AcceptTrade / RejectTrade
# ===========================================================================
#
# One card each way: the current player offers 1 card of ``give`` to ``partner``
# for 1 card of ``receive`` (give/receive packed into one index, see
# ``pack_trade``). Proposing is gated on *public* information only -- the
# proposer holds the give card and the partner's hand is non-empty -- so the
# legality mask never leaks the partner's hidden hand; whether the partner
# actually holds the asked-for card is checked only by AcceptTrade, whose mask
# is shown to the partner (who knows their own hand). The proposal parks the
# game in TRADE_RESPONSE, where the partner's only moves are Accept / Reject;
# either returns to MAIN, so the proposer may propose again (multiple trades
# per turn, per the rulebook). Disabled at 2 players.


def pack_trade(give: int, receive: int) -> int:
    """Pack a (give, receive) resource pair into ProposeTrade's primary index."""
    return give * N_RESOURCES + receive


def _unpack_trade(gr: IntScalar) -> tuple[IntScalar, IntScalar]:
    c = jnp.clip(gr, 0, N_RESOURCES * N_RESOURCES - 1)
    return c // N_RESOURCES, c % N_RESOURCES


def _propose_trade_avail(
    layout: BoardLayout, state: BoardState, params: tuple[IntScalar, IntScalar]
) -> BoolScalar:
    gr, partner = params
    n = state.n_players
    player = state.current_player.astype(jnp.int32)
    give, receive = _unpack_trade(gr)
    pc = jnp.clip(partner, 0, n - 1)
    in_range = (gr >= 0) & (gr < N_RESOURCES * N_RESOURCES)
    partner_ok = (partner >= 0) & (partner < n) & (partner != player)
    distinct = give != receive
    has_give = state.player_resources[player, give] >= 1
    partner_has_cards = state.player_resources[pc].astype(jnp.int32).sum() > 0
    return (
        main_after_roll(state)
        & (n > 2)
        & in_range
        & partner_ok
        & distinct
        & has_give
        & partner_has_cards
    )


def _propose_trade_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[IntScalar, IntScalar],
    available: BoolScalar,
) -> tuple[BoardState, IntScalar]:
    gr, partner = params
    give, receive = _unpack_trade(gr)
    pc = jnp.clip(partner, 0, state.n_players - 1)
    cand = state._replace(
        trade_partner=pc.astype(jnp.uint8),
        trade_give=give.astype(jnp.uint8),
        trade_receive=receive.astype(jnp.uint8),
        phase=jnp.uint8(GamePhase.TRADE_RESPONSE),
    )
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


def _accept_trade_avail(
    layout: BoardLayout, state: BoardState, params: None
) -> BoolScalar:
    partner = jnp.clip(state.trade_partner.astype(jnp.int32), 0, state.n_players - 1)
    receive = jnp.clip(state.trade_receive.astype(jnp.int32), 0, N_RESOURCES - 1)
    phase_ok = state.phase == GamePhase.TRADE_RESPONSE
    # The proposer still holds the give card (nothing moved since the propose);
    # only the partner's side needs checking.
    holds = state.player_resources[partner, receive] >= 1
    return phase_ok & holds


def _clear_trade(state: BoardState) -> BoardState:
    return state._replace(
        trade_partner=jnp.uint8(NO_INDEX),
        trade_give=jnp.uint8(0),
        trade_receive=jnp.uint8(0),
        phase=jnp.uint8(GamePhase.MAIN),
    )


def _accept_trade_apply(
    layout: BoardLayout, state: BoardState, params: None, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    proposer = state.current_player.astype(jnp.int32)
    partner = jnp.clip(state.trade_partner.astype(jnp.int32), 0, state.n_players - 1)
    give = jnp.clip(state.trade_give.astype(jnp.int32), 0, N_RESOURCES - 1)
    receive = jnp.clip(state.trade_receive.astype(jnp.int32), 0, N_RESOURCES - 1)
    res = state.player_resources.astype(jnp.int32)
    res = res.at[proposer, give].add(-1).at[partner, give].add(1)
    res = res.at[partner, receive].add(-1).at[proposer, receive].add(1)
    cand = _clear_trade(state._replace(player_resources=to_u8(res)))
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


def _reject_trade_avail(
    layout: BoardLayout, state: BoardState, params: None
) -> BoolScalar:
    return state.phase == GamePhase.TRADE_RESPONSE


def _reject_trade_apply(
    layout: BoardLayout, state: BoardState, params: None, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    cand = _clear_trade(state)
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_propose_trade_avail_b = jax.jit(jax.vmap(_propose_trade_avail))
_propose_trade_apply_b = jax.jit(jax.vmap(_propose_trade_apply))
_accept_trade_avail_b = jax.jit(jax.vmap(_accept_trade_avail, in_axes=(0, 0, None)))
_accept_trade_apply_b = jax.jit(jax.vmap(_accept_trade_apply, in_axes=(0, 0, None, 0)))
_reject_trade_avail_b = jax.jit(jax.vmap(_reject_trade_avail, in_axes=(0, 0, None)))
_reject_trade_apply_b = jax.jit(jax.vmap(_reject_trade_apply, in_axes=(0, 0, None, 0)))


def propose_trade_available(board: Board, params: TwoIndexParams) -> Mask:
    """``(batch,)`` legality of proposing a trade (params: (packed give/receive,
    partner); no state change)."""
    return cast(Mask, _propose_trade_avail_b(board[0], board[1], params))


def propose_trade_step(
    board: Board, params: TwoIndexParams
) -> tuple[BoardState, ResultCode]:
    """Propose a 1:1 trade to ``partner`` (params: (packed give/receive, partner)).

    Legality reads only public information: the proposer must hold the give
    card and the partner must hold *some* card; whether the partner holds the
    asked-for card is resolved by their Accept / Reject. Moves the game to
    TRADE_RESPONSE, where the partner acts. Never legal with 2 players.
    """
    available = _propose_trade_avail_b(board[0], board[1], params)
    return cast(
        "tuple[BoardState, ResultCode]",
        _propose_trade_apply_b(board[0], board[1], params, available),
    )


def accept_trade_available(board: Board, params: None = None) -> Mask:
    """``(batch,)`` legality of accepting the pending trade (no state change)."""
    return cast(Mask, _accept_trade_avail_b(board[0], board[1], None))


def accept_trade_step(
    board: Board, params: None = None
) -> tuple[BoardState, ResultCode]:
    """Accept the pending trade as the proposed-to partner.

    Swaps one give card from the proposer for one receive card from the
    partner, clears the proposal, and returns to MAIN. Legal only while the
    partner holds the asked-for card. Never wins.
    """
    available = _accept_trade_avail_b(board[0], board[1], None)
    return cast(
        "tuple[BoardState, ResultCode]",
        _accept_trade_apply_b(board[0], board[1], None, available),
    )


def reject_trade_available(board: Board, params: None = None) -> Mask:
    """``(batch,)`` legality of rejecting the pending trade (no state change)."""
    return cast(Mask, _reject_trade_avail_b(board[0], board[1], None))


def reject_trade_step(
    board: Board, params: None = None
) -> tuple[BoardState, ResultCode]:
    """Reject the pending trade as the proposed-to partner.

    Clears the proposal unchanged and returns to MAIN (the proposer may
    propose again). Always legal during TRADE_RESPONSE.
    """
    available = _reject_trade_avail_b(board[0], board[1], None)
    return cast(
        "tuple[BoardState, ResultCode]",
        _reject_trade_apply_b(board[0], board[1], None, available),
    )
