"""Maritime trade rules: the best bank/port exchange ratio for a resource.

Lives here rather than in ``port.py`` because the port *rule* needs the board
geometry (``layout.PORT_V``) and ``layout`` already imports ``port`` for the
``Port`` enum -- colocating would create an import cycle. ``port.py`` stays the
pure ``Port`` enum; ``trade.py`` is the rule over it.

Port membership is a per-vertex node feature derived by scattering over the dense
``PORT_V`` map (each port owns two vertices), so no padded vertex->port reverse
map / sentinel is needed.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.board import Board
from catan_engine.board.layout import N_VERTICES, PORT_V, BoardLayout, PortAllocVec
from catan_engine.board.port import Port
from catan_engine.board.resources import N_RESOURCES, bank_stock
from catan_engine.board.state import (
    BoardState,
    BoolScalar,
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
    vert_port = jnp.zeros((N_VERTICES,), port_allocation.dtype).at[_PORT_VERTS].set(
        ptypes
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
    layout: BoardLayout, state: BoardState, params: tuple[IntScalar, IntScalar]
) -> tuple[BoardState, IntScalar]:
    give, receive = params
    available = _maritime_avail(layout, state, params)
    player = state.current_player.astype(jnp.int32)
    g = jnp.clip(give, 0, N_RESOURCES - 1)
    r = jnp.clip(receive, 0, N_RESOURCES - 1)
    ratio = port_ratio(state.vertex_owner, layout.port_allocation, player, g)
    res = state.player_resources.astype(jnp.int32)
    res = res.at[player, g].add(-ratio)
    res = res.at[player, r].add(1)
    cand = state._replace(player_resources=to_u8(res))
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_maritime_avail_b = jax.jit(jax.vmap(_maritime_avail))
_maritime_apply_b = jax.jit(jax.vmap(_maritime_apply))


def maritime_available(board: Board, params: TwoIndexParams) -> Mask:
    """``(batch,)`` legality of a (give, receive) maritime trade (no state change)."""
    return cast(Mask, _maritime_avail_b(board[0], board[1], params))


def maritime_step(
    board: Board, params: TwoIndexParams
) -> tuple[BoardState, ResultCode]:
    """Trade with the bank at the best available ratio (params: (give, receive))."""
    return cast(
        "tuple[BoardState, ResultCode]", _maritime_apply_b(board[0], board[1], params)
    )
