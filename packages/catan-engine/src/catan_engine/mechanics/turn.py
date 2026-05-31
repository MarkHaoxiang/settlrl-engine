"""Turn-boundary mechanics: the ``EndTurn`` action core (single-game, traceable).

Ending a turn clears the per-turn flags (dice, dev-played, bought-this-turn,
free roads) and advances ``current_player``, returning the game to ROLL. (The
turn's *opening* move, RollDice, lives in ``dice`` with its production helpers.)
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.board import Board
from catan_engine.board.layout import BoardLayout
from catan_engine.board.resources import N_PLAYERS
from catan_engine.board.state import BoardState, GamePhase, tree_select
from catan_engine.mechanics.common import (
    INVALID,
    SUCCESS,
    Mask,
    ResultCode,
    main_after_roll,
)


def _end_turn_avail(layout: BoardLayout, state: BoardState, params: None) -> Mask:
    return main_after_roll(state)


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


def end_turn_available(board: Board, params: None = None) -> Mask:
    """``(batch,)`` legality of ending the current player's turn (no state change)."""
    return cast(Mask, _end_turn_avail_b(board[0], board[1], None))


def end_turn_step(board: Board, params: None = None) -> tuple[BoardState, ResultCode]:
    """End the current player's turn per game. Advances to the next player."""
    return cast(
        "tuple[BoardState, ResultCode]", _end_turn_apply_b(board[0], board[1], None)
    )
