"""Turn-boundary mechanics: the ``EndTurn`` action core (single-game, traceable).

Ending a turn clears the per-turn flags (dice, dev-played, bought-this-turn,
free roads) and advances ``current_player``, returning the game to ROLL. (The
turn's *opening* move, RollDice, lives in ``dice`` with its production helpers.)
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from settlrl_engine.board import Board
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import (
    BoardState,
    BoolScalar,
    GamePhase,
    IntScalar,
    tree_select,
)
from settlrl_engine.mechanics import awards
from settlrl_engine.mechanics.common import (
    INVALID,
    SUCCESS,
    Mask,
    ResultCode,
    main_after_roll,
)


def _end_turn_avail(layout: BoardLayout, state: BoardState, params: None) -> BoolScalar:
    return main_after_roll(state)


def _end_turn_apply(
    layout: BoardLayout, state: BoardState, params: None, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    nxt = (state.current_player.astype(jnp.int32) + 1) % state.n_players
    cand = state._replace(
        dice_roll=jnp.uint8(0),
        has_rolled=jnp.uint8(0),
        dev_played=jnp.uint8(0),
        dev_bought=jnp.zeros_like(state.dev_bought),
        free_roads=jnp.uint8(0),
        current_player=nxt.astype(state.current_player.dtype),
        phase=jnp.uint8(GamePhase.ROLL),
    )
    return tree_select(available, cand, state), jnp.where(available, SUCCESS, INVALID)


_end_turn_avail_b = jax.jit(jax.vmap(_end_turn_avail, in_axes=(0, 0, None)))
_end_turn_apply_b = jax.jit(jax.vmap(_end_turn_apply, in_axes=(0, 0, None, 0)))


def end_turn_available(board: Board, params: None = None) -> Mask:
    """``(batch,)`` legality of ending the current player's turn (no state change)."""
    return cast(Mask, _end_turn_avail_b(board[0], board[1], None))


def end_turn_step(board: Board, params: None = None) -> tuple[BoardState, ResultCode]:
    """End the current player's turn per game. Advances to the next player.

    Resolves any win: a player who reached 10 VP out of turn claims victory at
    the start of their own turn (see :func:`awards.current_player_won`).
    """
    available = _end_turn_avail_b(board[0], board[1], None)
    state, result = _end_turn_apply_b(board[0], board[1], None, available)
    # No piece moved, so the award holders cannot change (the recompute inside
    # is a cheap no-op); only the turn-start win claim is live here.
    return cast(
        "tuple[BoardState, ResultCode]",
        awards.resolve_step_b(state, result, jnp.zeros_like(result, jnp.bool_)),
    )
