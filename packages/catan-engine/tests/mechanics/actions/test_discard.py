"""Tests for the vectorized Discard action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.robber import discard_step
from catan_engine.board import Board, give, make_board
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt


def test_success(discard_board: Callable[..., Board]) -> None:
    board = discard_board(owed=4)
    state, result = discard_step(board, (jnp.array([0]), jnp.array([[4, 0, 0, 0, 0]])))
    assert_expected_inline(
        fmt(
            result,
            sheep=int(state.player_resources[0, 0, 0]),
            wheat=int(state.player_resources[0, 0, 1]),
            pending=int(state.pending_discard[0, 0]),
            phase=str(GamePhase(int(state.phase[0]))),
        ),
        """\
result=OK
sheep=0
wheat=4
pending=0
phase=MOVE_ROBBER""",
    )


def test_invalid_wrong_phase() -> None:
    board = make_board(seed=0)  # fresh SETUP-phase board
    board = give(board, 0, [4, 4, 0, 0, 0])
    before = np.asarray(board[1].player_resources)
    state, result = discard_step(board, (jnp.array([0]), jnp.array([[4, 0, 0, 0, 0]])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_wrong_count(discard_board: Callable[..., Board]) -> None:
    board = discard_board(owed=4)
    before = np.asarray(board[1].player_resources)
    state, result = discard_step(board, (jnp.array([0]), jnp.array([[3, 0, 0, 0, 0]])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_more_than_hand(discard_board: Callable[..., Board]) -> None:
    board = discard_board(owed=5)  # owe 5, but only hold 4 sheep
    before = np.asarray(board[1].player_resources)
    state, result = discard_step(board, (jnp.array([0]), jnp.array([[5, 0, 0, 0, 0]])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)
