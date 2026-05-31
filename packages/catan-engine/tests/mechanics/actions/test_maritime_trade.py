"""Tests for the vectorized MaritimeTrade action."""

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, MaritimeTrade
from catan_engine.board import Board, give, make_board, set_phase, to_main
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt


def test_success(trade_board: Board) -> None:
    state, result = MaritimeTrade()(trade_board, (jnp.array([0]), jnp.array([1])))
    assert_expected_inline(
        fmt(
            result,
            sheep=int(state.player_resources[0, 0, 0]),
            wheat=int(state.player_resources[0, 0, 1]),
        ),
        """\
result=OK
sheep=0
wheat=1""",
    )


def test_invalid_wrong_phase(trade_board: Board) -> None:
    board = set_phase(trade_board, GamePhase.ROLL)
    before = np.asarray(board[1].player_resources)
    state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_give_equals_receive(trade_board: Board) -> None:
    before = np.asarray(trade_board[1].player_resources)
    state, result = MaritimeTrade()(trade_board, (jnp.array([0]), jnp.array([0])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_insufficient_resources() -> None:
    board = to_main(make_board())
    board = give(board, 0, [3, 0, 0, 0, 0])  # only 3 sheep, ratio is 4
    before = np.asarray(board[1].player_resources)
    state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)
