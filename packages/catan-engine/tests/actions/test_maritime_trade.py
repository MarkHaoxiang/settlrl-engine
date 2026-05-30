"""Tests for the vectorized MaritimeTrade action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import MaritimeTrade
from catan_engine.board import Board, give, make_board, set_phase, to_main
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _trade_fixture() -> Board:
    """MAIN board where player 0 holds 4 sheep (no port -> 4:1 bank trade)."""
    board = to_main(make_board())
    board = give(board, 0, [4, 0, 0, 0, 0])  # 4 sheep
    return board


class TestMaritimeTrade(TestCase):
    def test_success(self) -> None:
        board = _trade_fixture()
        state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
        self.assertExpectedInline(
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

    def test_invalid_wrong_phase(self) -> None:
        board = _trade_fixture()
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].player_resources)
        state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_give_equals_receive(self) -> None:
        board = _trade_fixture()
        before = np.asarray(board[1].player_resources)
        state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([0])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_insufficient_resources(self) -> None:
        board = to_main(make_board())
        board = give(board, 0, [3, 0, 0, 0, 0])  # only 3 sheep, ratio is 4
        before = np.asarray(board[1].player_resources)
        state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)
