"""Tests for the vectorized RollDice action."""

import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import RollDice
from catan_engine.board import Board, make_board, place_settlement, set_phase, to_main
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _roll_ready() -> Board:
    """A ROLL-phase board (seed fixed -> deterministic dice) with one settlement."""
    board = set_phase(make_board(seed=0), GamePhase.ROLL)
    return place_settlement(board, 0, 0)


class TestRollDice(TestCase):
    def test_success(self) -> None:
        board = _roll_ready()
        state, result = RollDice()(board, None)
        self.assertExpectedInline(
            fmt(
                result,
                dice=int(state.dice_roll[0]),
                phase=str(GamePhase(int(state.phase[0]))),
                has_rolled=int(state.has_rolled[0]),
            ),
            """\
result=OK
dice=4
phase=MAIN
has_rolled=1""",
        )

    def test_invalid_not_roll_phase(self) -> None:
        board = to_main(make_board(seed=0))  # MAIN phase -> cannot roll
        before = np.asarray(board[1].phase)
        state, result = RollDice()(board, None)
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.phase), before)

    def test_invalid_already_rolled(self) -> None:
        layout, st = _roll_ready()
        st = st._replace(has_rolled=st.has_rolled.at[0].set(1))
        _, result = RollDice()((layout, st), None)
        assert int(result[0]) == ActionResult.INVALID.value
