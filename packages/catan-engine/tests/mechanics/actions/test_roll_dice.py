"""Tests for the vectorized RollDice action."""

import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, RollDice
from catan_engine.board import Board, make_board, to_main
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt


def test_success(roll_board: Board) -> None:
    state, result = RollDice()(roll_board, None)
    assert_expected_inline(
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


def test_invalid_not_roll_phase() -> None:
    board = to_main(make_board(seed=0))  # MAIN phase -> cannot roll
    before = np.asarray(board[1].phase)
    state, result = RollDice()(board, None)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.phase), before)


def test_invalid_already_rolled(roll_board: Board) -> None:
    layout, st = roll_board
    st = st._replace(has_rolled=st.has_rolled.at[0].set(1))
    _, result = RollDice()((layout, st), None)
    assert int(result[0]) == ActionResult.INVALID.value
