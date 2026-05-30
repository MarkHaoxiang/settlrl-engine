"""Tests for the vectorized EndTurn action."""

import numpy as np
from expecttest import assert_expected_inline

from catan_engine.action import ActionResult, EndTurn
from catan_engine.board import Board, make_board, to_main
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def test_success() -> None:
    board = to_main(make_board())  # current_player 0, MAIN, has_rolled=1
    state, result = EndTurn()(board, None)
    assert_expected_inline(
        fmt(
            result,
            current_player=int(state.current_player[0]),
            phase=str(GamePhase(int(state.phase[0]))),
            has_rolled=int(state.has_rolled[0]),
        ),
        """\
result=OK
current_player=1
phase=ROLL
has_rolled=0""",
    )


def test_invalid_setup_phase() -> None:
    board = make_board()  # fresh -> SETUP phase, cannot end turn
    before = np.asarray(board[1].current_player)
    state, result = EndTurn()(board, None)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.current_player), before)


def test_invalid_not_rolled() -> None:
    layout, st = make_board()
    st = st._replace(
        phase=st.phase.at[0].set(int(GamePhase.MAIN)),
        has_rolled=st.has_rolled.at[0].set(0),
    )
    board: Board = (layout, st)
    before = np.asarray(st.current_player)
    state, result = EndTurn()(board, None)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.current_player), before)
