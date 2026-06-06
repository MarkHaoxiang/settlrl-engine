"""Tests for the vectorized EndTurn action."""

import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.turn import end_turn_step
from catan_engine.board import Board, make_board, to_main
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt


def test_success() -> None:
    board = to_main(make_board())  # current_player 0, MAIN, has_rolled=1
    state, result = end_turn_step(board, None)
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


def test_rotation_wraps_at_n_players() -> None:
    # In a 2-player game the turn passes 0 -> 1 -> 0; players 2/3 never act.
    board = to_main(make_board(n_players=2), player=1)
    state, result = end_turn_step(board, None)
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.current_player[0]) == 0


def test_resets_turn_local_dev_and_road_state() -> None:
    # Set the per-turn flags EndTurn must clear: a dev card was played this turn,
    # a Knight was bought this turn (dev_bought), and free roads remain. The
    # dev_bought reset is what makes a just-bought card playable next turn.
    layout, st = to_main(make_board())
    st = st._replace(
        dev_played=st.dev_played.at[0].set(1),
        dev_bought=st.dev_bought.at[0, int(DevCard.KNIGHT)].set(1),
        free_roads=st.free_roads.at[0].set(2),
    )
    board: Board = (layout, st)

    state, result = end_turn_step(board, None)
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.dev_played[0]) == 0
    assert int(np.asarray(state.dev_bought[0]).sum()) == 0
    assert int(state.free_roads[0]) == 0


def test_invalid_setup_phase() -> None:
    board = make_board()  # fresh -> SETUP phase, cannot end turn
    before = np.asarray(board[1].current_player)
    state, result = end_turn_step(board, None)
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
    state, result = end_turn_step(board, None)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.current_player), before)
