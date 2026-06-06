"""Tests for the vectorized Discard action (one card per action)."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.robber import discard_step
from catan_engine.board import Board, give, make_board
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt

SHEEP, WHEAT, WOOD = 0, 1, 2


def test_single_card_decrements_owed(discard_board: Callable[..., Board]) -> None:
    board = discard_board(owed=4)
    state, result = discard_step(board, jnp.array([SHEEP]))
    assert_expected_inline(
        fmt(
            result,
            sheep=int(state.player_resources[0, 0, SHEEP]),
            wheat=int(state.player_resources[0, 0, WHEAT]),
            pending=int(state.pending_discard[0, 0]),
            phase=str(GamePhase(int(state.phase[0]))),
        ),
        """\
result=OK
sheep=3
wheat=4
pending=3
phase=DISCARD""",
    )


def test_last_card_advances_to_move_robber(
    discard_board: Callable[..., Board],
) -> None:
    board = discard_board(owed=1)
    state, result = discard_step(board, jnp.array([WHEAT]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.pending_discard[0, 0]) == 0
    assert int(state.phase[0]) == GamePhase.MOVE_ROBBER


def test_full_choice_over_a_sequence(discard_board: Callable[..., Board]) -> None:
    # Owe 4 from a 4 sheep + 4 wheat hand: choose 1 sheep then 3 wheat.
    board = discard_board(owed=4)
    for resource in (SHEEP, WHEAT, WHEAT, WHEAT):
        state, result = discard_step(board, jnp.array([resource]))
        assert int(result[0]) == ActionResult.SUCCESS.value
        board = (board[0], state)
    assert np.array_equal(
        np.asarray(board[1].player_resources[0, 0]), [3, 1, 0, 0, 0]
    )
    assert int(board[1].phase[0]) == GamePhase.MOVE_ROBBER


def test_next_owing_player_follows(discard_board: Callable[..., Board]) -> None:
    # Players 0 and 2 both owe; player 0 (lowest index) discards first, then
    # the prompt moves to player 2.
    board = discard_board(owed=1)
    board = give(board, 2, [0, 0, 2, 0, 0])
    layout, st = board
    st = st._replace(pending_discard=st.pending_discard.at[0, 2].set(1))
    board = (layout, st)

    state, result = discard_step(board, jnp.array([SHEEP]))  # player 0's card
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.phase[0]) == GamePhase.DISCARD  # player 2 still owes

    state, result = discard_step((layout, state), jnp.array([WOOD]))  # player 2's
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.player_resources[0, 2, WOOD]) == 1
    assert int(state.pending_discard[0, 2]) == 0
    assert int(state.phase[0]) == GamePhase.MOVE_ROBBER


def test_invalid_wrong_phase() -> None:
    board = make_board(seed=0)  # fresh SETUP-phase board
    board = give(board, 0, [4, 4, 0, 0, 0])
    before = np.asarray(board[1].player_resources)
    state, result = discard_step(board, jnp.array([SHEEP]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_resource_not_held(discard_board: Callable[..., Board]) -> None:
    board = discard_board(owed=4)  # holds only sheep + wheat
    before = np.asarray(board[1].player_resources)
    state, result = discard_step(board, jnp.array([WOOD]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_resource_out_of_range(discard_board: Callable[..., Board]) -> None:
    board = discard_board(owed=4)
    for bad in (-1, 5):
        state, result = discard_step(board, jnp.array([bad]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert int(state.pending_discard[0, 0]) == 4
