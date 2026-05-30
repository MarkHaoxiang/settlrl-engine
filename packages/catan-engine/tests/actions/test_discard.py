"""Tests for the vectorized Discard action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import Discard
from catan_engine.board import Board, give, make_board, set_phase
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _discard_fixture(owed: int = 4) -> Board:
    """DISCARD board where player 0 holds 8 cards and owes ``owed`` discards."""
    board = set_phase(make_board(seed=0), GamePhase.DISCARD)
    board = give(board, 0, [4, 4, 0, 0, 0])  # 8 cards
    layout, st = board
    st = st._replace(pending_discard=st.pending_discard.at[0, 0].set(owed))
    return (layout, st)


class TestDiscard(TestCase):
    def test_success(self) -> None:
        board = _discard_fixture(owed=4)
        state, result = Discard()(board, (jnp.array([0]), jnp.array([[4, 0, 0, 0, 0]])))
        self.assertExpectedInline(
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

    def test_invalid_wrong_phase(self) -> None:
        board = make_board(seed=0)  # fresh SETUP-phase board
        board = give(board, 0, [4, 4, 0, 0, 0])
        before = np.asarray(board[1].player_resources)
        state, result = Discard()(board, (jnp.array([0]), jnp.array([[4, 0, 0, 0, 0]])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_wrong_count(self) -> None:
        board = _discard_fixture(owed=4)
        before = np.asarray(board[1].player_resources)
        state, result = Discard()(board, (jnp.array([0]), jnp.array([[3, 0, 0, 0, 0]])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_more_than_hand(self) -> None:
        board = _discard_fixture(owed=5)  # owe 5, but only hold 4 sheep
        before = np.asarray(board[1].player_resources)
        state, result = Discard()(board, (jnp.array([0]), jnp.array([[5, 0, 0, 0, 0]])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)
