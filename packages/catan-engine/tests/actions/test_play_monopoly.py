"""Tests for the vectorized PlayMonopoly action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import PlayMonopoly
from catan_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    set_phase,
    to_main,
)
from catan_engine.dev_cards import DevCard
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _monopoly_fixture() -> Board:
    board = to_main(make_board())
    board = give(board, 0, [1, 0, 0, 0, 0])
    board = give(board, 1, [3, 0, 0, 0, 0])
    board = give(board, 2, [2, 0, 0, 0, 0])
    board = give_dev_card(board, 0, DevCard.MONOPOLY)
    return board


class TestPlayMonopoly(TestCase):
    def test_success(self) -> None:
        board = _monopoly_fixture()
        state, result = PlayMonopoly()(board, jnp.array([0]))
        self.assertExpectedInline(
            fmt(
                result,
                player0_sheep=int(state.player_resources[0, 0, 0]),
                player1_sheep=int(state.player_resources[0, 1, 0]),
                player2_sheep=int(state.player_resources[0, 2, 0]),
                dev_played=int(state.dev_played[0]),
                player0_monopoly=int(state.dev_hand[0, 0, DevCard.MONOPOLY]),
            ),
            """\
result=OK
player0_sheep=6
player1_sheep=0
player2_sheep=0
dev_played=1
player0_monopoly=0""",
        )

    def test_invalid_wrong_phase(self) -> None:
        board = _monopoly_fixture()
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].player_resources)
        state, result = PlayMonopoly()(board, jnp.array([0]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_no_card(self) -> None:
        board = to_main(make_board())
        board = give(board, 0, [1, 0, 0, 0, 0])
        before = np.asarray(board[1].player_resources)
        state, result = PlayMonopoly()(board, jnp.array([0]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_dev_already_played(self) -> None:
        board = _monopoly_fixture()
        layout, state = board
        board = (layout, state._replace(dev_played=state.dev_played.at[0].set(1)))
        before = np.asarray(board[1].player_resources)
        new_state, result = PlayMonopoly()(board, jnp.array([0]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(new_state.player_resources), before)

    def test_invalid_out_of_range(self) -> None:
        board = _monopoly_fixture()
        before = np.asarray(board[1].player_resources)
        state, result = PlayMonopoly()(board, jnp.array([-1]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)
