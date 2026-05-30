"""Tests for the vectorized PlayYearOfPlenty action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import PlayYearOfPlenty
from catan_engine.board import Board, give_dev_card, make_board, set_phase, to_main
from catan_engine.dev_cards import DevCard
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _yop_fixture() -> Board:
    """MAIN board where player 0 holds a Year of Plenty card and no resources."""
    board = to_main(make_board())
    board = give_dev_card(board, 0, DevCard.YEAR_OF_PLENTY)
    return board


class TestPlayYearOfPlenty(TestCase):
    def test_success(self) -> None:
        board = _yop_fixture()
        # Take wood (2) and brick (3).
        state, result = PlayYearOfPlenty()(board, (jnp.array([2]), jnp.array([3])))
        self.assertExpectedInline(
            fmt(
                result,
                wood=int(state.player_resources[0, 0, 2]),
                brick=int(state.player_resources[0, 0, 3]),
                dev_played=int(state.dev_played[0]),
                yop_hand=int(state.dev_hand[0, 0, DevCard.YEAR_OF_PLENTY]),
            ),
            """\
result=OK
wood=1
brick=1
dev_played=1
yop_hand=0""",
        )

    def test_success_same_resource(self) -> None:
        board = _yop_fixture()
        # a == b: take two sheep (0).
        state, result = PlayYearOfPlenty()(board, (jnp.array([0]), jnp.array([0])))
        assert int(result[0]) == ActionResult.SUCCESS.value
        assert int(state.player_resources[0, 0, 0]) == 2

    def test_invalid_wrong_phase(self) -> None:
        board = _yop_fixture()
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].player_resources)
        state, result = PlayYearOfPlenty()(board, (jnp.array([2]), jnp.array([3])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_no_card(self) -> None:
        board = to_main(make_board())  # no Year of Plenty card
        before = np.asarray(board[1].player_resources)
        state, result = PlayYearOfPlenty()(board, (jnp.array([2]), jnp.array([3])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_dev_already_played(self) -> None:
        layout, state = _yop_fixture()
        state = state._replace(dev_played=state.dev_played.at[0].set(1))
        board = (layout, state)
        before = np.asarray(state.player_resources)
        state, result = PlayYearOfPlenty()(board, (jnp.array([2]), jnp.array([3])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_out_of_range(self) -> None:
        board = _yop_fixture()
        before = np.asarray(board[1].player_resources)
        state, result = PlayYearOfPlenty()(board, (jnp.array([7]), jnp.array([3])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)
