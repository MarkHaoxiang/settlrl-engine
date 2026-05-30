"""Tests for the vectorized BuildCity action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import BuildCity
from catan_engine.board import (
    Board,
    give,
    make_board,
    place_settlement,
    set_phase,
    to_main,
)
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _city_fixture() -> tuple[Board, int]:
    """MAIN board where player 0 owns a settlement at vertex 0 with one city's worth."""
    board = to_main(make_board())
    board = place_settlement(board, 0, 0)
    board = give(board, 0, [0, 2, 0, 0, 3])  # one city's worth: 2 wheat + 3 ore
    return board, 0


class TestBuildCity(TestCase):
    def test_success(self) -> None:
        board, vertex = _city_fixture()
        state, result = BuildCity()(board, jnp.array([vertex]))
        self.assertExpectedInline(
            fmt(
                result,
                kind=int(state.vertex_type[0, vertex]),
                vp=int(state.victory_points[0, 0]),
                wheat=int(state.player_resources[0, 0, 1]),
                ore=int(state.player_resources[0, 0, 4]),
            ),
            """\
result=OK
kind=2
vp=2
wheat=0
ore=0""",
        )

    def test_invalid_wrong_phase(self) -> None:
        board, vertex = _city_fixture()
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].vertex_type)
        state, result = BuildCity()(board, jnp.array([vertex]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.vertex_type), before)

    def test_invalid_no_own_settlement(self) -> None:
        # A distant empty vertex holds no settlement of the player's.
        board, _ = _city_fixture()
        lonely = 40
        before = np.asarray(board[1].vertex_type)
        state, result = BuildCity()(board, jnp.array([lonely]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.vertex_type), before)

    def test_invalid_cannot_afford(self) -> None:
        board, vertex = _city_fixture()
        board = give(board, 0, [0, 0, 0, 0, 0])
        _, result = BuildCity()(board, jnp.array([vertex]))
        assert int(result[0]) == ActionResult.INVALID.value
