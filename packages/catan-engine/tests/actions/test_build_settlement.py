"""Tests for the vectorized BuildSettlement action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import BuildSettlement
from catan_engine.board import give, set_phase
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt, settlement_fixture


class TestBuildSettlement(TestCase):
    def test_success(self) -> None:
        board, vertex = settlement_fixture()
        state, result = BuildSettlement()(board, jnp.array([vertex]))
        self.assertExpectedInline(
            fmt(
                result,
                owner=int(state.vertex_owner[0, vertex]),
                kind=int(state.vertex_type[0, vertex]),
                vp=int(state.victory_points[0, 0]),
                resources=int(np.asarray(state.player_resources[0, 0]).sum()),
            ),
            """\
result=OK
owner=1
kind=1
vp=2
resources=0""",
        )

    def test_invalid_wrong_phase(self) -> None:
        board, vertex = settlement_fixture()
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].vertex_owner)
        state, result = BuildSettlement()(board, jnp.array([vertex]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.vertex_owner), before)

    def test_invalid_not_connected(self) -> None:
        # A distant empty vertex with no adjacent road is not connected.
        board, _ = settlement_fixture()
        lonely = 40
        _, result = BuildSettlement()(board, jnp.array([lonely]))
        assert int(result[0]) == ActionResult.INVALID.value

    def test_invalid_cannot_afford(self) -> None:
        board, vertex = settlement_fixture()
        board = give(board, 0, [0, 0, 0, 0, 0])
        _, result = BuildSettlement()(board, jnp.array([vertex]))
        assert int(result[0]) == ActionResult.INVALID.value
