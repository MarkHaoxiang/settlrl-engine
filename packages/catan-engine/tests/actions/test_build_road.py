"""Tests for the vectorized BuildRoad action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import BuildRoad
from catan_engine.board import set_phase
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt, road_fixture


class TestBuildRoad(TestCase):
    def test_success(self) -> None:
        board, edge = road_fixture()
        state, result = BuildRoad()(board, jnp.array([edge]))
        self.assertExpectedInline(
            fmt(
                result,
                edge_owner=int(state.edge_road[0, edge]),
                wood=int(state.player_resources[0, 0, 2]),
                brick=int(state.player_resources[0, 0, 3]),
                roads=int((np.asarray(state.edge_road[0]) == 1).sum()),
            ),
            """\
result=OK
edge_owner=1
wood=0
brick=0
roads=1""",
        )

    def test_invalid_wrong_phase(self) -> None:
        board, edge = road_fixture()
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].edge_road)
        state, result = BuildRoad()(board, jnp.array([edge]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.edge_road), before)

    def test_invalid_out_of_range(self) -> None:
        board, _ = road_fixture()
        before = np.asarray(board[1].edge_road)
        state, result = BuildRoad()(board, jnp.array([-1]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.edge_road), before)

    def test_invalid_cannot_afford(self) -> None:
        from catan_engine.board import give

        board, edge = road_fixture()
        board = give(board, 0, [0, 0, 0, 0, 0])  # no wood/brick, no free roads
        _, result = BuildRoad()(board, jnp.array([edge]))
        assert int(result[0]) == ActionResult.INVALID.value
