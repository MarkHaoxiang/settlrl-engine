"""Tests for the vectorized SetupRoad action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import SetupRoad
from catan_engine.board import make_board, place_road, place_settlement, set_phase
from catan_engine.rules_vec import EDGE_V, N_SETUP
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt

# First edge incident to vertex 0.
_E0 = int(np.where((np.asarray(EDGE_V) == 0).any(axis=1))[0][0])


def _setup_road_board() -> tuple:
    """SETUP_ROAD board: player 0 just placed a settlement at vertex 0."""
    board = set_phase(make_board(seed=0), GamePhase.SETUP_ROAD)
    return place_settlement(board, 0, 0)


class TestSetupRoad(TestCase):
    def test_success(self) -> None:
        board = _setup_road_board()
        state, result = SetupRoad()(board, jnp.array([_E0]))
        self.assertExpectedInline(
            fmt(
                result,
                edge_owner=int(state.edge_road[0, _E0]),
                setup_index=int(state.setup_index[0]),
                phase=str(GamePhase(int(state.phase[0]))),
                current_player=int(state.current_player[0]),
            ),
            """\
result=OK
edge_owner=1
setup_index=1
phase=SETUP_SETTLEMENT
current_player=1""",
        )

    def test_setup_complete(self) -> None:
        layout, st = _setup_road_board()
        st = st._replace(setup_index=st.setup_index.at[0].set(N_SETUP - 1))
        board = (layout, st)
        state, result = SetupRoad()(board, jnp.array([_E0]))
        assert int(result[0]) == ActionResult.SUCCESS.value
        assert int(state.setup_index[0]) == N_SETUP
        assert int(state.phase[0]) == GamePhase.ROLL
        assert int(state.current_player[0]) == 0

    def test_invalid_wrong_phase(self) -> None:
        board = _setup_road_board()
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].edge_road)
        state, result = SetupRoad()(board, jnp.array([_E0]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.edge_road), before)

    def test_invalid_edge_occupied(self) -> None:
        board = _setup_road_board()
        board = place_road(board, 0, _E0)
        before = np.asarray(board[1].edge_road)
        state, result = SetupRoad()(board, jnp.array([_E0]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.edge_road), before)

    def test_invalid_does_not_touch_settlement(self) -> None:
        board = _setup_road_board()
        # An edge with no endpoint owned by the player.
        edges = np.asarray(EDGE_V)
        far = int(np.where(~(edges == 0).any(axis=1))[0][0])
        before = np.asarray(board[1].edge_road)
        state, result = SetupRoad()(board, jnp.array([far]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.edge_road), before)
