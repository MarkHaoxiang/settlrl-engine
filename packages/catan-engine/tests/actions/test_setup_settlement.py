"""Tests for the vectorized SetupSettlement action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import SetupSettlement
from catan_engine.board import make_board, place_settlement, set_phase
from catan_engine.resources import N_PLAYERS
from catan_engine.rules_vec import V_NBR
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


class TestSetupSettlement(TestCase):
    def test_success(self) -> None:
        board = make_board(seed=0)
        state, result = SetupSettlement()(board, jnp.array([0]))
        self.assertExpectedInline(
            fmt(
                result,
                owner=int(state.vertex_owner[0, 0]),
                kind=int(state.vertex_type[0, 0]),
                vp=int(state.victory_points[0, 0]),
                phase=str(GamePhase(int(state.phase[0]))),
                resources_total=int(np.asarray(state.player_resources[0, 0]).sum()),
            ),
            """\
result=OK
owner=1
kind=1
vp=1
phase=SETUP_ROAD
resources_total=0""",
        )

    def test_second_settlement_grants_resources(self) -> None:
        layout, st = make_board(seed=0)
        st = st._replace(setup_index=st.setup_index.at[0].set(N_PLAYERS))
        board = (layout, st)
        state, result = SetupSettlement()(board, jnp.array([0]))
        assert int(result[0]) == ActionResult.SUCCESS.value
        assert int(np.asarray(state.player_resources[0, 0]).sum()) > 0

    def test_invalid_wrong_phase(self) -> None:
        board = make_board(seed=0)
        board = set_phase(board, GamePhase.ROLL)
        before = np.asarray(board[1].vertex_owner)
        state, result = SetupSettlement()(board, jnp.array([0]))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.vertex_owner), before)

    def test_invalid_distance_rule(self) -> None:
        board = make_board(seed=0)
        board = place_settlement(board, 0, 0)
        neighbour = int(np.asarray(V_NBR)[0, 0])
        _, result = SetupSettlement()(board, jnp.array([neighbour]))
        assert int(result[0]) == ActionResult.INVALID.value
