"""Tests for the unified action dispatch (catan_engine.env)."""

import jax.numpy as jnp
import numpy as np

from catan_engine import env
from catan_engine.action import ActionParams, ActionResult, ActionType, BuildRoad
from catan_engine.board import make_board, replicate, set_phase
from catan_engine.resources import N_RESOURCES
from catan_engine.state import GamePhase
from tests.actions.fixtures import road_fixture


def _params(index: list[int], target: list[int], batch: int) -> ActionParams:
    return ActionParams(
        idx=jnp.asarray(index, dtype=jnp.int32),
        target=jnp.asarray(target, dtype=jnp.int32),
        resources=jnp.zeros((batch, N_RESOURCES), dtype=jnp.int32),
    )


class TestEnvStep:
    def test_dispatch_matches_direct_call(self) -> None:
        # BUILD_ROAD via env.step must equal calling BuildRoad directly.
        board, edge = road_fixture()
        atype = jnp.asarray([ActionType.BUILD_ROAD], dtype=jnp.int32)
        params = _params([edge], [-1], batch=1)

        state_env, result_env = env.step(board, atype, params)
        state_dir, result_dir = BuildRoad()(board, jnp.asarray([edge]))

        assert int(result_env[0]) == int(result_dir[0]) == ActionResult.SUCCESS.value
        assert np.array_equal(
            np.asarray(state_env.edge_road), np.asarray(state_dir.edge_road)
        )

    def test_heterogeneous_batch(self) -> None:
        # Two games, two different actions in one batched step.
        board, edge = road_fixture()
        board2 = replicate(board, 2)
        atype = jnp.asarray(
            [ActionType.BUILD_ROAD, ActionType.END_TURN], dtype=jnp.int32
        )
        params = _params([edge, 0], [-1, -1], batch=2)

        state, result = env.step(board2, atype, params)

        # Game 0 built a road; game 1 ended its turn.
        assert int(result[0]) == ActionResult.SUCCESS.value
        assert int(state.edge_road[0, edge]) == 1
        assert int(result[1]) == ActionResult.SUCCESS.value
        assert int(state.phase[1]) == GamePhase.ROLL
        assert int(state.current_player[1]) == 1

    def test_available_matches_and_illegal_is_unchanged(self) -> None:
        board, edge = road_fixture()
        atype = jnp.asarray([ActionType.BUILD_ROAD], dtype=jnp.int32)

        legal = _params([edge], [-1], batch=1)
        assert bool(env.available(board, atype, legal)[0])

        # Out-of-range edge -> illegal: INVALID and state untouched.
        before = np.asarray(board[1].edge_road)
        bad = _params([-1], [-1], batch=1)
        assert not bool(env.available(board, atype, bad)[0])
        state, result = env.step(board, atype, bad)
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.edge_road), before)

    def test_roll_dice_via_dispatch(self) -> None:
        # Parameterless action dispatches and transitions ROLL -> MAIN (no 7 here).
        board = set_phase(make_board(seed=0), GamePhase.ROLL)  # has_rolled = 0
        atype = jnp.asarray([ActionType.ROLL_DICE], dtype=jnp.int32)
        state, result = env.step(board, atype, _params([0], [0], batch=1))
        assert int(result[0]) == ActionResult.SUCCESS.value
        assert int(state.has_rolled[0]) == 1
