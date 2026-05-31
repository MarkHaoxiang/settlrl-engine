"""Tests for the unified action dispatch and the batched AEC env
(catan_engine.env)."""

import jax.numpy as jnp
import numpy as np
import pytest

from catan_engine import env
from catan_engine.action import ActionParams, ActionResult, ActionType, BuildRoad
from catan_engine.board import make_board, replicate, set_phase
from catan_engine.env import N_ACTION_TYPES, BatchedCatanEnv, Box, Discrete
from catan_engine.resources import N_PLAYERS, N_RESOURCES
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


def _batch_params(idx: list[int], target: list[int] | None = None) -> ActionParams:
    b = len(idx)
    tgt = target if target is not None else [0] * b
    return ActionParams(
        idx=jnp.asarray(idx, dtype=jnp.int32),
        target=jnp.asarray(tgt, dtype=jnp.int32),
        resources=jnp.zeros((b, N_RESOURCES), dtype=jnp.int32),
    )


def _first_legal(mask_row: np.ndarray) -> int:
    return int(np.where(mask_row)[0][0])


class TestBatchedCatanEnv:
    def test_reset_shapes_and_agents(self) -> None:
        e = BatchedCatanEnv(batch_size=3, seed=1)
        assert e.agents == [f"player_{i}" for i in range(N_PLAYERS)]
        assert e.num_agents == N_PLAYERS
        obs, reward, term, trunc, info = e.last()
        assert reward.shape == (3,) and term.shape == (3,) and trunc.shape == (3,)
        assert not bool(term.any())
        assert obs["vertex_owner"].shape == (3, 54)
        assert obs["self_resources"].shape == (3, N_RESOURCES)
        assert info["action_mask"].shape == (3, N_ACTION_TYPES)
        # Fresh games are in setup; player 0 acts in every lane.
        assert np.array_equal(np.asarray(e.agent_selection), [0, 0, 0])

    def test_setup_phase_only_allows_setup_settlement(self) -> None:
        e = BatchedCatanEnv(batch_size=2, seed=4)
        mask = np.asarray(e.action_mask())
        for b in range(2):
            legal = set(np.where(mask[b])[0])
            assert legal == {int(ActionType.SETUP_SETTLEMENT)}
        # Every vertex is a legal opening settlement on an empty board.
        vmask = np.asarray(e.available_indices(ActionType.SETUP_SETTLEMENT))
        assert vmask.all()

    def test_step_advances_setup(self) -> None:
        e = BatchedCatanEnv(batch_size=2, seed=5)
        vmask = np.asarray(e.available_indices(ActionType.SETUP_SETTLEMENT))
        idx = [_first_legal(vmask[b]) for b in range(2)]
        at = jnp.full((2,), int(ActionType.SETUP_SETTLEMENT), jnp.int32)
        e.step(at, _batch_params(idx))
        assert np.array_equal(np.asarray(e._state.phase), [GamePhase.SETUP_ROAD] * 2)
        assert np.array_equal(np.asarray(e.infos["result"]), [0, 0])

    def test_partial_observation_hides_opponent_hands(self) -> None:
        e = BatchedCatanEnv(batch_size=1, seed=6)
        # Give every player a distinct hand.
        res = e._state.player_resources.at[0].set(
            jnp.array([[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 4, 0]])
        )
        e._state = e._state._replace(player_resources=res)
        obs0 = e.observe(0)
        obs2 = e.observe("player_2")
        # Each observer sees only its own composition...
        assert np.array_equal(np.asarray(obs0["self_resources"][0]), [1, 0, 0, 0, 0])
        assert np.array_equal(np.asarray(obs2["self_resources"][0]), [0, 0, 3, 0, 0])
        # ...but the public hand_size counts are identical across observers.
        assert np.array_equal(
            np.asarray(obs0["hand_size"]), np.asarray(obs2["hand_size"])
        )
        assert np.array_equal(np.asarray(obs0["hand_size"][0]), [1, 2, 3, 4])

    def test_agent_selection_tracks_discarder(self) -> None:
        e = BatchedCatanEnv(batch_size=1, seed=7)
        # DISCARD phase with only player 2 owing -> player 2 is the acting agent.
        e._state = e._state._replace(
            phase=e._state.phase.at[0].set(int(GamePhase.DISCARD)),
            current_player=e._state.current_player.at[0].set(0),
            pending_discard=e._state.pending_discard.at[0, 2].set(4),
        )
        assert int(e.agent_selection[0]) == 2

    def test_auto_reset_only_terminated_lane(self) -> None:
        e = BatchedCatanEnv(batch_size=3, seed=8, reward="sparse")
        e._state = e._state._replace(
            victory_points=e._state.victory_points.at[1, 0].set(10),
            vertex_owner=e._state.vertex_owner.at[1, 5].set(2),
        )
        at = jnp.full((3,), int(ActionType.END_TURN), jnp.int32)  # no-op in setup
        e.step(at, _batch_params([0, 0, 0]))
        term = np.asarray(e.terminations)[:, 0]
        assert list(term) == [False, True, False]
        # Winner rewarded; terminated lane reset to a fresh board.
        assert float(e.rewards[1, 0]) == 1.0
        assert float(e.rewards[1, 1]) == 0.0
        assert int(e._state.vertex_owner[1, 5]) == 0
        assert int(e._state.victory_points[1, 0]) == 0

    def test_vp_delta_reward(self) -> None:
        from catan_engine.env import _total_vp_b

        e = BatchedCatanEnv(batch_size=1, seed=9, reward="vp_delta")
        e._vps = _total_vp_b(e._state)
        vmask = np.asarray(e.available_indices(ActionType.SETUP_SETTLEMENT))
        idx = [_first_legal(vmask[0])]
        at = jnp.full((1,), int(ActionType.SETUP_SETTLEMENT), jnp.int32)
        e.step(at, _batch_params(idx))
        # Placing a settlement is +1 building VP for the acting player (player 0).
        assert float(e.rewards[0, 0]) == 1.0
        assert float(e.rewards[0, 1]) == 0.0

    def test_available_indices_rejects_non_index_action(self) -> None:
        e = BatchedCatanEnv(batch_size=1, seed=10)
        with pytest.raises(ValueError, match="no single primary-index domain"):
            e.available_indices(ActionType.MARITIME_TRADE)

    def test_invalid_reward_mode(self) -> None:
        with pytest.raises(ValueError, match="reward must be"):
            BatchedCatanEnv(batch_size=1, reward="bogus")

    def test_spaces_descriptors(self) -> None:
        e = BatchedCatanEnv(batch_size=1, seed=11)
        aspace = e.action_space()
        assert aspace["action_type"] == Discrete(N_ACTION_TYPES)
        ospace = e.observation_space()
        assert isinstance(ospace["vertex_owner"], Box)
        assert ospace["vertex_owner"].shape == (54,)
