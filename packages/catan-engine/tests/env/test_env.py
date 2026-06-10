"""Tests for the unified action dispatch and the batched AEC env
(catan_engine.env)."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from catan_engine import env
from catan_engine.mechanics.action import (
    ActionParams,
    ActionResult,
    ActionType,
)
from catan_engine.mechanics.flat import flat_available_b
from catan_engine.mechanics.placement import build_road_step
from catan_engine.board import make_board, replicate, set_phase
from catan_engine.env import (
    N_ACTION_TYPES,
    BatchedCatanEnv,
    Box,
    Discrete,
    Infos,
    Observation,
)
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import road_fixture


def _params(index: list[int], target: list[int], batch: int) -> ActionParams:
    return ActionParams(
        idx=jnp.asarray(index, dtype=jnp.int32),
        target=jnp.asarray(target, dtype=jnp.int32),
    )


class TestEnvStep:
    def test_dispatch_matches_direct_call(self) -> None:
        # BUILD_ROAD via env.step must equal calling BuildRoad directly.
        board, edge = road_fixture()
        atype = jnp.asarray([ActionType.BUILD_ROAD], dtype=jnp.int32)
        params = _params([edge], [-1], batch=1)

        state_env, result_env = env.step(board, atype, params)
        state_dir, result_dir = build_road_step(board, jnp.asarray([edge]))

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
            jnp.array(
                [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 4, 0]]
            )
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
        # agent_selection is cached and refreshed by step/reset, so drive one
        # step; END_TURN is illegal during DISCARD and leaves the poked state.
        e.step(jnp.asarray([int(ActionType.END_TURN)], jnp.int32), _batch_params([0]))
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
        from catan_engine.env.batched import _total_vp_b

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

    def test_space_and_info_keys_match_typed_dicts(self) -> None:
        # mypy ties observe() to the Observation TypedDict; this pins the
        # space descriptor (and Infos) to the same key sets so they can't drift.
        e = BatchedCatanEnv(batch_size=1, seed=11)
        assert set(e.observation_space()) == set(Observation.__annotations__)
        assert set(e.infos) == set(Infos.__annotations__)


class TestDiscardOneCard:
    """DISCARD is one card of one resource per step: the discarder chooses freely
    among held resources, repeating until the owed count reaches zero."""

    def _discard_env(self) -> BatchedCatanEnv:
        """Lane 0 in the DISCARD phase: player 0 holds 4 sheep + 4 wheat, owes 4."""
        e = BatchedCatanEnv(batch_size=1, seed=12, auto_reset=False)
        e._state = e._state._replace(
            phase=e._state.phase.at[0].set(int(GamePhase.DISCARD)),
            player_resources=e._state.player_resources.at[0, 0].set(
                jnp.asarray([4, 4, 0, 0, 0], dtype=jnp.uint8)
            ),
            pending_discard=e._state.pending_discard.at[0, 0].set(4),
        )
        # The step gate reads the cached flat legality, so refresh it after the
        # direct state surgery above.
        e._avail = flat_available_b(e._layout, e._state)
        return e

    @staticmethod
    def _discard(e: BatchedCatanEnv, resource: int) -> None:
        at = jnp.asarray([int(ActionType.DISCARD)], dtype=jnp.int32)
        e.step(at, _batch_params([resource]))

    def test_mask_offers_held_resources_only(self) -> None:
        e = self._discard_env()
        mask = np.asarray(e.available_indices(ActionType.DISCARD))
        assert np.array_equal(mask[0], [True, True, False, False, False])
        type_mask = np.asarray(e.action_mask())
        assert set(np.where(type_mask[0])[0]) == {int(ActionType.DISCARD)}

    def test_chosen_sequence_applies(self) -> None:
        # Greedy would strip sheep first; choose 1 sheep + 3 wheat instead.
        e = self._discard_env()
        for resource in (0, 1, 1, 1):
            self._discard(e, resource)
            assert int(e.infos["result"][0]) == ActionResult.SUCCESS.value
        assert np.array_equal(
            np.asarray(e._state.player_resources[0, 0]), [3, 1, 0, 0, 0]
        )
        assert int(e._state.pending_discard[0, 0]) == 0
        assert int(e._state.phase[0]) == GamePhase.MOVE_ROBBER

    def test_unheld_resource_rejected(self) -> None:
        e = self._discard_env()
        before = np.asarray(e._state.player_resources)
        self._discard(e, 2)  # wood: not held
        assert int(e.infos["result"][0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(e._state.player_resources), before)
        assert int(e._state.phase[0]) == GamePhase.DISCARD


class TestTwoPlayerMode:
    def test_agents_and_validation(self) -> None:
        e = BatchedCatanEnv(batch_size=1, seed=0, n_players=2)
        assert e.possible_agents == ["player_0", "player_1"]
        assert e.num_agents == 2
        for bad in (1, N_PLAYERS + 1):
            with pytest.raises(ValueError, match="n_players"):
                BatchedCatanEnv(n_players=bad)

    def test_setup_snake_and_first_rotation(self) -> None:
        # 2-player snake: settlements (and their roads) go 0, 1, 1, 0; then ROLL
        # returns to player 0. Drive each step with the first legal vertex/edge.
        e = BatchedCatanEnv(batch_size=1, seed=2, n_players=2, auto_reset=False)
        for placer in (0, 1, 1, 0):
            for at in (ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD):
                assert int(e.agent_selection[0]) == placer
                mask = np.asarray(e.available_indices(at))
                e.step(
                    jnp.asarray([int(at)], jnp.int32),
                    _batch_params([_first_legal(mask[0])]),
                )
        assert int(e._state.phase[0]) == GamePhase.ROLL
        assert int(e.agent_selection[0]) == 0

    def test_random_rollout_never_touches_unseated_players(self) -> None:
        e = BatchedCatanEnv(batch_size=2, seed=1, n_players=2)
        key = jax.random.key(0)
        for _ in range(120):
            key, sub = jax.random.split(key)
            at, params = e.random_actions(sub)
            e.step(at, params)
            assert (np.asarray(e.agent_selection) < 2).all()
        # The unseated rows (players 2/3) own nothing throughout.
        assert np.asarray(e._state.player_resources)[:, 2:].sum() == 0
        assert np.asarray(e._state.victory_points)[:, 2:].sum() == 0
        assert (np.asarray(e._state.vertex_owner) <= 2).all()
        assert (np.asarray(e._state.edge_road) <= 2).all()


def _states_equal(a: Any, b: Any) -> bool:
    """Full BoardState equality (the PRNG key compared on its raw data)."""
    for field in a._fields:
        x, y = getattr(a, field), getattr(b, field)
        if field == "key":
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    return True


class TestCacheGatedStep:
    """The env gates the chosen action with its cached flat legality and applies an
    avail-free core; this must match the self-validating functional ``step`` (which
    computes legality via the per-action ``action_available`` switch) exactly."""

    def test_env_step_matches_validating_functional_step(self) -> None:
        # auto_reset=False so a snapshot board and the env stay in lockstep.
        e = BatchedCatanEnv(batch_size=8, seed=3, auto_reset=False)
        key = jax.random.key(0)
        for _ in range(60):
            board = (e._layout, e._state)
            key, sub = jax.random.split(key)
            at, params = e.random_actions(sub)
            # Same action, same starting state -> identical transition (incl. the
            # stochastic roll/steal, which both consume state.key).
            f_state, _ = env.step(board, at, params)
            e.step(at, params)
            assert _states_equal(e._state, f_state)

    def test_env_step_gates_illegal_via_cache(self) -> None:
        # BUILD_CITY is illegal in the setup phase: the cache must reject it, the
        # core no-ops, and the result matches the validating path (INVALID).
        e = BatchedCatanEnv(batch_size=1, seed=0, auto_reset=False)
        board = (e._layout, e._state)
        before = np.asarray(e._state.vertex_owner)
        at = jnp.asarray([int(ActionType.BUILD_CITY)], dtype=jnp.int32)
        params = _batch_params([0])
        f_state, f_result = env.step(board, at, params)
        e.step(at, params)
        assert int(f_result[0]) == ActionResult.INVALID.value
        assert int(e.infos["result"][0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(e._state.vertex_owner), before)
        assert _states_equal(e._state, f_state)


class TestRollout:
    """``rollout(key, n)`` is one fused scan over the same per-step driver, so
    it must reproduce the ``random_actions`` + ``step`` loop bit-for-bit."""

    @pytest.mark.parametrize("track_beliefs", [False, True])
    def test_rollout_matches_step_loop(self, track_beliefs: bool) -> None:
        n_steps, key = 60, jax.random.key(7)
        loop = BatchedCatanEnv(batch_size=4, seed=5, track_beliefs=track_beliefs)
        fused = BatchedCatanEnv(batch_size=4, seed=5, track_beliefs=track_beliefs)

        total = jnp.zeros((4, loop.n_players), jnp.float32)
        k = key
        for _ in range(n_steps):
            k, sub = jax.random.split(k)
            at, params = loop.random_actions(sub)
            loop.step(at, params)
            total = total + loop.rewards

        cum = fused.rollout(key, n_steps)

        assert _states_equal(loop._state, fused._state)
        for name in ("_avail", "_vps", "_reward", "_terminations", "_result"):
            assert np.array_equal(
                np.asarray(getattr(loop, name)), np.asarray(getattr(fused, name))
            ), name
        assert np.array_equal(
            np.asarray(loop.agent_selection), np.asarray(fused.agent_selection)
        )
        assert np.array_equal(np.asarray(total), np.asarray(cum))
        if track_beliefs:
            for lf, ff in zip(loop.beliefs, fused.beliefs, strict=True):
                assert np.array_equal(np.asarray(lf), np.asarray(ff))

    def test_rollout_reward_counts_wins(self) -> None:
        # Sparse reward: the summed rollout reward counts terminal-step wins, so
        # over a long window it is non-negative integers (and usually nonzero).
        e = BatchedCatanEnv(batch_size=8, seed=0)
        cum = np.asarray(e.rollout(jax.random.key(1), 800))
        assert cum.shape == (8, e.n_players)
        assert (cum >= 0).all() and (cum == cum.astype(int)).all()
