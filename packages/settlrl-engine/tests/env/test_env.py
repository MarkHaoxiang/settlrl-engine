"""Tests for the unified action dispatch and the batched AEC env
(settlrl_engine.env)."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from settlrl_engine import env
from settlrl_engine.board import make_board, replicate, set_phase
from settlrl_engine.board.resources import N_PLAYERS, N_RESOURCES
from settlrl_engine.board.state import GamePhase
from settlrl_engine.env import (
    N_ACTION_TYPES,
    BatchedSettlrlEnv,
    Infos,
    Observation,
)
from settlrl_engine.mechanics.action import (
    ActionParams,
    ActionResult,
    ActionType,
)
from settlrl_engine.mechanics.flat import flat_available_b
from settlrl_engine.mechanics.placement import build_road_step

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


class TestBatchedSettlrlEnv:
    def test_reset_shapes_and_agents(self) -> None:
        e = BatchedSettlrlEnv(batch_size=3, seed=1)
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
        e = BatchedSettlrlEnv(batch_size=2, seed=4)
        mask = np.asarray(e.action_mask())
        for b in range(2):
            legal = set(np.where(mask[b])[0])
            assert legal == {int(ActionType.SETUP_SETTLEMENT)}
        # Every vertex is a legal opening settlement on an empty board.
        vmask = np.asarray(e.available_indices(ActionType.SETUP_SETTLEMENT))
        assert vmask.all()

    def test_step_advances_setup(self) -> None:
        e = BatchedSettlrlEnv(batch_size=2, seed=5)
        vmask = np.asarray(e.available_indices(ActionType.SETUP_SETTLEMENT))
        idx = [_first_legal(vmask[b]) for b in range(2)]
        at = jnp.full((2,), int(ActionType.SETUP_SETTLEMENT), jnp.int32)
        e.step(at, _batch_params(idx))
        assert np.array_equal(np.asarray(e._state.phase), [GamePhase.SETUP_ROAD] * 2)
        assert np.array_equal(np.asarray(e.infos["result"]), [0, 0])

    def test_partial_observation_hides_opponent_hands(self) -> None:
        e = BatchedSettlrlEnv(batch_size=1, seed=6)
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
        e = BatchedSettlrlEnv(batch_size=1, seed=7)
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
        e = BatchedSettlrlEnv(batch_size=3, seed=8, reward="sparse")
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

    def test_no_auto_reset_freezes_terminated_lane(self) -> None:
        # auto_reset=False: a finished lane freezes at its terminal board. The
        # win is credited exactly once (on the step that reaches it); further
        # steps neither mutate the lane nor re-credit the sparse reward.
        from settlrl_engine.env.batched import _total_vp_b

        e = BatchedSettlrlEnv(batch_size=1, seed=8, n_players=2, auto_reset=False)
        # Player 1 sits at 10 VP off-turn; player 0 (current, MAIN) ends the
        # turn, so player 1 claims the win at their turn start (rulebook p.5).
        e._state = e._state._replace(
            phase=e._state.phase.at[0].set(int(GamePhase.MAIN)),
            has_rolled=e._state.has_rolled.at[0].set(1),
            current_player=e._state.current_player.at[0].set(0),
            victory_points=e._state.victory_points.at[0, 1].set(10),
        )
        # Refresh the caches the step gate reads after the direct state surgery.
        e._avail = flat_available_b(e._layout, e._state)
        e._vps = _total_vp_b(e._state)
        end = jnp.asarray([int(ActionType.END_TURN)], jnp.int32)

        e.step(end, _batch_params([0]))
        assert bool(e.terminations[0, 0])
        assert float(e.rewards[0, 1]) == 1.0  # winner credited once
        assert float(e.rewards[0, 0]) == 0.0
        frozen = np.asarray(e._state.current_player)

        for _ in range(2):  # frozen: no mutation, no re-credit, still terminal
            e.step(end, _batch_params([0]))
            assert bool(e.terminations[0, 0])
            assert float(e.rewards[0, 1]) == 0.0
            assert float(e.rewards[0, 0]) == 0.0
            assert np.array_equal(np.asarray(e._state.current_player), frozen)

    def test_vp_delta_reward(self) -> None:
        from settlrl_engine.env.batched import _total_vp_b

        e = BatchedSettlrlEnv(batch_size=1, seed=9, reward="vp_delta")
        e._vps = _total_vp_b(e._state)
        vmask = np.asarray(e.available_indices(ActionType.SETUP_SETTLEMENT))
        idx = [_first_legal(vmask[0])]
        at = jnp.full((1,), int(ActionType.SETUP_SETTLEMENT), jnp.int32)
        e.step(at, _batch_params(idx))
        # Placing a settlement is +1 building VP for the acting player (player 0).
        assert float(e.rewards[0, 0]) == 1.0
        assert float(e.rewards[0, 1]) == 0.0

    def test_available_indices_rejects_non_index_action(self) -> None:
        e = BatchedSettlrlEnv(batch_size=1, seed=10)
        with pytest.raises(ValueError, match="no single primary-index domain"):
            e.available_indices(ActionType.MARITIME_TRADE)

    def test_invalid_reward_mode(self) -> None:
        with pytest.raises(ValueError, match="reward must be"):
            BatchedSettlrlEnv(batch_size=1, reward="bogus")

    def test_space_and_info_keys_match_typed_dicts(self) -> None:
        # mypy ties observe() to the Observation TypedDict; this pins the
        # space descriptor (and Infos) to the same key sets so they can't drift.
        e = BatchedSettlrlEnv(batch_size=1, seed=11)
        assert set(e.observation_space()) == set(Observation.__annotations__)
        assert set(e.infos) == set(Infos.__annotations__)


class TestDiscardOneCard:
    """DISCARD is one card of one resource per step: the discarder chooses freely
    among held resources, repeating until the owed count reaches zero."""

    def _discard_env(self) -> BatchedSettlrlEnv:
        """Lane 0 in the DISCARD phase: player 0 holds 4 sheep + 4 wheat, owes 4."""
        e = BatchedSettlrlEnv(batch_size=1, seed=12, auto_reset=False)
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
    def _discard(e: BatchedSettlrlEnv, resource: int) -> None:
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
        e = BatchedSettlrlEnv(batch_size=1, seed=0, n_players=2)
        assert e.possible_agents == ["player_0", "player_1"]
        assert e.num_agents == 2
        for bad in (1, N_PLAYERS + 1):
            with pytest.raises(ValueError, match="n_players"):
                BatchedSettlrlEnv(n_players=bad)

    def test_setup_snake_and_first_rotation(self) -> None:
        # 2-player snake: settlements (and their roads) go 0, 1, 1, 0; then ROLL
        # returns to player 0. Drive each step with the first legal vertex/edge.
        e = BatchedSettlrlEnv(batch_size=1, seed=2, n_players=2, auto_reset=False)
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
        e = BatchedSettlrlEnv(batch_size=2, seed=1, n_players=2)
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
        e = BatchedSettlrlEnv(batch_size=8, seed=3, auto_reset=False)
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
        e = BatchedSettlrlEnv(batch_size=1, seed=0, auto_reset=False)
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
        loop = BatchedSettlrlEnv(batch_size=4, seed=5, track_beliefs=track_beliefs)
        fused = BatchedSettlrlEnv(batch_size=4, seed=5, track_beliefs=track_beliefs)

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
        e = BatchedSettlrlEnv(batch_size=8, seed=0)
        cum = np.asarray(e.rollout(jax.random.key(1), 800))
        assert cum.shape == (8, e.n_players)
        assert (cum >= 0).all() and (cum == cum.astype(int)).all()


class TestBundleTradeThroughTheEnv:
    """Bundle proposals live outside the flat table; the env validates them
    with the trade core directly (see ``_env_step_core``)."""

    def test_bundle_proposal_steps_and_resolves(self) -> None:
        from settlrl_engine.board import give, to_main
        from settlrl_engine.mechanics.trade import pack_trade

        e = env.BatchedSettlrlEnv(batch_size=1, seed=0, auto_reset=False)
        layout, st = to_main(
            give(give(e.board, 0, [2, 0, 0, 0, 0]), 1, [0, 0, 1, 1, 0])
        )
        e._state = st
        e._avail = flat_available_b(layout, st)
        idx, target = pack_trade([2, 0, 0, 0, 0], [0, 0, 1, 1, 0], partner=1)
        e.step(
            jnp.array([int(ActionType.PROPOSE_TRADE)], jnp.int32),
            _batch_params([idx], [target]),
        )
        assert int(e._state.phase[0]) == GamePhase.TRADE_RESPONSE
        assert int(e.agent_selection[0]) == 1
        # The partner accepts: both multisets move.
        e.step(
            jnp.array([int(ActionType.ACCEPT_TRADE)], jnp.int32),
            _batch_params([0], [0]),
        )
        assert np.asarray(e._state.player_resources[0, 0]).tolist() == [0, 0, 1, 1, 0]
        assert np.asarray(e._state.player_resources[0, 1]).tolist() == [2, 0, 0, 0, 0]

    def test_illegal_bundle_is_rejected(self) -> None:
        from settlrl_engine.board import give, to_main
        from settlrl_engine.mechanics.trade import pack_trade

        e = env.BatchedSettlrlEnv(batch_size=1, seed=0, auto_reset=False)
        layout, st = to_main(give(e.board, 0, [1, 0, 0, 0, 0]))
        e._state = st
        e._avail = flat_available_b(layout, st)
        # Asking a card from an empty-handed partner can never complete.
        idx, target = pack_trade([1, 0, 0, 0, 0], [0, 0, 1, 0, 0], partner=1)
        e.step(
            jnp.array([int(ActionType.PROPOSE_TRADE)], jnp.int32),
            _batch_params([idx], [target]),
        )
        assert int(e._result[0]) == ActionResult.INVALID.value
        assert int(e._state.phase[0]) == GamePhase.MAIN


def _install(e: BatchedSettlrlEnv, board: tuple[Any, Any]) -> None:
    """Point ``e`` at a hand-built board and refresh the caches step() reads:
    the cached flat legality (the action gate) and the VP baseline the reward
    diffs against (``was_done`` / ``vp_delta``)."""
    from settlrl_engine.env.batched import _agent_selection_b, _total_vp_b

    e._layout, e._state = board
    e._avail = flat_available_b(*board)
    e._vps = _total_vp_b(board[1])
    e._agent_sel = _agent_selection_b(board[1])


def _legal_knight(e: BatchedSettlrlEnv) -> tuple[int, int]:
    """A legal ``(tile, victim)`` PlayKnight for lane 0, decoded off the table
    (the steal target depends on the board, so read it from the legality sweep
    rather than hard-coding an encoding)."""
    from settlrl_engine.env import N_FLAT, flat_to_action

    types, params = flat_to_action(jnp.arange(N_FLAT, dtype=jnp.int32))
    legal = np.asarray(e.flat_mask()[0])
    rows = np.flatnonzero((np.asarray(types) == int(ActionType.PLAY_KNIGHT)) & legal)
    assert rows.size, "no legal PlayKnight on the hand-built board"
    row = int(rows[0])
    return int(params.idx[row]), int(params.target[row])


class TestExpectedReward:
    """Hand-designed boards with a known reward for the next move, driven end to
    end through ``step()``: sparse (+1 to the winner on the terminal step) and
    vp_delta (each player's total-VP change this step), plus the freeze that
    keeps a terminal lane from being re-credited."""

    def test_sparse_win_by_city_build(self) -> None:
        from settlrl_engine.board import give, place_city, place_settlement, to_main

        # 2 cities (4 VP) + 5 settlements (5 VP) = 9; upgrading a settlement to a
        # city reaches 10 on player 0's own turn -> player 0 wins.
        b = make_board(seed=0, n_players=2)
        for v in (0, 2):
            b = place_city(b, 0, v)
        for v in (4, 6, 8, 10, 12):
            b = place_settlement(b, 0, v)
        b = to_main(give(b, 0, [0, 5, 0, 0, 5]))  # ample city resources
        # auto_reset=False keeps the terminal board for inspection.
        e = BatchedSettlrlEnv(
            batch_size=1, seed=0, n_players=2, reward="sparse", auto_reset=False
        )
        _install(e, b)

        e.step(jnp.array([int(ActionType.BUILD_CITY)], jnp.int32), _batch_params([4]))
        assert int(e.infos["result"][0]) == ActionResult.GAME_COMPLETE.value
        assert int(e._vps[0, 0]) == 10  # 3 cities + 4 settlements, on own turn
        assert bool(e.terminations[0, 0])
        assert [float(x) for x in e.rewards[0]] == [1.0, 0.0]

    def test_sparse_win_by_largest_army(self) -> None:
        from settlrl_engine.board import give_dev_card, place_city, to_main
        from settlrl_engine.board.dev_cards import DevCard

        # 4 cities (8 VP); playing a 3rd knight takes Largest Army (+2) -> 10 VP.
        b = make_board(seed=1, n_players=2)
        for v in (0, 2, 4, 6):
            b = place_city(b, 0, v)
        b = give_dev_card(b, 0, DevCard.KNIGHT, 1)
        layout, st = to_main(b)
        st = st._replace(knights_played=st.knights_played.at[0, 0].set(2))
        e = BatchedSettlrlEnv(
            batch_size=1, seed=1, n_players=2, reward="sparse", auto_reset=False
        )
        _install(e, (layout, st))

        tile, victim = _legal_knight(e)
        e.step(
            jnp.array([int(ActionType.PLAY_KNIGHT)], jnp.int32),
            _batch_params([tile], [victim]),
        )
        assert int(e._state.largest_army_owner[0]) == 0  # award taken
        assert int(e._vps[0, 0]) == 10  # 8 building + 2 for the army
        assert bool(e.terminations[0, 0])
        assert [float(x) for x in e.rewards[0]] == [1.0, 0.0]

    def test_sparse_non_winning_build_pays_nothing(self) -> None:
        from settlrl_engine.board import give, place_settlement, to_main

        # Upgrading a lone settlement reaches only 2 VP: nobody wins, no reward.
        b = place_settlement(make_board(seed=2, n_players=2), 0, 0)
        b = to_main(give(b, 0, [0, 5, 0, 0, 5]))
        e = BatchedSettlrlEnv(batch_size=1, seed=2, n_players=2, reward="sparse")
        _install(e, b)

        e.step(jnp.array([int(ActionType.BUILD_CITY)], jnp.int32), _batch_params([0]))
        assert int(e.infos["result"][0]) == ActionResult.SUCCESS.value
        assert not bool(e.terminations[0, 0])
        assert [float(x) for x in e.rewards[0]] == [0.0, 0.0]

    def test_vp_delta_credits_award_gain(self) -> None:
        from settlrl_engine.board import give_dev_card, to_main
        from settlrl_engine.board.dev_cards import DevCard

        # No buildings; a 3rd knight grants Largest Army -> +2 VP this step.
        b = give_dev_card(make_board(seed=3, n_players=2), 0, DevCard.KNIGHT, 1)
        layout, st = to_main(b)
        st = st._replace(knights_played=st.knights_played.at[0, 0].set(2))
        e = BatchedSettlrlEnv(batch_size=1, seed=3, n_players=2, reward="vp_delta")
        _install(e, (layout, st))

        tile, victim = _legal_knight(e)
        e.step(
            jnp.array([int(ActionType.PLAY_KNIGHT)], jnp.int32),
            _batch_params([tile], [victim]),
        )
        assert int(e._state.largest_army_owner[0]) == 0
        assert not bool(e.terminations[0, 0])
        assert [float(x) for x in e.rewards[0]] == [2.0, 0.0]

    def test_vp_delta_zero_when_no_vp_change(self) -> None:
        from settlrl_engine.board import give, to_main

        # A maritime trade moves cards but no VP: zero reward for everyone.
        b = to_main(give(make_board(seed=4, n_players=2), 0, [4, 0, 0, 0, 0]))
        e = BatchedSettlrlEnv(batch_size=1, seed=4, n_players=2, reward="vp_delta")
        _install(e, b)

        # 4 sheep -> 1 wheat at the bank's 4:1 rate (no port owned).
        e.step(
            jnp.array([int(ActionType.MARITIME_TRADE)], jnp.int32),
            _batch_params([0], [1]),
        )
        assert int(e.infos["result"][0]) == ActionResult.SUCCESS.value
        assert [float(x) for x in e.rewards[0]] == [0.0, 0.0]

    def test_terminal_lane_credits_win_once_then_freezes(self) -> None:
        from settlrl_engine.board import to_main

        # Player 1 sits at 10 VP off-turn; player 0 (current, MAIN) ends the
        # turn, so player 1 claims the win at their turn start (rulebook p.5).
        # auto_reset=False: the win is credited once, then the lane freezes and
        # is never re-credited (the regression this guards against).
        layout, st = to_main(make_board(seed=8, n_players=2))
        st = st._replace(victory_points=st.victory_points.at[0, 1].set(10))
        e = BatchedSettlrlEnv(
            batch_size=1, seed=8, n_players=2, reward="sparse", auto_reset=False
        )
        _install(e, (layout, st))
        end = jnp.array([int(ActionType.END_TURN)], jnp.int32)

        e.step(end, _batch_params([0]))
        assert bool(e.terminations[0, 0])
        assert [float(x) for x in e.rewards[0]] == [0.0, 1.0]  # credited once
        frozen = np.asarray(e._state.current_player)

        for _ in range(2):  # frozen: no mutation, no re-credit, still terminal
            e.step(end, _batch_params([0]))
            assert bool(e.terminations[0, 0])
            assert [float(x) for x in e.rewards[0]] == [0.0, 0.0]
            assert np.array_equal(np.asarray(e._state.current_player), frozen)
