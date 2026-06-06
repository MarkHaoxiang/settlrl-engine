"""Tests for the PettingZoo-AEC wrapper (catan_engine.env.aec).

Includes the PettingZoo-provided ``api_test`` compliance check.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from pettingzoo.test import api_test

from catan_engine.env.aec import CatanAECEnv, env
from catan_engine.mechanics.action import _ATYPE, _DISCARD_FLAT, ActionType
from catan_engine.board.state import GamePhase


class TestCatanAEC:
    def test_pettingzoo_api(self) -> None:
        # The official compliance test: drives random (masked) play and checks
        # the AEC invariants (spaces, masks, reward accumulation, dead-stepping).
        api_test(env(seed=0), num_cycles=40)

    def test_reset_and_agents(self) -> None:
        e = CatanAECEnv(seed=1)
        assert e.agents == [f"player_{i}" for i in range(4)]
        assert e.agent_selection == "player_0"  # setup starts with player 0
        obs, reward, term, trunc, info = e.last()
        assert set(obs) == {"observation", "action_mask"}
        assert obs["action_mask"].dtype == np.int8
        assert not term and not trunc
        assert reward == 0.0

    def test_mask_matches_phase(self) -> None:
        e = CatanAECEnv(seed=2)
        obs, *_ = e.last()
        mask = obs["action_mask"]
        # Only opening settlement placements are legal at the very start.
        legal_types = {int(a) for a, m in zip(_ATYPE, mask) if m}
        assert legal_types == {int(ActionType.SETUP_SETTLEMENT)}

    def test_observation_in_space(self) -> None:
        e = CatanAECEnv(seed=3)
        for agent in e.agents:
            assert e.observation_space(agent).contains(e.observe(agent))

    def test_render_ansi(self) -> None:
        e = CatanAECEnv(seed=4, render_mode="ansi")
        out = e.render()
        assert isinstance(out, str) and "lane 0" in out


class TestDiscardChoice:
    """The discard action takes its per-resource amounts as a vector:
    ``step((flat, resources))`` chooses exactly what to give up; a bare index
    falls back to the canonical greedy fill."""

    @staticmethod
    def _force_discard(e: CatanAECEnv) -> None:
        """Put the game in DISCARD: player 0 holds 4 sheep + 4 wheat, owes 4."""
        st = e._env._state
        e._env._state = st._replace(
            phase=st.phase.at[0].set(int(GamePhase.DISCARD)),
            player_resources=st.player_resources.at[0, 0].set(
                jnp.asarray([4, 4, 0, 0, 0], dtype=jnp.uint8)
            ),
            pending_discard=st.pending_discard.at[0, 0].set(4),
        )
        e.agent_selection = e._acting_agent()

    def test_vector_choice(self) -> None:
        e = CatanAECEnv(seed=5)
        self._force_discard(e)
        e.step((int(_DISCARD_FLAT), [0, 4, 0, 0, 0]))  # give up the wheat
        hand = np.asarray(e._env._state.player_resources[0, 0])
        assert list(hand) == [4, 0, 0, 0, 0]
        assert int(e._env._state.phase[0]) == GamePhase.MOVE_ROBBER

    def test_bare_index_is_canonical(self) -> None:
        e = CatanAECEnv(seed=6)
        self._force_discard(e)
        e.step(int(_DISCARD_FLAT))  # greedy in resource order: takes the sheep
        hand = np.asarray(e._env._state.player_resources[0, 0])
        assert list(hand) == [0, 4, 0, 0, 0]

    def test_illegal_vector_no_ops(self) -> None:
        # Well-formed but illegal (wrong total): rejected by the engine gate,
        # like any other illegal action.
        e = CatanAECEnv(seed=7)
        self._force_discard(e)
        e.step((int(_DISCARD_FLAT), [1, 1, 0, 0, 0]))
        hand = np.asarray(e._env._state.player_resources[0, 0])
        assert list(hand) == [4, 4, 0, 0, 0]
        assert int(e._env._state.phase[0]) == GamePhase.DISCARD

    def test_malformed_vector_raises(self) -> None:
        e = CatanAECEnv(seed=8)
        self._force_discard(e)
        with pytest.raises(ValueError, match="nonnegative"):
            e.step((int(_DISCARD_FLAT), [1, 2, 3]))
