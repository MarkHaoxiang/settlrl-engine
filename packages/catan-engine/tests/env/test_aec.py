"""Tests for the PettingZoo-AEC wrapper (catan_engine.env.aec).

Includes the PettingZoo-provided ``api_test`` compliance check.
"""

import jax.numpy as jnp
import numpy as np
from pettingzoo.test import api_test

from catan_engine.env.aec import CatanAECEnv, env
from catan_engine.mechanics.action import _ATYPE, _IDX, ActionType, _flat_available_b
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


# Flat index of "discard one card of resource r", from the static table.
_DISCARD_ROWS = {
    int(_IDX[f]): int(f)
    for f in np.where(_ATYPE == int(ActionType.DISCARD))[0]
}


class TestDiscardOneCard:
    """Discarding is one flat action per resource, one card per step; the mask
    offers only the resources the discarder still holds."""

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
        # The action mask and step gate read the cached flat legality; refresh
        # it after the direct state surgery above.
        e._env._avail = _flat_available_b(e._env._layout, e._env._state)
        e.agent_selection = e._acting_agent()

    def test_mask_offers_held_resources_only(self) -> None:
        e = CatanAECEnv(seed=5)
        self._force_discard(e)
        obs, *_ = e.last()
        mask = obs["action_mask"]
        legal = set(np.where(mask)[0])
        assert legal == {_DISCARD_ROWS[0], _DISCARD_ROWS[1]}  # sheep + wheat only

    def test_chosen_sequence_applies(self) -> None:
        # Choose 1 sheep then 3 wheat (greedy order would strip all the sheep).
        e = CatanAECEnv(seed=6)
        self._force_discard(e)
        for resource in (0, 1, 1, 1):
            assert e.agent_selection == "player_0"
            e.step(_DISCARD_ROWS[resource])
        hand = np.asarray(e._env._state.player_resources[0, 0])
        assert list(hand) == [3, 1, 0, 0, 0]
        assert int(e._env._state.phase[0]) == GamePhase.MOVE_ROBBER

    def test_unheld_resource_no_ops(self) -> None:
        # Masked out and rejected by the engine gate, like any illegal action.
        e = CatanAECEnv(seed=7)
        self._force_discard(e)
        e.step(_DISCARD_ROWS[2])  # wood: not held
        hand = np.asarray(e._env._state.player_resources[0, 0])
        assert list(hand) == [4, 4, 0, 0, 0]
        assert int(e._env._state.phase[0]) == GamePhase.DISCARD
