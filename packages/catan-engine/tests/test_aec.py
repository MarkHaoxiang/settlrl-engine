"""Tests for the PettingZoo-AEC wrapper (catan_engine.aec).

Includes the PettingZoo-provided ``api_test`` compliance check.
"""

import numpy as np
from pettingzoo.test import api_test

from catan_engine.aec import _ATYPE, CatanAECEnv, env
from catan_engine.action import ActionType


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
