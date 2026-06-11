"""Single-game PettingZoo-AEC wrapper around the batched engine.

``CatanAECEnv`` drives one game via ``BatchedCatanEnv(batch_size=1,
auto_reset=False)`` (real terminal states); for vectorised rollouts use
``BatchedCatanEnv`` directly. The action space is one flat ``Discrete`` over
every concrete move; legality is exposed PettingZoo-style as
``observation["action_mask"]``, so ``action_space(agent).sample(mask)`` only
picks legal moves. Observations wrap the batched env's partial view as
``{"observation": {...}, "action_mask": ...}``. Requires the optional ``rl``
extra (``pettingzoo``, ``gymnasium``).
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import gymnasium.spaces as spaces
import jax.numpy as jnp
import numpy as np
from pettingzoo.utils.env import AECEnv

from catan_engine.board.resources import N_PLAYERS
from catan_engine.env.batched import N_FLAT, BatchedCatanEnv, flat_to_action

__all__ = ["CatanAECEnv", "env"]

# The flat action table (index -> (ActionType, ActionParams)) lives in
# ``mechanics/flat.py`` -- shared with ``BatchedCatanEnv.random_actions``.


class CatanAECEnv(AECEnv):  # type: ignore[misc]  # pettingzoo is untyped (Any base)
    """A single Catan game behind the PettingZoo AEC API.

    Args:
        seed: PRNG seed for the board and dice/steal randomness.
        reward: ``"sparse"`` or ``"vp_delta"`` (see ``BatchedCatanEnv``).
        render_mode: ``None``, ``"ansi"`` (returns a status string) or
            ``"human"`` (prints it).
        number_placement: ``"random"`` or ``"spiral"`` number-token placement
            (see ``BatchedCatanEnv``).
        n_players: players seated (2..4, default 4); sets the agent list.
        track_beliefs: maintain per-observer card counting (see
            ``BatchedCatanEnv``).
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "ansi"],
        "name": "catan_aec_v0",
    }

    def __init__(
        self,
        seed: int = 0,
        reward: str = "sparse",
        render_mode: str | None = None,
        number_placement: Literal["random", "spiral"] = "random",
        n_players: int = N_PLAYERS,
        track_beliefs: bool = False,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self._env = BatchedCatanEnv(
            batch_size=1,
            seed=seed,
            reward=reward,
            auto_reset=False,
            number_placement=number_placement,
            n_players=n_players,
            track_beliefs=track_beliefs,
        )
        self.possible_agents = list(self._env.possible_agents)
        self._index = {a: i for i, a in enumerate(self.possible_agents)}

        # Spaces are built once and returned by identity (PettingZoo requires the
        # same object each call so space seeding works).
        self.action_spaces: dict[str, spaces.Space[Any]] = {
            a: spaces.Discrete(N_FLAT) for a in self.possible_agents
        }
        obs_space = self._build_observation_space()
        self.observation_spaces: dict[str, spaces.Space[Any]] = dict.fromkeys(
            self.possible_agents, obs_space
        )

        self.reset(seed)

    # -- spaces -----------------------------------------------------------

    def _build_observation_space(self) -> spaces.Dict:
        sample = self._env.observe(0)
        inner: dict[str, spaces.Space[Any]] = {}
        for key, value in sample.items():
            arr = np.asarray(value)[0]  # strip the batch axis
            inner[key] = spaces.Box(
                low=0,
                high=int(np.iinfo(arr.dtype).max),
                shape=arr.shape,
                dtype=arr.dtype,
            )
        return spaces.Dict(
            {
                "observation": spaces.Dict(inner),
                "action_mask": spaces.Box(0, 1, (N_FLAT,), dtype=np.int8),
            }
        )

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    # -- AEC lifecycle ----------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._env.reset(self._seed)
        self.agents = list(self.possible_agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos: dict[str, dict[str, Any]] = {a: {} for a in self.agents}
        self.agent_selection = self._acting_agent()

    def step(self, action: int | None) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self._apply(int(action))  # type: ignore[arg-type]

        reward = np.asarray(self._env.rewards[0])  # (n_players,)
        done = bool(np.asarray(self._env.terminations[0, 0]))
        self.rewards = {a: float(reward[i]) for i, a in enumerate(self.possible_agents)}
        if done:
            self.terminations = dict.fromkeys(self.agents, True)
        self.agent_selection = self._acting_agent()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: str) -> dict[str, Any]:
        obs = self._env.observe(self._index[agent])
        inner = {key: np.asarray(value)[0] for key, value in obs.items()}
        return {"observation": inner, "action_mask": self._action_mask()}

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        text = self._env.render(0)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        self._env.close()

    # -- internals --------------------------------------------------------

    def _acting_agent(self) -> str:
        """The acting player (current player, the discarder in DISCARD, or the
        trade partner in TRADE_RESPONSE)."""
        return self.possible_agents[int(self._env.agent_selection[0])]

    def _action_mask(self) -> np.ndarray:
        """Binary legality vector over the flat action set for the acting player."""
        return np.asarray(self._env.flat_mask()[0]).astype(np.int8)

    def _apply(self, flat: int) -> None:
        at, params = flat_to_action(jnp.asarray([flat], dtype=jnp.int32))
        self._env.step(at, params)


def env(
    seed: int = 0,
    reward: str = "sparse",
    render_mode: str | None = None,
    number_placement: Literal["random", "spiral"] = "random",
    n_players: int = N_PLAYERS,
    track_beliefs: bool = False,
) -> CatanAECEnv:
    """PettingZoo-style constructor returning a :class:`CatanAECEnv`."""
    return CatanAECEnv(
        seed=seed,
        reward=reward,
        render_mode=render_mode,
        number_placement=number_placement,
        n_players=n_players,
        track_beliefs=track_beliefs,
    )
