"""Bot seats: the catan-agents registry adapted to the renderer's single game.

``POLICIES`` is catan-agents' registry (name -> ``AgentSpec``); a spec's class
declares which protocol its policy speaks (``ObservationSpec`` /
``BeliefSpec``) and the seat counts it supports. :func:`bot_act` hides the
dispatch: it slices the single game out of the renderer's batch-of-one env and
calls the policy through the right protocol (belief seats read the env's
honest ``belief_view``, so the env must be built with ``track_beliefs=True``
when any are seated).
"""

from typing import cast

import jax
import jax.numpy as jnp
from catan_agents import POLICIES, AgentSpec, BeliefSpec, ObservationSpec
from catan_agents.shared.policy import BeliefPolicy, Policy
from catan_engine.board.state import KeyScalar
from catan_engine.env import BatchedCatanEnv, Observation

__all__ = ["POLICIES", "AgentSpec", "BeliefSpec", "bot_act", "supported_counts"]

# Jitted policies, compiled once per kind across sessions.
_OBS_ACTS: dict[str, Policy] = {}
_BELIEF_ACTS: dict[str, BeliefPolicy] = {}


def supported_counts() -> dict[str, list[int]]:
    """Seat counts each bot kind supports, by name."""
    return {name: sorted(spec.n_players) for name, spec in POLICIES.items()}


def bot_act(kind: str, key: KeyScalar, benv: BatchedCatanEnv, seat: int) -> int:
    """One move for ``seat`` from bot ``kind`` on the (batch-of-one) env."""
    spec = POLICIES[kind]
    mask = benv.flat_mask()[0]
    if isinstance(spec, ObservationSpec):
        if kind not in _OBS_ACTS:
            _OBS_ACTS[kind] = jax.jit(spec.policy)
        obs = cast(Observation, jax.tree.map(lambda x: x[0], benv.observe(seat)))
        return int(_OBS_ACTS[kind](key, obs, mask))
    if kind not in _BELIEF_ACTS:
        _BELIEF_ACTS[kind] = jax.jit(spec.policy)
    layout = jax.tree.map(lambda x: x[0], benv.board[0])
    view = jax.tree.map(lambda x: x[0], benv.belief_view(seat))
    return int(_BELIEF_ACTS[kind](key, layout, view, jnp.int32(seat), mask))
