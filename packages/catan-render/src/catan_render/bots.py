"""Bot seats: the catan-agents registry adapted to the renderer's single game.

``POLICIES`` is catan-agents' registry (name -> ``AgentSpec``); a spec declares
which protocol its policy speaks (observation- or state-based) and the seat
counts it supports (the two-player lookahead / mcts agents are only offered in
two-player games). :func:`bot_act` hides the dispatch: it slices the single
game out of the renderer's batch-of-one env and calls the policy through the
right protocol.
"""

from typing import cast

import jax
import jax.numpy as jnp
from catan_agents import POLICIES, AgentSpec
from catan_agents.shared.policy import Policy, StatePolicy
from catan_engine.env import BatchedCatanEnv

__all__ = ["POLICIES", "AgentSpec", "bot_act", "supported_counts"]

# Jitted policies, compiled once per kind across sessions.
_BOT_ACTS: dict[str, Policy | StatePolicy] = {}


def supported_counts() -> dict[str, list[int]]:
    """Seat counts each bot kind supports, by name."""
    return {name: sorted(spec.n_players) for name, spec in POLICIES.items()}


def bot_act(kind: str, key: jax.Array, benv: BatchedCatanEnv, seat: int) -> int:
    """One move for ``seat`` from bot ``kind`` on the (batch-of-one) env."""
    spec = POLICIES[kind]
    if kind not in _BOT_ACTS:
        _BOT_ACTS[kind] = jax.jit(spec.policy)
    mask = benv.flat_mask()[0]
    if spec.observes == "observation":
        obs = {k: v[0] for k, v in benv.observe(seat).items()}
        return int(cast(Policy, _BOT_ACTS[kind])(key, obs, mask))
    layout, state = jax.tree.map(lambda x: x[0], benv.board)
    return int(
        cast(StatePolicy, _BOT_ACTS[kind])(key, layout, state, jnp.int32(seat), mask)
    )
