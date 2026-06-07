"""Catan-playing agents: pure-JAX policies over catan-engine's flat action space."""

from catan_engine.board.resources import N_PLAYERS

from catan_agents.shared import (
    AgentSpec,
    EvalResult,
    FlatAction,
    FlatMask,
    Policy,
    StatePolicy,
    Value,
    ValueFunction,
    evaluate,
    greedy_policy,
    heuristic_value,
    random_policy,
)
from catan_agents.two_player import (
    lookahead_policy,
    make_greedy,
    make_mcts,
    mcts_policy,
)

_ANY_COUNT = frozenset(range(2, N_PLAYERS + 1))
_TWO_PLAYER = frozenset((2,))

POLICIES: dict[str, AgentSpec] = {
    "random": AgentSpec(random_policy, "observation", _ANY_COUNT),
    "greedy": AgentSpec(greedy_policy, "observation", _ANY_COUNT),
    "lookahead": AgentSpec(lookahead_policy, "state", _TWO_PLAYER),
    "mcts": AgentSpec(mcts_policy, "state", _TWO_PLAYER),
}
"""Every shipped agent by name."""

__all__ = [
    "AgentSpec",
    "EvalResult",
    "FlatAction",
    "FlatMask",
    "POLICIES",
    "Policy",
    "StatePolicy",
    "Value",
    "ValueFunction",
    "evaluate",
    "greedy_policy",
    "heuristic_value",
    "lookahead_policy",
    "make_greedy",
    "make_mcts",
    "mcts_policy",
    "random_policy",
]
