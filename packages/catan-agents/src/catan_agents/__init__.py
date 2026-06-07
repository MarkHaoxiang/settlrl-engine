"""Catan-playing agents: pure-JAX policies over catan-engine's flat action space."""

from catan_engine.board.resources import N_PLAYERS

from catan_agents.shared import (
    AgentSpec,
    BeliefPolicy,
    EvalResult,
    FlatAction,
    FlatMask,
    Policy,
    Value,
    ValueFunction,
    evaluate,
    greedy_policy,
    heuristic_value,
    make_heuristic,
    random_policy,
    sample_world,
)
from catan_agents.search import (
    lookahead_policy,
    make_greedy,
    make_mcts,
    mcts_policy,
)

_ANY_COUNT = frozenset(range(2, N_PLAYERS + 1))

POLICIES: dict[str, AgentSpec] = {
    "random": AgentSpec(random_policy, "observation", _ANY_COUNT),
    "greedy": AgentSpec(greedy_policy, "observation", _ANY_COUNT),
    "lookahead": AgentSpec(lookahead_policy, "belief", _ANY_COUNT),
    "mcts": AgentSpec(mcts_policy, "belief", _ANY_COUNT),
}
"""Every shipped agent by name."""

__all__ = [
    "AgentSpec",
    "BeliefPolicy",
    "EvalResult",
    "FlatAction",
    "FlatMask",
    "POLICIES",
    "Policy",
    "Value",
    "ValueFunction",
    "evaluate",
    "greedy_policy",
    "heuristic_value",
    "lookahead_policy",
    "make_greedy",
    "make_heuristic",
    "make_mcts",
    "mcts_policy",
    "random_policy",
    "sample_world",
]
