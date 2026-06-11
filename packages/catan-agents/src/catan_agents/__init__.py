"""Catan-playing agents: pure-JAX policies over catan-engine's flat action space."""

from catan_engine.board.resources import N_PLAYERS

from catan_agents.search import (
    lookahead_policy,
    make_greedy,
    make_mcts,
    mcts_policy,
)
from catan_agents.shared import (
    AgentSpec,
    BeliefPolicy,
    BeliefSpec,
    EvalResult,
    FlatAction,
    FlatMask,
    ObservationSpec,
    Policy,
    PolicyPrior,
    Value,
    ValueFunction,
    evaluate,
    greedy_policy,
    heuristic_value,
    make_heuristic,
    random_policy,
    sample_world,
)

_ANY_COUNT = frozenset(range(2, N_PLAYERS + 1))

POLICIES: dict[str, ObservationSpec | BeliefSpec] = {
    "random": ObservationSpec(lambda: random_policy, _ANY_COUNT),
    "greedy": ObservationSpec(lambda: greedy_policy, _ANY_COUNT),
    "lookahead": BeliefSpec(
        make_greedy, _ANY_COUNT, defaults={"value": heuristic_value}
    ),
    "mcts": BeliefSpec(
        make_mcts,
        _ANY_COUNT,
        defaults={"value": heuristic_value},
        # One world/future and a small Gumbel budget: a cheap member of the
        # same family for the protocol tests.
        for_testing={
            "num_worlds": 1,
            "num_futures": 1,
            "num_simulations": 8,
            "max_num_considered_actions": 8,
        },
    ),
}
"""Every shipped agent by name (the family at its ``defaults``)."""

__all__ = [
    "POLICIES",
    "AgentSpec",
    "BeliefPolicy",
    "BeliefSpec",
    "EvalResult",
    "FlatAction",
    "FlatMask",
    "ObservationSpec",
    "Policy",
    "PolicyPrior",
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
