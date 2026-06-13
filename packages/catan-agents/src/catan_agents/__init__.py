"""Catan-playing agents over catan-engine's flat action space: pure-JAX
policies plus stateful plain-Python planners."""

from catan_engine.board.resources import N_PLAYERS

from catan_agents.baselines import random_policy
from catan_agents.evaluate import EvalResult, evaluate
from catan_agents.greedy import greedy_policy
from catan_agents.planner import make_planner
from catan_agents.policy import (
    AgentSpec,
    BeliefPolicy,
    BeliefSpec,
    FlatAction,
    FlatMask,
    GameAgent,
    ObservationSpec,
    Policy,
    PolicyPrior,
    StatefulPolicy,
    StatefulSpec,
)
from catan_agents.sample import sample_world
from catan_agents.search import (
    lookahead_policy,
    make_greedy,
    make_mcts,
    mcts_policy,
)
from catan_agents.value import (
    TUNED_WEIGHTS,
    Value,
    ValueFunction,
    heuristic_value,
    make_heuristic,
    make_linear,
    tuned_value,
)

_ANY_COUNT = frozenset(range(2, N_PLAYERS + 1))

POLICIES: dict[str, ObservationSpec | BeliefSpec | StatefulSpec] = {
    "random": ObservationSpec(lambda: random_policy, _ANY_COUNT),
    "greedy": ObservationSpec(lambda: greedy_policy, _ANY_COUNT),
    "planner": StatefulSpec(make_planner, _ANY_COUNT),
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
    "TUNED_WEIGHTS",
    "AgentSpec",
    "BeliefPolicy",
    "BeliefSpec",
    "EvalResult",
    "FlatAction",
    "FlatMask",
    "GameAgent",
    "ObservationSpec",
    "Policy",
    "PolicyPrior",
    "StatefulPolicy",
    "StatefulSpec",
    "Value",
    "ValueFunction",
    "evaluate",
    "greedy_policy",
    "heuristic_value",
    "lookahead_policy",
    "make_greedy",
    "make_heuristic",
    "make_linear",
    "make_mcts",
    "make_planner",
    "mcts_policy",
    "random_policy",
    "sample_world",
    "tuned_value",
]
