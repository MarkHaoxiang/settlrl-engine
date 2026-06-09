"""Model-based agents: value-driven search over sampled worlds.

Both agents are :class:`~catan_agents.shared.policy.BeliefPolicy` seats at any
player count: they receive a :class:`~catan_engine.belief.BeliefView`,
determinize it with one ``sample_world`` draw, and search in the sample (PIMC
— the simulated opponents share the sampled world).
"""

from catan_agents.search.greedy import lookahead_policy, make_greedy
from catan_agents.search.mcts import make_mcts, mcts_policy

__all__ = [
    "lookahead_policy",
    "make_greedy",
    "make_mcts",
    "mcts_policy",
]
