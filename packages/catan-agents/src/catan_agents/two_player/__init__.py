"""Two-player agents: full-state seats (see ``shared.policy.StatePolicy``).

With two players the board state is publicly inferable, so these agents may
consume it directly as a world model for lookahead and search.
"""

from catan_agents.two_player.greedy import lookahead_policy, make_greedy
from catan_agents.two_player.mcts import make_mcts, mcts_policy

__all__ = [
    "lookahead_policy",
    "make_greedy",
    "make_mcts",
    "mcts_policy",
]
