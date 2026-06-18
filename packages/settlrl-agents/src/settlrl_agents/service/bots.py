"""The bundled engine-backed bots, each a :class:`Bot` over a settlrl-agents policy.

The framework replays a game into a :class:`~settlrl_game.session.GameSession`; an
``EngineBot`` bridges that session to a single-game engine env so the JAX policy can
observe and choose, then translates the chosen engine action back to a structured
move. Only the non-stateful, fully-observable-or-belief policies in ``POLICIES`` are
seatable this way (one move is a pure function of the position).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from settlrl_engine.env import Observation
from settlrl_game.actions import move_for_flat
from settlrl_game.botproto import MoveModel

from settlrl_agents import POLICIES, BeliefSpec, ObservationSpec
from settlrl_agents.policy import BeliefPolicy, Policy
from settlrl_agents.service.bridge import engine_env, game_flat
from settlrl_agents.service.sdk import Bot, GameView

__all__ = ["BUNDLED", "EngineBot", "make_bot"]

# The kinds the bundled service can host (the seatable, non-stateful policies).
BUNDLED = ["random", "greedy", "lookahead", "mcts"]

_TITLES = {
    "random": "Random",
    "greedy": "Greedy",
    "lookahead": "Lookahead",
    "mcts": "MCTS",
}
# Short user-facing blurbs for the new-game bot picker.
_DESCRIPTIONS = {
    "random": "Plays a random legal move — the gentle baseline.",
    "greedy": "Grabs the best move it can see right now; fast, no planning ahead.",
    "lookahead": "Looks a move ahead and weighs trades — a solid intermediate.",
    "mcts": "Monte-Carlo tree search: the strongest, but slower.",
}

# Configured policies, jitted once per kind across games.
_ACTS: dict[str, Policy | BeliefPolicy] = {}


def _policy(kind: str) -> Policy | BeliefPolicy:
    if kind not in _ACTS:
        spec = POLICIES[kind]
        if not isinstance(spec, ObservationSpec | BeliefSpec):
            raise ValueError(f"bot kind {kind!r} is not seatable (stateful family)")
        _ACTS[kind] = jax.jit(spec.policy)
    return _ACTS[kind]


def _engine_move(kind: str, view: GameView) -> MoveModel:
    """The move ``kind`` plays in ``view``: reason on an engine env bridged from
    the reference game, then translate the chosen action back to a structured move.
    """
    session, seat = view.session, view.seat
    benv = engine_env(session.game, session.belief_state)
    # Reproducible per position, independent of the bridged env's own key.
    key = jax.random.fold_in(jax.random.key(0), len(session.moves_flat()))
    act = _policy(kind)
    mask = benv.flat_mask()[0]
    if isinstance(POLICIES[kind], ObservationSpec):
        obs = cast(Observation, jax.tree.map(lambda x: x[0], benv.observe(seat)))
        engine_flat = int(cast(Policy, act)(key, obs, mask))
    else:
        layout = jax.tree.map(lambda x: x[0], benv.board[0])
        belief = jax.tree.map(lambda x: x[0], benv.belief_view(seat))
        engine_flat = int(
            cast(BeliefPolicy, act)(key, layout, belief, jnp.int32(seat), mask)
        )
    return move_for_flat(game_flat(engine_flat))


class EngineBot(Bot):
    """A bundled bot wrapping the settlrl-agents policy named ``kind``."""

    def __init__(self, kind: str) -> None:
        spec = POLICIES[kind]
        if not isinstance(spec, ObservationSpec | BeliefSpec):
            raise ValueError(f"bot kind {kind!r} is not seatable (stateful family)")
        self.name = kind
        self.title = _TITLES.get(kind, kind.title())
        self.description = _DESCRIPTIONS.get(kind, "")
        self.counts = sorted(spec.n_players)

    def act(self, view: GameView) -> MoveModel:
        return _engine_move(self.name, view)


def make_bot(kind: str) -> Bot:
    """The bundled bot for ``kind`` (one of :data:`BUNDLED`)."""
    return EngineBot(kind)
