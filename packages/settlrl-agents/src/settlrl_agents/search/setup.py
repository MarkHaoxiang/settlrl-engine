"""A specialized search for the **setup phase** (the initial settlement/road
placements) -- a depth-limited, beam-pruned *probabilistic expectimax*.

Setup is a short, fully-observable, high-leverage sub-game: there are no cards
yet, so nothing is hidden, and the two opening settlements decide the whole
economy. It is also structurally distinct from the main loop, so the learned net
does not handle it -- this fixed policy does.

The search backs up a **per-player value vector** (the heuristic value from each
seat) so every node knows the mover's own value. At the searcher's nodes it
maximizes (we assume we play the opener optimally); at an opponent's nodes it
takes a **Boltzmann expectation** over that opponent's self-interested moves --
``softmax(opponent_value / temperature)``. ``temperature`` is the chance the
opponent plays a non-optimal placement: ``-> 0`` is an optimal (minimax)
opponent, large is uniform-random. Modeling opponents as imperfect matters most
with **more than two players**, where assuming three optimal openers is far too
pessimistic. A beam (top-``beam`` placements by their one-ply value) keeps the
``beam**depth`` tree tractable under ``vmap`` for batched self-play.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, GamePhase, IntScalar, KeyScalar
from settlrl_engine.mechanics.action import ActionType, action_available, apply_action
from settlrl_engine.mechanics.common import agent_selection_single

from settlrl_agents.internal.rows import ROW_PARAMS, ROW_TYPE
from settlrl_agents.policy import BeliefPolicy, FlatAction, FlatMask
from settlrl_agents.sample import sample_world
from settlrl_agents.value import ValueFunction

__all__ = ["make_setup_search"]

# The flat rows that are setup placements (initial settlement or road).
_SETUP_IDX = jnp.asarray(
    np.flatnonzero(
        (np.asarray(ROW_TYPE) == int(ActionType.SETUP_SETTLEMENT))
        | (np.asarray(ROW_TYPE) == int(ActionType.SETUP_ROAD))
    )
)
_SETUP_TYPE = ROW_TYPE[_SETUP_IDX]
_SETUP_PARAMS = jax.tree.map(lambda x: x[_SETUP_IDX], ROW_PARAMS)
_N_SETUP = int(_SETUP_IDX.shape[0])

_Vec = Float[Array, "players"]


def make_setup_search(
    value: ValueFunction,
    *,
    n_players: int,
    depth: int = 3,
    temperature: float = 2.0,
    beam: int = 8,
) -> BeliefPolicy:
    """A setup-phase :class:`BeliefPolicy`: ``depth``-ply probabilistic expectimax
    over the opening placements, opponents Boltzmann-rational at ``temperature``,
    pruned to the top-``beam`` placements per node. Only meaningful at a setup
    state (the caller gates on the phase); elsewhere it returns an arbitrary
    setup index (discarded)."""
    beam = min(beam, _N_SETUP)
    players = jnp.arange(n_players, dtype=jnp.int32)

    def value_vec(layout: BoardLayout, state: BoardState) -> _Vec:
        return jax.vmap(lambda p: value(layout, state, p))(players)

    def successors(layout: BoardLayout, state: BoardState) -> tuple[BoardState, Array]:
        """Apply every setup placement; return the (batched) successor states and
        a legality mask over them."""
        avail = jax.vmap(
            lambda t, prm: action_available(layout, state, t, prm)
        )(_SETUP_TYPE, _SETUP_PARAMS)  # fmt: skip
        succ, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, _SETUP_TYPE, _SETUP_PARAMS, avail
        )
        return succ, avail

    def topk(
        layout: BoardLayout, succ: BoardState, legal: Array, mover: IntScalar
    ) -> Array:
        """Indices of the ``beam`` placements with the highest one-ply value for
        ``mover`` (the cheap prune before the deep search)."""
        oneply = jax.vmap(lambda s: value(layout, s, mover))(succ)
        return jax.lax.top_k(jnp.where(legal, oneply, -jnp.inf), beam)[1]

    def node_value(
        layout: BoardLayout, state: BoardState, d: int, me: IntScalar
    ) -> _Vec:
        """The value vector at ``state``, searching ``d`` more setup plies."""
        leaf = value_vec(layout, state)
        if d == 0:
            return leaf
        in_setup = state.phase <= jnp.uint8(GamePhase.SETUP_ROAD)
        mover = agent_selection_single(state).astype(jnp.int32)
        succ, legal = successors(layout, state)
        top = topk(layout, succ, legal, mover)
        top_succ = jax.tree.map(lambda x: x[top], succ)
        child = jax.vmap(lambda s: node_value(layout, s, d - 1, me))(
            top_succ
        )  # (beam, players)
        score = jnp.where(legal[top], child[:, mover], -jnp.inf)
        # the searcher maximizes; opponents are Boltzmann at `temperature`.
        dist = jnp.where(
            mover == me,
            jax.nn.one_hot(jnp.argmax(score), beam),
            jax.nn.softmax(score / temperature),
        )
        searched = (dist[:, None] * child).sum(0)
        return jnp.where(in_setup, searched, leaf)

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        me = player.astype(jnp.int32)
        state = sample_world(key, view, me)  # exact at setup (nothing hidden yet)
        succ, avail = successors(layout, state)
        legal = (mask[_SETUP_IDX] > 0) & avail
        top = topk(layout, succ, legal, me)
        top_succ = jax.tree.map(lambda x: x[top], succ)
        child = jax.vmap(lambda s: node_value(layout, s, depth - 1, me))(top_succ)
        my = jnp.where(legal[top], child[:, me], -jnp.inf)
        return _SETUP_IDX[top[jnp.argmax(my)]].astype(jnp.int32)

    return policy
