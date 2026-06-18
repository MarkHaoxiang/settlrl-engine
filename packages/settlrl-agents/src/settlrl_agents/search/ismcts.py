"""True Single-Observer ISMCTS over the engine, as one jitted on-device program.

The mctx search (``search/__init__.py``) integrates the belief by re-sampling a
world per simulation, but its *statistics* live on mctx's fixed dense action
axis with visit-count selection and a root-only legality mask -- an action
illegal under a given simulation's world is still a selectable edge that
no-ops. That is the half of ISMCTS mctx cannot express (Cowling, Powley &
Whitehouse 2012; the Canopy reference builds a custom tree for exactly this
reason).

This is that custom tree, built to run like mctx does -- a single jitted XLA
program over a **fixed-capacity arena** (pre-allocated node/edge arrays sized to
``num_simulations + 1`` nodes), so the whole search stays on device with no
host round-trips and ``vmap``s over lanes. Each simulation determinizes once
(``sample_world``) and descends *forward*, stepping the engine so the legal set
at every node comes from ``flat_available_for`` on the live determinized state --
selection therefore only ever considers actions legal in *this* world (true
per-simulation legality, no no-op edges). The descent is a ``while_loop`` that
stops at the first unexpanded edge / terminal, so each simulation pays only its
own depth of engine steps, not a fixed ``max_depth``.

Selection is PUCT with the prior renormalized over the legal set and first-play
urgency; the value frame is the same two-sided *paranoid* reduction the mctx
search uses (searcher vs the table, exact zero-sum at 2p): every node stores the
searcher-frame value and selects ``sign * Q + U`` with ``sign = +1`` at the
searcher's nodes, ``-1`` at the rest. The root prior is the one-step value sweep
(lookahead); interior priors are greedy's tier table (a constant -- no
per-expansion engine sweep). Leaf/prior value come from any
:class:`ValueFunction`. Built additively beside the mctx search for a strength
comparison before that path is retired.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, BoolScalar, IntScalar, KeyScalar
from settlrl_engine.env import N_FLAT, flat_to_action
from settlrl_engine.mechanics.action import action_available, apply_action
from settlrl_engine.mechanics.common import agent_selection_single
from settlrl_engine.mechanics.flat import flat_available_for

from settlrl_agents.internal.rows import ROW_PARAMS, ROW_TYPE
from settlrl_agents.policy import FlatMask
from settlrl_agents.sample import sample_world
from settlrl_agents.value import Value, ValueFunction

from . import _TIER_LOGITS, _terminal, _winner

__all__ = ["ismcts_move", "ismcts_weights"]

_C_PUCT = 1.25  # PUCT exploration constant (interior + root)
_FPU = 0.25  # first-play-urgency reduction: unvisited Q = node mean - this

_Mask = Float[Array, f"flat={N_FLAT}"]
_NodeI = Int[Array, "node"]
_NodeB = Bool[Array, "node"]
_NodeF = Float[Array, "node"]
_EdgeI = Int[Array, "node act"]
_EdgeF = Float[Array, "node act"]
_PathI = Int[Array, "depth"]


class _Arena(NamedTuple):
    """The fixed-capacity tree: ``node`` rows index up to ``num_simulations + 1``
    nodes, ``act`` columns the flat action space. ``children`` is the child node
    id per edge (-1 = unexpanded); ``size`` the nodes in use."""

    mover: _NodeI
    visits: _NodeF
    children: _EdgeI
    n: _EdgeF  # edge visit counts
    w: _EdgeF  # edge value sums (searcher frame)
    prior: _EdgeF
    size: IntScalar


class _Descent(NamedTuple):
    """One simulation's forward walk; carried through the descent ``while_loop``.
    ``exp_parent`` >= 0 marks the node a new leaf attaches to (the expansion)."""

    state: BoardState
    legal: _Mask
    leaf: Value  # searcher-frame value to back up
    cur: IntScalar
    depth: IntScalar  # edges taken so far
    path_node: _PathI
    path_act: _PathI
    done: BoolScalar
    exp_parent: IntScalar
    exp_act: IntScalar
    exp_mover: IntScalar


def _softmax_legal(logits: Array, legal: _Mask) -> _Mask:
    """Prior over the legal actions (0 elsewhere)."""
    masked = jnp.where(legal > 0, logits, -jnp.inf)
    masked = masked - masked.max()
    p = jnp.where(legal > 0, jnp.exp(masked), 0.0)
    return p / jnp.maximum(p.sum(), 1e-9)


def _puct(arena: _Arena, node: IntScalar, legal: _Mask, player: IntScalar) -> IntScalar:
    """The PUCT pick among this determinization's legal actions. Prior
    renormalized over the legal set; ``sign`` makes opponents minimize the
    searcher value; unvisited actions take first-play-urgency Q = node mean minus
    ``_FPU`` (so a visited action is no permanent head start over its siblings)."""
    n, w, p = arena.n[node], arena.w[node], arena.prior[node]
    visits = arena.visits[node]
    sign = jnp.where(arena.mover[node] == player, 1.0, -1.0)
    node_v = w.sum() / jnp.maximum(visits, 1.0)
    fpu = node_v - _FPU * sign
    q = jnp.where(n > 0, w / jnp.maximum(n, 1.0), fpu)
    pp = p * legal
    pp = pp / jnp.maximum(pp.sum(), 1e-9)
    u = _C_PUCT * pp * jnp.sqrt(visits + 1.0) / (1.0 + n)
    return jnp.argmax(jnp.where(legal > 0, sign * q + u, -jnp.inf)).astype(jnp.int32)


@functools.lru_cache(maxsize=8)
def _tree_fn(
    value: ValueFunction,
    value_scale: float,
    prior_scale: float,
    num_simulations: int,
    max_depth: int,
) -> Callable[[KeyScalar, BoardLayout, BeliefView, IntScalar], _Mask]:
    """A jitted SO-ISMCTS that returns the root edge-visit counts. Compiled once
    per ``(value, ...)`` and ``vmap``-able over lanes."""
    n_nodes = num_simulations + 1

    def facts(
        layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> tuple[_Mask, IntScalar, BoolScalar, Value]:
        legal = flat_available_for(layout, state).astype(jnp.float32)
        term = _terminal(state)
        win = _winner(state) == player
        v = jnp.tanh(value(layout, state, player) / value_scale)
        leaf = jnp.where(term, jnp.where(win, 1.0, -1.0), v)
        return legal, agent_selection_single(state).astype(jnp.int32), term, leaf

    def step(layout: BoardLayout, state: BoardState, action: IntScalar) -> BoardState:
        atype, aparams = flat_to_action(action)
        avail = action_available(layout, state, atype, aparams)
        nxt, _ = apply_action(layout, state, atype, aparams, avail)
        return nxt

    def root_prior(
        layout: BoardLayout, state: BoardState, player: IntScalar, legal: _Mask
    ) -> _Mask:
        succ, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, ROW_TYPE, ROW_PARAMS, legal > 0
        )
        vals = jax.vmap(value, in_axes=(None, 0, None))(layout, succ, player)
        return _softmax_legal(
            _TIER_LOGITS + jnp.tanh(vals / value_scale) / prior_scale, legal
        )

    @jax.jit
    def tree(
        key: KeyScalar, layout: BoardLayout, view: BeliefView, player: IntScalar
    ) -> _Mask:
        player = player.astype(jnp.int32)
        keys = jax.random.split(key, num_simulations + 1)
        root_state = sample_world(keys[0], view, player)
        r_legal, r_mover, _r_term, _ = facts(layout, root_state, player)
        arena = _Arena(
            mover=jnp.zeros((n_nodes,), jnp.int32).at[0].set(r_mover),
            visits=jnp.zeros((n_nodes,), jnp.float32),
            children=-jnp.ones((n_nodes, N_FLAT), jnp.int32),
            n=jnp.zeros((n_nodes, N_FLAT), jnp.float32),
            w=jnp.zeros((n_nodes, N_FLAT), jnp.float32),
            prior=jnp.zeros((n_nodes, N_FLAT), jnp.float32)
            .at[0]
            .set(root_prior(layout, root_state, player, r_legal)),
            size=jnp.int32(1),
        )

        def simulate(s: IntScalar, arena: _Arena) -> _Arena:
            state = sample_world(keys[s + 1], view, player)
            legal, _, term, leaf = facts(layout, state, player)
            d0 = _Descent(
                state=state,
                legal=legal,
                leaf=leaf,
                cur=jnp.int32(0),
                depth=jnp.int32(0),
                path_node=jnp.zeros((max_depth,), jnp.int32),
                path_act=jnp.zeros((max_depth,), jnp.int32),
                done=term | (legal.sum() == 0),  # nothing to search from the root
                exp_parent=jnp.int32(-1),
                exp_act=jnp.int32(0),
                exp_mover=jnp.int32(0),
            )

            def cond(d: _Descent) -> BoolScalar:
                return (~d.done) & (d.depth < max_depth)

            def body(d: _Descent) -> _Descent:
                # `arena` (this sim's tree) is read-only during the descent. The
                # current node is guaranteed non-terminal with a legal action.
                cur0 = d.cur
                a = _puct(arena, cur0, d.legal, player)
                nstate = step(layout, d.state, a)
                legal2, mover2, term2, leaf2 = facts(layout, nstate, player)
                is_leaf = arena.children[cur0, a] < 0  # unexpanded edge -> expand
                return _Descent(
                    state=nstate,
                    legal=legal2,
                    leaf=leaf2,
                    cur=jnp.where(is_leaf, cur0, arena.children[cur0, a]),
                    depth=d.depth + 1,
                    path_node=d.path_node.at[d.depth].set(cur0),
                    path_act=d.path_act.at[d.depth].set(a),
                    done=is_leaf | term2 | (legal2.sum() == 0),
                    exp_parent=jnp.where(is_leaf, cur0, jnp.int32(-1)),
                    exp_act=jnp.where(is_leaf, a, jnp.int32(0)),
                    exp_mover=jnp.where(is_leaf, mover2, jnp.int32(0)),
                )

            d = jax.lax.while_loop(cond, body, d0)

            # Expansion: attach the new leaf (guarded -- a sim that only revisited
            # the existing tree to max_depth grows nothing).
            grew = d.exp_parent >= 0
            new_id = arena.size  # always <= n_nodes - 1 (<=1 node added per sim)
            safe_parent = jnp.maximum(d.exp_parent, 0)
            arena = arena._replace(
                mover=arena.mover.at[new_id].set(
                    jnp.where(grew, d.exp_mover, arena.mover[new_id])
                ),
                prior=arena.prior.at[new_id].set(
                    jnp.where(grew, _softmax_legal(_TIER_LOGITS, d.legal), arena.prior[new_id])
                ),
                children=arena.children.at[safe_parent, d.exp_act].set(
                    jnp.where(grew, new_id, arena.children[safe_parent, d.exp_act])
                ),
                size=arena.size + grew.astype(jnp.int32),
            )

            def backup(j: IntScalar, ar: _Arena) -> _Arena:
                node, act = d.path_node[j], d.path_act[j]
                use = (j < d.depth).astype(jnp.float32)
                return ar._replace(
                    visits=ar.visits.at[node].add(use),
                    n=ar.n.at[node, act].add(use),
                    w=ar.w.at[node, act].add(use * d.leaf),
                )

            return cast(_Arena, jax.lax.fori_loop(0, max_depth, backup, arena))

        arena = cast(_Arena, jax.lax.fori_loop(0, num_simulations, simulate, arena))
        return arena.n[0]

    return tree


def ismcts_weights(
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    player: IntScalar,
    mask: FlatMask,
    *,
    value: ValueFunction,
    num_simulations: int = 32,
    max_depth: int = 12,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
) -> np.ndarray:
    """Root edge-visit distribution from one SO-ISMCTS tree (normalized over the
    legal actions) -- the improved policy / AlphaZero target."""
    tree = _tree_fn(value, value_scale, prior_scale, num_simulations, max_depth)  # type: ignore[arg-type]
    counts = np.asarray(tree(key, layout, view, jnp.int32(player)))
    legal = np.asarray(mask) > 0
    counts = np.where(legal, counts, 0.0)
    total = counts.sum()
    return counts / total if total > 0 else legal / max(int(legal.sum()), 1)


def ismcts_move(
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    player: IntScalar,
    mask: FlatMask,
    *,
    value: ValueFunction,
    num_simulations: int = 32,
    max_depth: int = 12,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
) -> int:
    """The most-visited legal root action (the SO-ISMCTS decision)."""
    w = ismcts_weights(
        key, layout, view, player, mask, value=value,
        num_simulations=num_simulations, max_depth=max_depth,
        value_scale=value_scale, prior_scale=prior_scale,
    )  # fmt: skip
    return int(np.argmax(np.where(np.asarray(mask) > 0, w, -np.inf)))
