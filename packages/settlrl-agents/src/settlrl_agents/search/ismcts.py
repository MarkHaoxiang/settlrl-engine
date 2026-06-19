"""True Single-Observer ISMCTS over the engine, as one jitted on-device program.

The mctx search (``search/__init__.py``) integrates the belief by re-sampling a
world per simulation, but its *statistics* live on mctx's fixed dense action
axis with a root-only legality mask -- an action illegal under a given
simulation's world is still a selectable edge that no-ops. That is the half of
ISMCTS mctx cannot express (Cowling, Powley & Whitehouse 2012; the Canopy
reference builds a custom tree for exactly this reason).

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

**Selection is mctx's Gumbel-MuZero, on the true-legality tree** (the port that
took it to parity with the mctx search): the root runs Gumbel + Sequential
Halving (sample ``max_num_considered_actions`` by Gumbel + prior, give the
budget to the survivors by a static visit schedule), interior nodes use the
deterministic visit-count selection that tracks ``softmax(prior + completed_Q)``,
and Q-values are completed by the mixed-value transform (unvisited actions take a
prior-weighted blend of the node value and its children's Q) scaled by
``(maxvisit_init + max_visits) * mix_scale`` -- all per node, over *this*
determinization's legal set. The value frame is the same two-sided *paranoid*
reduction the mctx search uses (searcher vs the table, exact zero-sum at 2p):
every node stores the searcher-frame value, and ``completed_Q`` flips its sign at
opponent nodes so each mover's improved policy maximizes its own side. The root
prior is the one-step value sweep (lookahead); interior priors are greedy's tier
table (a constant). Leaf/prior value come from any :class:`ValueFunction`. The
returned ``action_weights`` -- ``softmax(prior + completed_Q)`` at the root -- is
the AlphaZero policy target, identical in form to mctx's; the move is its
masked argmax (as :func:`search.make_search` argmaxes mctx's weights).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, BoolScalar, IntScalar, KeyScalar
from settlrl_engine.env import N_FLAT, flat_to_action
from settlrl_engine.mechanics.action import ActionType, action_available, apply_action
from settlrl_engine.mechanics.common import agent_selection_single
from settlrl_engine.mechanics.dice import distribute_resources
from settlrl_engine.mechanics.flat import flat_available_for

from settlrl_agents.internal.rows import ROW_PARAMS, ROW_TYPE
from settlrl_agents.policy import BeliefPolicy, FlatAction, FlatMask
from settlrl_agents.sample import sample_world
from settlrl_agents.value import Value, ValueFunction

from . import (
    _NO_PROPOSE,
    _ROLL_P,
    _ROLLS,
    _TIER_LOGITS,
    PolicyWeights,
    _terminal,
    _winner,
)

_ROLL_T = jnp.int32(ActionType.ROLL_DICE)

__all__ = ["ismcts_move", "ismcts_weights", "make_ismcts", "make_ismcts_weights"]

# mctx's qtransform_completed_by_mix_value defaults: the completed Q-values are
# scaled by (_MAXVISIT_INIT + max_visits) * _MIX_SCALE before being added to the
# prior logits (rescale_values=False -- absolute, not min-max normalized).
_MIX_SCALE = 0.1
_MAXVISIT_INIT = 50.0

_Mask = Float[Array, f"flat={N_FLAT}"]
_NodeI = Int[Array, "node"]
_NodeF = Float[Array, "node"]
_EdgeI = Int[Array, "node act"]
_Table = Int[Array, "m sims"]
_EdgeF = Float[Array, "node act"]
_PathI = Int[Array, "depth"]


class _Arena(NamedTuple):
    """The fixed-capacity tree: ``node`` rows index up to ``num_simulations + 1``
    nodes, ``act`` columns the flat action space. ``children`` is the child node
    id per edge (-1 = unexpanded); ``prior`` the raw prior logits per node (the
    value sweep at the root, the tier table at interior nodes); ``raw`` the
    searcher-frame leaf value at the node (mctx's ``raw_values``, for the
    mixed-value Q completion); ``size`` the nodes in use."""

    mover: _NodeI
    visits: _NodeF
    children: _EdgeI
    n: _EdgeF  # edge visit counts
    w: _EdgeF  # edge value sums (searcher frame)
    prior: _EdgeF  # raw prior logits
    raw: _NodeF  # searcher-frame node value
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


def _considered_visits_seq(m: int, n: int) -> tuple[int, ...]:
    """Sequential Halving's visit schedule (Karnin 2013; mctx's
    ``get_sequence_of_considered_visits``): length-``n`` list whose entry ``s`` is
    the visit count a candidate must currently hold to be selected at simulation
    ``s``. Replicated here so the search carries no ``mctx`` dependency."""
    if m <= 1:
        return tuple(range(n))
    log2max = math.ceil(math.log2(m))
    seq: list[int] = []
    visits = [0] * m
    num_considered = m
    while len(seq) < n:
        extra = max(1, int(n / (log2max * num_considered)))
        for _ in range(extra):
            seq.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(seq[:n])


def _considered_table(m: int, n: int) -> np.ndarray:
    """Row ``k`` is the schedule for ``k`` considered actions (shape
    ``[m + 1, n]``); indexed by ``min(m, num_legal)`` at search time."""
    return np.asarray([_considered_visits_seq(k, n) for k in range(m + 1)], np.int32)


def _completed_q(
    arena: _Arena, node: IntScalar, legal: _Mask, player: IntScalar
) -> tuple[_Mask, _Mask]:
    """mctx's ``qtransform_completed_by_mix_value`` (rescale off) on one node,
    in the *mover's* frame, over this determinization's legal set.

    Returns the scaled completed Q-values and the legal-masked prior logits.
    Unvisited actions take the mixed value (a prior-weighted blend of the node's
    raw value and its visited children's Q); the result is scaled by
    ``(maxvisit_init + max_visits) * mix_scale`` so it is commensurate with the
    prior logits regardless of budget."""
    n, w = arena.n[node], arena.w[node]
    sign = jnp.where(arena.mover[node] == player, 1.0, -1.0)
    visited = n > 0
    q = sign * jnp.where(visited, w / jnp.maximum(n, 1.0), 0.0)
    raw = sign * arena.raw[node]
    logits = jnp.where(legal > 0, arena.prior[node], -jnp.inf)
    probs = jax.nn.softmax(logits)
    # Mixed value (Appendix D): (raw + sum_n * prior-weighted visited-Q) / (sum_n+1).
    sum_n = n.sum()
    safe = jnp.maximum(1e-37, probs)  # floor so a zero-prior visited action is finite
    sum_probs = jnp.where(visited, safe, 0.0).sum()
    weighted_q = jnp.where(
        visited, safe * q / jnp.where(sum_probs > 0, sum_probs, 1.0), 0.0
    ).sum()
    mixed = (raw + sum_n * weighted_q) / (sum_n + 1.0)
    completed = jnp.where(visited, q, mixed)
    scaled = (_MAXVISIT_INIT + n.max()) * _MIX_SCALE * completed
    return scaled, logits


def _interior_select(
    arena: _Arena, node: IntScalar, legal: _Mask, player: IntScalar
) -> IntScalar:
    """mctx's deterministic interior selection: the action whose visit share most
    lags ``softmax(prior + completed_Q)`` -- so visits track the improved policy."""
    cq, logits = _completed_q(arena, node, legal, player)
    improved = jax.nn.softmax(jnp.where(legal > 0, logits + cq, -jnp.inf))
    sum_n = arena.n[node].sum()
    to_argmax = jnp.where(legal > 0, improved - arena.n[node] / (1.0 + sum_n), -jnp.inf)
    return jnp.argmax(to_argmax).astype(jnp.int32)


def _root_select(
    arena: _Arena,
    gumbel: _Mask,
    sim_index: IntScalar,
    num_considered: IntScalar,
    legal: _Mask,
    player: IntScalar,
    table: _Table,
) -> IntScalar:
    """The root action for this simulation under Gumbel + Sequential Halving:
    among candidates at the schedule's current visit count, the highest
    ``gumbel + prior + completed_Q`` (mctx's ``score_considered``)."""
    cq, logits = _completed_q(arena, jnp.int32(0), legal, player)
    visits = arena.n[0]
    considered_visit = table[num_considered, sim_index]
    norm_logits = logits - jnp.max(logits)
    penalty = jnp.where(visits == considered_visit, 0.0, -jnp.inf)
    score = jnp.maximum(-1e9, gumbel + norm_logits + cq) + penalty
    return jnp.argmax(jnp.where(legal > 0, score, -jnp.inf)).astype(jnp.int32)


@functools.lru_cache(maxsize=8)
def _tree_fn(
    value: ValueFunction,
    value_scale: float,
    prior_scale: float,
    num_simulations: int,
    max_depth: int,
    max_considered: int,
) -> Callable[[KeyScalar, BoardLayout, BeliefView, IntScalar], _Mask]:
    """A jitted SO-ISMCTS returning the root ``action_weights`` -- the AlphaZero
    policy target ``softmax(prior + completed_Q)`` over the legal actions.
    Compiled once per ``(value, ...)`` and ``vmap``-able over lanes."""
    n_nodes = num_simulations + 1
    table = jnp.asarray(_considered_table(max_considered, num_simulations))

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

    def root_logits(
        layout: BoardLayout, state: BoardState, player: IntScalar, legal: _Mask
    ) -> _Mask:
        """The one-step value sweep as prior logits (the lookahead prior)."""
        succ, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, ROW_TYPE, ROW_PARAMS, legal > 0
        )
        vals = jax.vmap(value, in_axes=(None, 0, None))(layout, succ, player)
        return vals / prior_scale + _NO_PROPOSE

    def roll_ev(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        """E over the 11 dice rolls of the post-payout value of a pre-roll
        ``state`` -- the leaf value of a ROLL_DICE edge, so the search reads the
        roll's expectation instead of the one die the determinization sampled."""
        vals = jax.vmap(
            lambda r: jnp.tanh(
                value(layout, distribute_resources(layout, state, r), player)
                / value_scale
            )
        )(_ROLLS)
        return _ROLL_P @ vals

    @jax.jit
    def tree(
        key: KeyScalar, layout: BoardLayout, view: BeliefView, player: IntScalar
    ) -> _Mask:
        player = player.astype(jnp.int32)
        key, k_gumbel = jax.random.split(key)
        keys = jax.random.split(key, num_simulations + 1)
        root_state = sample_world(keys[0], view, player)
        # The searcher's own move: its legal set is invariant to the hidden state,
        # so this one determinization fixes the root candidate set for halving.
        r_legal, r_mover, _term, r_leaf = facts(layout, root_state, player)
        logits = root_logits(layout, root_state, player, r_legal)
        gumbel = jax.random.gumbel(k_gumbel, (N_FLAT,))
        num_considered = jnp.minimum(
            max_considered, (r_legal > 0).sum().astype(jnp.int32)
        )
        arena = _Arena(
            mover=jnp.zeros((n_nodes,), jnp.int32).at[0].set(r_mover),
            visits=jnp.zeros((n_nodes,), jnp.float32),
            children=-jnp.ones((n_nodes, N_FLAT), jnp.int32),
            n=jnp.zeros((n_nodes, N_FLAT), jnp.float32),
            w=jnp.zeros((n_nodes, N_FLAT), jnp.float32),
            prior=jnp.zeros((n_nodes, N_FLAT), jnp.float32).at[0].set(logits),
            raw=jnp.zeros((n_nodes,), jnp.float32).at[0].set(r_leaf),
            size=jnp.int32(1),
        )

        def simulate(s: IntScalar, arena: _Arena) -> _Arena:
            state = sample_world(keys[s + 1], view, player)
            legal, _, term, leaf = facts(layout, state, player)
            sim_index = jnp.minimum(
                arena.n[0].sum().astype(jnp.int32), num_simulations - 1
            )
            a_root = _root_select(
                arena, gumbel, sim_index, num_considered, r_legal, player, table
            )
            # Guard: the root candidate set is hidden-state-invariant, but fall
            # back to interior selection if this world disagrees.
            a_root = jnp.where(
                legal[a_root] > 0,
                a_root,
                _interior_select(arena, jnp.int32(0), legal, player),
            )
            d0 = _Descent(
                state=state,
                legal=legal,
                leaf=leaf,
                cur=jnp.int32(0),
                depth=jnp.int32(0),
                path_node=jnp.zeros((max_depth,), jnp.int32),
                path_act=jnp.zeros((max_depth,), jnp.int32),
                done=term | (legal.sum() == 0),
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
                at_root = (cur0 == 0) & (d.depth == 0)
                a = jnp.where(
                    at_root, a_root, _interior_select(arena, cur0, d.legal, player)
                )
                nstate = step(layout, d.state, a)
                legal2, mover2, term2, leaf2 = facts(layout, nstate, player)
                is_leaf = arena.children[cur0, a] < 0  # unexpanded edge -> expand
                leaf2 = jax.lax.cond(
                    (ROW_TYPE[a] == _ROLL_T) & ~term2,
                    lambda: roll_ev(layout, d.state, player),
                    lambda: leaf2,
                )
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
                    jnp.where(grew, _TIER_LOGITS, arena.prior[new_id])
                ),
                raw=arena.raw.at[new_id].set(
                    jnp.where(grew, d.leaf, arena.raw[new_id])
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
        # action_weights = softmax(prior + completed_Q) over the legal set.
        cq, logits_l = _completed_q(arena, jnp.int32(0), r_legal, player)
        return jax.nn.softmax(logits_l + cq)

    return tree


def make_ismcts_weights(
    value: ValueFunction,
    *,
    num_simulations: int = 32,
    max_depth: int = 12,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
) -> PolicyWeights:
    """The SO-ISMCTS improved policy as on-device weights (the AlphaZero policy
    target; symmetric with :func:`search.make_search_weights`). Renormalized over
    the passed ``mask`` -- all-zero when nothing is legal."""
    tree = _tree_fn(
        value,  # type: ignore[arg-type]  # ValueFunction is not Hashable for lru_cache
        value_scale, prior_scale, num_simulations, max_depth,
        max_num_considered_actions,
    )  # fmt: skip

    def weights(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> _Mask:
        w = jnp.where(mask, tree(key, layout, view, jnp.int32(player)), 0.0)
        return w / jnp.maximum(w.sum(), 1e-9)

    return weights


def make_ismcts(
    value: ValueFunction,
    *,
    num_simulations: int = 32,
    max_depth: int = 12,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
) -> BeliefPolicy:
    """SO-ISMCTS as a :class:`BeliefPolicy`: the masked argmax of the improved
    policy (symmetric with :func:`search.make_search`)."""
    weights = make_ismcts_weights(
        value,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_num_considered_actions=max_num_considered_actions,
        value_scale=value_scale,
        prior_scale=prior_scale,
    )

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        noise = jax.random.uniform(key, (N_FLAT,)) * 1e-4
        w = weights(key, layout, view, player, mask)
        return jnp.argmax(jnp.where(mask, w + noise, -jnp.inf))

    return policy


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
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
) -> np.ndarray:
    """Host-side improved policy (the AZ policy target / move distribution)."""
    w = make_ismcts_weights(
        value, num_simulations=num_simulations, max_depth=max_depth,
        max_num_considered_actions=max_num_considered_actions,
        value_scale=value_scale, prior_scale=prior_scale,
    )(key, layout, view, jnp.int32(player), jnp.asarray(mask))  # fmt: skip
    return np.asarray(w, dtype=np.float64)


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
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
) -> int:
    """The masked argmax of the improved policy (the SO-ISMCTS decision)."""
    w = ismcts_weights(
        key, layout, view, player, mask, value=value,
        num_simulations=num_simulations, max_depth=max_depth,
        max_num_considered_actions=max_num_considered_actions,
        value_scale=value_scale, prior_scale=prior_scale,
    )  # fmt: skip
    return int(np.argmax(np.where(np.asarray(mask) > 0, w, -np.inf)))
