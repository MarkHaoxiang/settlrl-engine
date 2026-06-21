"""The Single-Observer ISMCTS tree (:func:`make_search` in
``search/__init__.py`` is its public wrapper).

:func:`make_tree` builds one search as a single XLA program over a
fixed-capacity node pool (the :class:`_Tree`), so it stays on device and
``vmap``s over lanes.

A *simulation* is one MCTS iteration over a freshly determinized world (so every
node's legal set is that world's true legality). Its phases:

  - determinize -- sample a world consistent with the belief;
  - select      -- descend the tree to an unexpanded edge (root by Sequential
                   Halving, interior by the improved-policy rule);
  - expand      -- attach the new leaf node;
  - evaluate    -- score the leaf with the value function (there is no rollout);
  - backup      -- add the leaf value to every edge on the path.

The result is the improved-policy weights ``softmax(root_logits + completed_Q)``
over the legal set; the caller supplies the root prior and takes the masked
argmax.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import (
    BoardState,
    BoolScalar,
    IntScalar,
    KeyScalar,
    Player,
)
from settlrl_engine.env import N_FLAT, flat_to_action
from settlrl_engine.mechanics.action import ActionType, action_available, apply_action
from settlrl_engine.mechanics.awards import current_player_won
from settlrl_engine.mechanics.common import agent_selection_single
from settlrl_engine.mechanics.dice import distribute_resources
from settlrl_engine.mechanics.flat import flat_available_for

from settlrl_agents.internal.rows import ROW_TYPE
from settlrl_agents.policy import PolicyPrior
from settlrl_agents.sample import sample_world
from settlrl_agents.value import Value, ValueFunction

from ._common import _ROLL_P, _ROLLS, _TIER_LOGITS, _Weights

_ROLL_T = jnp.int32(ActionType.ROLL_DICE)

# completed-Q scale: (_MAXVISIT_INIT + max_visits) * _MIX_SCALE, added to the
# prior logits (absolute, not min-max normalized).
_MIX_SCALE = 0.1
_MAXVISIT_INIT = 50.0

_FlatVec = Float[Array, f"flat={N_FLAT}"]  # a value per flat action
_LegalMask = _FlatVec  # 1.0 on the legal flat actions, else 0.0
_PriorLogits = _FlatVec  # prior logits over the flat actions
_NodeI = Int[Array, "node"]
_NodeF = Float[Array, "node"]
_EdgeI = Int[Array, "node act"]
_Table = Int[Array, "m sims"]
_EdgeF = Float[Array, "node act"]
_PathI = Int[Array, "depth"]

# Search-local scalars (the engine's IntScalar, named for their role).
_Node = IntScalar  # a node id in the _Tree (0 = root)
_Action = IntScalar  # a flat action index

TreeSearch = Callable[
    [KeyScalar, BoardLayout, BeliefView, Player, _LegalMask, _PriorLogits], _Weights
]
"""One ISMCTS search (what :func:`make_tree` returns): the searcher's legal set
and root prior in, the improved-policy ``action_weights`` out."""


# --- tree storage ---


class _Tree(NamedTuple):
    """The fixed-capacity tree a :data:`TreeSearch` builds and consumes: ``node``
    rows index up to ``num_simulations + 1`` nodes, ``act`` columns the flat
    action space."""

    # TODO(perf): the 126 setup rows (a contiguous [0:126] prefix of N_FLAT) are
    # always illegal in the main loop, where ISMCTS runs, so the `act` axis could
    # be the suffix slice. Profile first — the cost is the per-descent engine
    # steps + roll_ev, not the action-axis width, so this likely won't move it.

    mover: _NodeI
    visits: _NodeF
    children: _EdgeI  # child node id per edge, -1 = unexpanded
    n: _EdgeF  # edge visit counts
    w: _EdgeF  # edge value sums (searcher frame)
    prior: _EdgeF  # raw prior logits (root logits at node 0, interior prior elsewhere)
    raw: _NodeF  # searcher-frame node value
    size: IntScalar  # nodes in use


class _Descent(NamedTuple):
    """One simulation's forward walk; carried through the descent ``while_loop``.
    ``exp_parent`` >= 0 marks the node a new leaf attaches to (the expansion)."""

    state: BoardState
    legal: _LegalMask
    leaf: Value  # searcher-frame value to back up
    cur: _Node
    depth: IntScalar  # edges taken so far
    path_node: _PathI
    path_act: _PathI
    done: BoolScalar
    exp_parent: _Node
    exp_act: _Action
    exp_mover: Player


# --- Sequential Halving: the considered-visits schedule (static, baked) ---


def _considered_visits_seq(m: int, n: int) -> tuple[int, ...]:
    """Sequential Halving's visit schedule: length-``n`` list whose entry ``s`` is
    the visit count a candidate must hold to be selected at simulation ``s``."""
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


# --- select: the Gumbel-MuZero policy over the determinized legal set ---


def _completed_q(
    tree: _Tree, node: _Node, legal: _LegalMask, player: Player
) -> tuple[_FlatVec, _PriorLogits]:
    """The scaled completed Q-values and legal-masked prior logits at one node,
    in the mover's frame over this determinization's legal set. Unvisited actions
    take a prior-weighted blend of the node value and its visited children's Q."""
    n, w = tree.n[node], tree.w[node]
    sign = jnp.where(tree.mover[node] == player, 1.0, -1.0)
    visited = n > 0
    q = sign * jnp.where(visited, w / jnp.maximum(n, 1.0), 0.0)
    raw = sign * tree.raw[node]
    logits = jnp.where(legal > 0, tree.prior[node], -jnp.inf)
    probs = jax.nn.softmax(logits)
    # mixed value: blend the node's raw value with its prior-weighted visited-Q.
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
    tree: _Tree, node: _Node, legal: _LegalMask, player: Player
) -> _Action:
    """The interior action whose visit share most lags ``softmax(prior +
    completed_Q)``, so visits track the improved policy."""
    cq, logits = _completed_q(tree, node, legal, player)
    improved = jax.nn.softmax(jnp.where(legal > 0, logits + cq, -jnp.inf))
    sum_n = tree.n[node].sum()
    to_argmax = jnp.where(legal > 0, improved - tree.n[node] / (1.0 + sum_n), -jnp.inf)
    return jnp.argmax(to_argmax).astype(jnp.int32)


def _root_select(
    tree: _Tree,
    gumbel: _FlatVec,
    sim_index: IntScalar,
    num_considered: IntScalar,
    legal: _LegalMask,
    player: Player,
    table: _Table,
) -> _Action:
    """The root action under Gumbel + Sequential Halving: among candidates at the
    schedule's current visit count, the highest ``gumbel + prior + completed_Q``."""
    cq, logits = _completed_q(tree, jnp.int32(0), legal, player)
    visits = tree.n[0]
    considered_visit = table[num_considered, sim_index]
    norm_logits = logits - jnp.max(logits)
    penalty = jnp.where(visits == considered_visit, 0.0, -jnp.inf)
    score = jnp.maximum(-1e9, gumbel + norm_logits + cq) + penalty
    return jnp.argmax(jnp.where(legal > 0, score, -jnp.inf)).astype(jnp.int32)


# --- expand + backup: grow the tree, then propagate the leaf value ---


def _expand(tree: _Tree, walk: _Descent, leaf_prior: _PriorLogits) -> _Tree:
    """Attach the descent's new leaf node — its mover, prior, and value — at the
    next free slot (a no-op when the simulation grew no node)."""
    grew = walk.exp_parent >= 0
    new_id = tree.size  # always <= n_nodes - 1 (<=1 node added per sim)
    safe_parent = jnp.maximum(walk.exp_parent, 0)
    return tree._replace(
        mover=tree.mover.at[new_id].set(jnp.where(grew, walk.exp_mover, tree.mover[new_id])),
        prior=tree.prior.at[new_id].set(jnp.where(grew, leaf_prior, tree.prior[new_id])),
        raw=tree.raw.at[new_id].set(jnp.where(grew, walk.leaf, tree.raw[new_id])),
        children=tree.children.at[safe_parent, walk.exp_act].set(
            jnp.where(grew, new_id, tree.children[safe_parent, walk.exp_act])
        ),
        size=tree.size + grew.astype(jnp.int32),
    )  # fmt: skip


def _backup(tree: _Tree, walk: _Descent, max_depth: int) -> _Tree:
    """Add the descent's leaf value and a visit to every edge on its path."""

    def body(j: IntScalar, tree: _Tree) -> _Tree:
        node, act = walk.path_node[j], walk.path_act[j]
        use = (j < walk.depth).astype(jnp.float32)  # past the real depth -> no-op
        return tree._replace(
            visits=tree.visits.at[node].add(use),
            n=tree.n.at[node, act].add(use),
            w=tree.w.at[node, act].add(use * walk.leaf),
        )

    return cast(_Tree, jax.lax.fori_loop(0, max_depth, body, tree))


# --- search configuration ---


class _Cfg(NamedTuple):
    """The static search configuration :func:`make_tree` captures and threads
    through the (module-level) phase functions below."""

    value: ValueFunction
    prior: PolicyPrior | None  # interior-node prior; tier table when None
    num_simulations: int
    max_depth: int
    max_considered: int
    value_scale: float
    n_nodes: int  # num_simulations + 1
    table: _Table  # the Sequential-Halving considered-visits schedule


# --- engine interface: facts about one determinized state ---


def _facts(
    cfg: _Cfg, layout: BoardLayout, state: BoardState, player: Player
) -> tuple[_LegalMask, Player, BoolScalar, Value]:
    legal = flat_available_for(layout, state).astype(jnp.float32)
    term = current_player_won(state)
    win = state.current_player.astype(jnp.int32) == player
    v = jnp.tanh(cfg.value(layout, state, player) / cfg.value_scale)
    leaf = jnp.where(term, jnp.where(win, 1.0, -1.0), v)
    return legal, agent_selection_single(state).astype(jnp.int32), term, leaf


def _step(layout: BoardLayout, state: BoardState, action: _Action) -> BoardState:
    atype, aparams = flat_to_action(action)
    avail = action_available(layout, state, atype, aparams)
    nxt, _ = apply_action(layout, state, atype, aparams, avail)
    return nxt


def _interior_logits(
    cfg: _Cfg, layout: BoardLayout, state: BoardState, player: Player
) -> _PriorLogits:
    """The prior over a freshly expanded node's actions: a learned policy head if
    one was supplied, else the constant tier table."""
    if cfg.prior is None:
        return _TIER_LOGITS
    return cfg.prior(layout, state, player)


def _roll_ev(
    cfg: _Cfg, layout: BoardLayout, state: BoardState, player: Player
) -> Value:
    """Expected post-payout value of a pre-roll ``state`` over the 11 dice rolls —
    the leaf value of a ``ROLL_DICE`` edge."""
    vals = jax.vmap(
        lambda r: jnp.tanh(
            cfg.value(layout, distribute_resources(layout, state, r), player)
            / cfg.value_scale
        )
    )(_ROLLS)
    return _ROLL_P @ vals


# --- the simulation phases: determinize, then select (descend) ---


def _determinize(
    cfg: _Cfg, key: KeyScalar, layout: BoardLayout, view: BeliefView, player: Player
) -> _Descent:
    # DETERMINIZE: sample a world consistent with the belief, and seed the walk
    # over it at the root (depth 0, node 0, nothing expanded yet).
    state = sample_world(key, view, player)
    legal, _, term, leaf = _facts(cfg, layout, state, player)
    return _Descent(
        state=state,
        legal=legal,
        leaf=leaf,
        cur=jnp.int32(0),
        depth=jnp.int32(0),
        path_node=jnp.zeros((cfg.max_depth,), jnp.int32),
        path_act=jnp.zeros((cfg.max_depth,), jnp.int32),
        done=term | (legal.sum() == 0),
        exp_parent=jnp.int32(-1),
        exp_act=jnp.int32(0),
        exp_mover=jnp.int32(0),
    )


def _descend(
    cfg: _Cfg,
    tree: _Tree,
    a_root: _Action,
    walk: _Descent,
    layout: BoardLayout,
    player: Player,
) -> _Descent:
    # SELECT: from the root, follow already-expanded edges (root action by
    # Sequential Halving, interior by the improved-policy rule) until an
    # unexpanded edge or a terminal/dead end. The leaf state is EVALUATED as it is
    # reached (there is no rollout — `_facts`/`_roll_ev` give the leaf value
    # carried in `_Descent.leaf`).
    def cond(walk: _Descent) -> BoolScalar:
        return (~walk.done) & (walk.depth < cfg.max_depth)

    def body(walk: _Descent) -> _Descent:
        # `tree` is read-only here; the current node is non-terminal with a legal
        # action.
        cur0 = walk.cur
        at_root = (cur0 == 0) & (walk.depth == 0)
        a = jnp.where(at_root, a_root, _interior_select(tree, cur0, walk.legal, player))
        nstate = _step(layout, walk.state, a)
        legal2, mover2, term2, leaf2 = _facts(cfg, layout, nstate, player)
        is_leaf = tree.children[cur0, a] < 0  # unexpanded edge -> stop here
        # A dice edge's value is the expectation over the 11 rolls.
        leaf2 = jax.lax.cond(
            (ROW_TYPE[a] == _ROLL_T) & ~term2,
            lambda: _roll_ev(cfg, layout, walk.state, player),
            lambda: leaf2,
        )
        return _Descent(
            state=nstate,
            legal=legal2,
            leaf=leaf2,
            cur=jnp.where(is_leaf, cur0, tree.children[cur0, a]),
            depth=walk.depth + 1,
            path_node=walk.path_node.at[walk.depth].set(cur0),
            path_act=walk.path_act.at[walk.depth].set(a),
            done=is_leaf | term2 | (legal2.sum() == 0),
            exp_parent=jnp.where(is_leaf, cur0, jnp.int32(-1)),
            exp_act=jnp.where(is_leaf, a, jnp.int32(0)),
            exp_mover=jnp.where(is_leaf, mover2, jnp.int32(0)),
        )

    return jax.lax.while_loop(cond, body, walk)


# --- the search: one re-determinizing tree, built over the engine ---


def _run(
    cfg: _Cfg,
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    player: Player,
    mask: _LegalMask,
    root_logits: _PriorLogits,
) -> _Weights:
    player = player.astype(jnp.int32)
    key, k_gumbel = jax.random.split(key)
    keys = jax.random.split(key, cfg.num_simulations + 1)
    # Root mover is the searcher and its legal set is invariant to the hidden state
    # (its own move), so `mask` fixes the candidate set for halving; one
    # determinization gives the searcher-frame root value for the Q mix.
    r_legal = mask.astype(jnp.float32)
    _, _, _, r_leaf = _facts(cfg, layout, sample_world(keys[0], view, player), player)
    gumbel = jax.random.gumbel(k_gumbel, (N_FLAT,))
    num_considered = jnp.minimum(
        cfg.max_considered, (r_legal > 0).sum().astype(jnp.int32)
    )
    tree = _Tree(
        mover=jnp.zeros((cfg.n_nodes,), jnp.int32).at[0].set(player),
        visits=jnp.zeros((cfg.n_nodes,), jnp.float32),
        children=-jnp.ones((cfg.n_nodes, N_FLAT), jnp.int32),
        n=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32),
        w=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32),
        prior=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32).at[0].set(root_logits),
        raw=jnp.zeros((cfg.n_nodes,), jnp.float32).at[0].set(r_leaf),
        size=jnp.int32(1),
    )

    def simulate(s: IntScalar, tree: _Tree) -> _Tree:
        # DETERMINIZE: sample a world and seed the root descent over it.
        walk = _determinize(cfg, keys[s + 1], layout, view, player)

        # SELECT (root): the Sequential-Halving action, guarded to this world's
        # legality (the candidate set is hidden-state-invariant except for e.g.
        # steal targets — fall back to interior selection there).
        sim_index = jnp.minimum(
            tree.n[0].sum().astype(jnp.int32), cfg.num_simulations - 1
        )
        a_root = _root_select(
            tree, gumbel, sim_index, num_considered, r_legal, player, cfg.table
        )
        a_root = jnp.where(
            walk.legal[a_root] > 0,
            a_root,
            _interior_select(tree, jnp.int32(0), walk.legal, player),
        )

        # SELECT (descend) + EVALUATE: walk to an unexpanded leaf and score it.
        walk = _descend(cfg, tree, a_root, walk, layout, player)

        # EXPAND: attach the new leaf node (its interior prior is a forward on the
        # leaf state; a no-op when the descent grew no node).
        tree = _expand(tree, walk, _interior_logits(cfg, layout, walk.state, player))

        # BACKUP: add the leaf value to every edge on the path.
        return _backup(tree, walk, cfg.max_depth)

    tree = cast(_Tree, jax.lax.fori_loop(0, cfg.num_simulations, simulate, tree))
    # action_weights = softmax(root_logits + completed_Q) over the legal set.
    cq, logits_l = _completed_q(tree, jnp.int32(0), r_legal, player)
    return jax.nn.softmax(logits_l + cq)


# --- factory ---


def make_tree(
    value: ValueFunction,
    prior: PolicyPrior | None,
    *,
    num_simulations: int,
    max_depth: int,
    max_considered: int,
    value_scale: float,
) -> TreeSearch:
    """Build one SO-ISMCTS tree (a :data:`TreeSearch` over :func:`_run`).
    ``prior`` (when given) is the *interior* node prior — a learned policy head;
    otherwise interior nodes use the tier table. The root prior is supplied per
    call (``root_logits``). The returned function is pure (the caller
    ``jit``/``vmap``s it)."""
    cfg = _Cfg(
        value=value,
        prior=prior,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_considered=max_considered,
        value_scale=value_scale,
        n_nodes=num_simulations + 1,
        table=jnp.asarray(_considered_table(max_considered, num_simulations)),
    )
    return cast(TreeSearch, partial(_run, cfg))
