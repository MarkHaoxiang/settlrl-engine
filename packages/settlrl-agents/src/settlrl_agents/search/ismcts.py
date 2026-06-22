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
from settlrl_engine.board.dev_cards import N_DEV_CARD_TYPES
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import (
    BoardState,
    BoolScalar,
    IntScalar,
    KeyScalar,
    Player,
)
from settlrl_engine.env import N_FLAT, flat_to_action
from settlrl_engine.mechanics.action import (
    ActionParams,
    ActionType,
    action_available,
    apply_action,
)
from settlrl_engine.mechanics.awards import current_player_won
from settlrl_engine.mechanics.common import agent_selection_single
from settlrl_engine.mechanics.dice import distribute_resources
from settlrl_engine.mechanics.flat import flat_available_for

from settlrl_agents.internal.rows import ROW_TYPE
from settlrl_agents.policy import PolicyPrior
from settlrl_agents.sample import sample_world
from settlrl_agents.value import Value, ValueFunction

from ._common import _ILLEGAL, _ROLL_P, _ROLLS, _TIER_LOGITS, _Weights

_ROLL_T = jnp.int32(ActionType.ROLL_DICE)
_BUY_T = jnp.int32(ActionType.BUY_DEVELOPMENT_CARD)

# Explicit chance nodes: a decision node's stochastic action (a dice roll, or a
# dev-card buy when `dev_chance`) leads to a *chance* node whose children are the
# real outcomes, sampled at their true probability and applied via the engine's
# forced-outcome seam. `_N_OUTCOMES` bounds the per-chance-node child axis: 11
# dice outcomes (2..12) or `N_DEV_CARD_TYPES` card types.
_DECISION = jnp.int32(0)
_CHANCE = jnp.int32(1)
_N_DICE = 11
_N_OUTCOMES = max(_N_DICE, N_DEV_CARD_TYPES)
_ROLL_LOGITS = jnp.log(_ROLL_P)  # the true two-dice outcome distribution

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
    [KeyScalar, BoardLayout, BeliefView, Player, _LegalMask, _PriorLogits],
    tuple[_Weights, Value],
]
"""One ISMCTS search (what :func:`make_tree` returns): the searcher's legal set
and root prior in, the improved-policy ``action_weights`` and the searched root
value (searcher frame) out."""


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
    children: _EdgeI  # child node id per edge, -1 = unexpanded
    n: _EdgeF  # edge visit counts
    w: _EdgeF  # edge value sums (searcher frame)
    prior: _EdgeF  # raw prior logits (root logits at node 0, interior prior elsewhere)
    raw: _NodeF  # searcher-frame node value
    kind: _NodeI  # _DECISION or _CHANCE (chance nodes index children by outcome)
    size: IntScalar  # nodes in use


class _Descent(NamedTuple):
    """One simulation's forward walk; carried through the descent ``while_loop``.
    ``exp_parent`` >= 0 marks the node a new leaf attaches to (the expansion). The
    leaf is scored once *after* the loop (:func:`_evaluate`); ``prev_state`` carries
    the pre-step parent for a dice leaf's roll expectation, and is ``None`` (not
    carried) when ``expected_rolls`` is off — the single-sample leaf needs no
    parent."""

    state: BoardState
    prev_state: BoardState | None  # step's parent for _roll_ev; None if single-sample
    key: KeyScalar  # descent RNG, for sampling chance outcomes (chance_nodes)
    cur: _Node
    depth: IntScalar  # edges taken so far
    path_node: _PathI
    path_act: _PathI
    done: BoolScalar
    exp_parent: _Node
    exp_act: _Action
    exp_kind: _Node  # kind of the leaf node being expanded (chance_nodes)


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
    floored_probs = jnp.maximum(1e-37, probs)  # floor a zero-prior visited action
    sum_probs = jnp.where(visited, floored_probs, 0.0).sum()
    weighted_q = jnp.where(
        visited, floored_probs * q / jnp.where(sum_probs > 0, sum_probs, 1.0), 0.0
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
    cfg: _Cfg,
    gumbel: _FlatVec,
    num_considered: IntScalar,
    candidates: _LegalMask,
    world_legal: _LegalMask,
    player: Player,
) -> _Action:
    """The root action for this simulation: the Sequential-Halving pick over the
    fixed ``candidates`` set (the highest ``gumbel + prior + completed_Q`` among
    candidates at the schedule's current visit count), guarded to this
    determinization's ``world_legal``. A candidate can be illegal in some worlds
    (e.g. steal targets), so fall back to interior selection when the pick is."""
    sim_index = jnp.minimum(tree.n[0].sum().astype(jnp.int32), cfg.num_simulations - 1)
    cq, logits = _completed_q(tree, jnp.int32(0), candidates, player)
    visits = tree.n[0]
    considered_visit = cfg.table[num_considered, sim_index]
    norm_logits = logits - jnp.max(logits)
    penalty = jnp.where(visits == considered_visit, 0.0, -jnp.inf)
    score = jnp.maximum(-1e9, gumbel + norm_logits + cq) + penalty
    a_root = jnp.argmax(jnp.where(candidates > 0, score, -jnp.inf)).astype(jnp.int32)
    return jnp.where(
        world_legal[a_root] > 0,
        a_root,
        _interior_select(tree, jnp.int32(0), world_legal, player),
    )


# --- expand + backup: grow the tree, then propagate the leaf value ---


def _expand(
    tree: _Tree, walk: _Descent, value: Value, mover: Player, leaf_prior: _PriorLogits
) -> _Tree:
    """Attach the descent's new leaf node — its mover, prior, and value — at the
    next free slot (a no-op when the simulation grew no node)."""
    grew = walk.exp_parent >= 0
    new_id = tree.size  # always <= n_nodes - 1 (<=1 node added per sim)
    safe_parent = jnp.maximum(walk.exp_parent, 0)
    return tree._replace(
        mover=tree.mover.at[new_id].set(jnp.where(grew, mover, tree.mover[new_id])),
        prior=tree.prior.at[new_id].set(jnp.where(grew, leaf_prior, tree.prior[new_id])),
        raw=tree.raw.at[new_id].set(jnp.where(grew, value, tree.raw[new_id])),
        kind=tree.kind.at[new_id].set(jnp.where(grew, walk.exp_kind, tree.kind[new_id])),
        children=tree.children.at[safe_parent, walk.exp_act].set(
            jnp.where(grew, new_id, tree.children[safe_parent, walk.exp_act])
        ),
        size=tree.size + grew.astype(jnp.int32),
    )  # fmt: skip


def _backup(tree: _Tree, walk: _Descent, value: Value, max_depth: int) -> _Tree:
    """Add the descent's leaf value and a visit to every edge on its path."""

    def body(j: IntScalar, tree: _Tree) -> _Tree:
        node, act = walk.path_node[j], walk.path_act[j]
        use = (j < walk.depth).astype(jnp.float32)  # past the real depth -> no-op
        return tree._replace(
            n=tree.n.at[node, act].add(use),
            w=tree.w.at[node, act].add(use * value),
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
    expected_rolls: bool  # roll leaf = exact 11-roll expectation; else 1 sampled roll
    chance_nodes: bool  # explicit dice (+dev) chance nodes in the tree
    dev_chance: bool  # also make BUY_DEVELOPMENT_CARD a chance node (chance_nodes)
    n_nodes: int  # num_simulations + 1
    table: _Table  # the Sequential-Halving considered-visits schedule


# --- engine interface: facts about one determinized state ---


def _legal_mask(layout: BoardLayout, state: BoardState) -> _LegalMask:
    return flat_available_for(layout, state).astype(jnp.float32)


def _value(cfg: _Cfg, layout: BoardLayout, state: BoardState, player: Player) -> Value:
    """Searcher-frame value of a state: ±1 once the mover has won, else the
    tanh-squashed value function."""
    terminal = current_player_won(state)
    win = state.current_player.astype(jnp.int32) == player
    eval_value = jnp.tanh(cfg.value(layout, state, player) / cfg.value_scale)
    return jnp.where(terminal, jnp.where(win, 1.0, -1.0), eval_value)


def _step(layout: BoardLayout, state: BoardState, action: _Action) -> BoardState:
    atype, aparams = flat_to_action(action)
    avail = action_available(layout, state, atype, aparams)
    next_state, _ = apply_action(layout, state, atype, aparams, avail)
    return next_state


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


# --- explicit chance nodes: stochastic actions resolve through nature nodes ---


def _select_state(pred: BoolScalar, a: BoardState, b: BoardState) -> BoardState:
    """``a`` where ``pred`` else ``b`` (a state pytree ``where``)."""
    return cast(BoardState, jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b))


def _is_stochastic(cfg: _Cfg, action: _Action) -> BoolScalar:
    """Whether taking ``action`` lands in a chance node: a dice roll always, a
    dev-card buy when ``dev_chance``."""
    atype = ROW_TYPE[action]
    return cast(BoolScalar, (atype == _ROLL_T) | (cfg.dev_chance & (atype == _BUY_T)))


def _sample_outcome(
    cfg: _Cfg, key: KeyScalar, pending: _Action, state: BoardState
) -> IntScalar:
    """Sample a chance outcome index for the ``pending`` stochastic action: a dice
    outcome ``0..10`` (roll-2) at the true probabilities, or a dev-card type at the
    deck's current composition."""
    is_buy = cfg.dev_chance & (ROW_TYPE[pending] == _BUY_T)
    deck = state.dev_deck.astype(jnp.float32)
    deck_logits = jnp.where(
        deck > 0, jnp.log(deck / jnp.maximum(deck.sum(), 1.0)), _ILLEGAL
    )
    buy_logits = jnp.concatenate(
        [deck_logits, jnp.full((_N_OUTCOMES - N_DEV_CARD_TYPES,), _ILLEGAL)]
    )
    roll_logits = jnp.concatenate(
        [_ROLL_LOGITS, jnp.full((_N_OUTCOMES - _N_DICE,), _ILLEGAL)]
    )
    logits = jnp.where(is_buy, buy_logits, roll_logits)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def _resolve_chance(
    cfg: _Cfg,
    layout: BoardLayout,
    state: BoardState,
    pending: _Action,
    outcome: IntScalar,
) -> BoardState:
    """Apply the ``pending`` stochastic action with its sampled ``outcome`` forced
    through the engine seam (roll ``outcome+2``, or dev-card type ``outcome``)."""
    atype, _ = flat_to_action(pending)
    is_buy = cfg.dev_chance & (ROW_TYPE[pending] == _BUY_T)
    forced = jnp.where(is_buy, outcome + 1, outcome + 2).astype(jnp.int32)
    params = ActionParams(idx=forced, target=jnp.int32(0))
    avail = action_available(layout, state, atype, params)
    next_state, _ = apply_action(layout, state, atype, params, avail)
    return next_state


# --- the simulation phases: determinize, then select (descend) ---


def _determinize(
    cfg: _Cfg, key: KeyScalar, layout: BoardLayout, view: BeliefView, player: Player
) -> _Descent:
    # DETERMINIZE: sample a world consistent with the belief, and seed the walk
    # over it at the root (depth 0, node 0, nothing expanded yet).
    k_world, k_desc = jax.random.split(key)
    state = sample_world(k_world, view, player)
    return _Descent(
        state=state,
        prev_state=state if cfg.expected_rolls else None,
        key=k_desc,
        cur=jnp.int32(0),
        depth=jnp.int32(0),
        path_node=jnp.zeros((cfg.max_depth,), jnp.int32),
        path_act=jnp.zeros((cfg.max_depth,), jnp.int32),
        done=current_player_won(state),
        exp_parent=jnp.int32(-1),
        exp_act=jnp.int32(0),
        exp_kind=_DECISION,
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
    # Sequential Halving, interior by the improved-policy rule) until an unexpanded
    # edge or a terminal state. No value function runs in the loop — the leaf is
    # scored once after it by `_evaluate`; the body does only the cheap terminal
    # test needed to stop.
    def cond(walk: _Descent) -> BoolScalar:
        return (~walk.done) & (walk.depth < cfg.max_depth)

    def body(walk: _Descent) -> _Descent:
        # `tree` is read-only here; the current node is non-terminal with a legal
        # action.
        legal = _legal_mask(layout, walk.state)
        at_root = (walk.cur == 0) & (walk.depth == 0)
        action = jnp.where(
            at_root, a_root, _interior_select(tree, walk.cur, legal, player)
        )
        next_state = _step(layout, walk.state, action)
        is_leaf = tree.children[walk.cur, action] < 0  # unexpanded edge -> stop here
        return _Descent(
            state=next_state,
            prev_state=walk.state if cfg.expected_rolls else None,
            key=walk.key,
            cur=jnp.where(is_leaf, walk.cur, tree.children[walk.cur, action]),
            depth=walk.depth + 1,
            path_node=walk.path_node.at[walk.depth].set(walk.cur),
            path_act=walk.path_act.at[walk.depth].set(action),
            done=is_leaf | current_player_won(next_state),
            exp_parent=jnp.where(is_leaf, walk.cur, jnp.int32(-1)),
            exp_act=jnp.where(is_leaf, action, jnp.int32(0)),
            exp_kind=_DECISION,
        )

    def body_chance(walk: _Descent) -> _Descent:
        # Decision/chance state machine. A decision node selects an action (root by
        # Sequential Halving, else the improved-policy rule); a *stochastic* action
        # (roll, or dev-buy under dev_chance) defers to a chance node (the
        # afterstate, action unapplied). A chance node samples its outcome at the
        # true probability and applies the forced transition. Both branches are
        # computed every step (vmap runs both) and `where`-selected by node kind.
        key, k_out = jax.random.split(walk.key)
        is_chance = tree.kind[walk.cur] == _CHANCE
        legal = _legal_mask(layout, walk.state)
        at_root = (walk.cur == 0) & (walk.depth == 0)
        action = jnp.where(
            at_root, a_root, _interior_select(tree, walk.cur, legal, player)
        )
        stoch = _is_stochastic(cfg, action)
        pending = walk.path_act[walk.depth - 1]  # the action that created this node
        outcome = _sample_outcome(cfg, k_out, pending, walk.state)

        # decision step: defer a stochastic action (afterstate = current state),
        # else apply it. chance step: resolve the pending action's forced outcome.
        dec_state = _select_state(stoch, walk.state, _step(layout, walk.state, action))
        chance_state = _resolve_chance(cfg, layout, walk.state, pending, outcome)
        next_state = _select_state(is_chance, chance_state, dec_state)
        edge = jnp.where(is_chance, outcome, action)
        next_kind = jnp.where(
            is_chance, _DECISION, jnp.where(stoch, _CHANCE, _DECISION)
        )
        is_leaf = tree.children[walk.cur, edge] < 0
        return _Descent(
            state=next_state,
            prev_state=None,
            key=key,
            cur=jnp.where(is_leaf, walk.cur, tree.children[walk.cur, edge]),
            depth=walk.depth + 1,
            path_node=walk.path_node.at[walk.depth].set(walk.cur),
            path_act=walk.path_act.at[walk.depth].set(edge),
            done=is_leaf | current_player_won(next_state),
            exp_parent=jnp.where(is_leaf, walk.cur, jnp.int32(-1)),
            exp_act=jnp.where(is_leaf, edge, jnp.int32(0)),
            exp_kind=next_kind,
        )

    return jax.lax.while_loop(cond, body_chance if cfg.chance_nodes else body, walk)


def _evaluate(
    cfg: _Cfg, layout: BoardLayout, walk: _Descent, player: Player
) -> tuple[Value, Player]:
    # EVALUATE: score the descent's leaf once, here, rather than on every step.
    # The leaf's mover is stored on the new node for the backup frame.
    #
    # `_step` already applies one sampled dice outcome, so the plain leaf value is
    # a single-sample roll. With `expected_rolls`, a roll-edge leaf instead takes
    # the exact 11-roll expectation from the pre-roll parent (`prev_state`) —
    # variance reduction at 11x the value-fn calls. Keying off the last path action
    # (not just a grown edge) also covers a `max_depth` truncation onto a roll.
    leaf = walk.state
    mover = agent_selection_single(leaf).astype(jnp.int32)
    if not cfg.expected_rolls:
        return _value(cfg, layout, leaf, player), mover
    assert walk.prev_state is not None  # carried whenever expected_rolls is on
    last_action = walk.path_act[walk.depth - 1]  # edge into the leaf
    roll_leaf = (walk.depth > 0) & (ROW_TYPE[last_action] == _ROLL_T)
    value = jnp.where(
        roll_leaf & ~current_player_won(leaf),
        _roll_ev(cfg, layout, walk.prev_state, player),
        _value(cfg, layout, leaf, player),
    )
    return value, mover


# --- the search: one re-determinizing tree, built over the engine ---


def _run(
    cfg: _Cfg,
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    player: Player,
    mask: _LegalMask,
    root_logits: _PriorLogits,
) -> tuple[_Weights, Value]:
    player = player.astype(jnp.int32)
    key, k_gumbel = jax.random.split(key)
    keys = jax.random.split(key, cfg.num_simulations + 1)
    # Root mover is the searcher and its legal set is invariant to the hidden state
    # (its own move), so `mask` fixes the candidate set for halving; one
    # determinization gives the searcher-frame root value for the Q mix.
    root_legal = mask.astype(jnp.float32)
    root_value = _value(cfg, layout, sample_world(keys[0], view, player), player)
    gumbel = jax.random.gumbel(k_gumbel, (N_FLAT,))
    num_considered = jnp.minimum(
        cfg.max_considered, (root_legal > 0).sum().astype(jnp.int32)
    )
    tree = _Tree(
        mover=jnp.zeros((cfg.n_nodes,), jnp.int32).at[0].set(player),
        children=-jnp.ones((cfg.n_nodes, N_FLAT), jnp.int32),
        n=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32),
        w=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32),
        prior=jnp.zeros((cfg.n_nodes, N_FLAT), jnp.float32).at[0].set(root_logits),
        raw=jnp.zeros((cfg.n_nodes,), jnp.float32).at[0].set(root_value),
        kind=jnp.zeros((cfg.n_nodes,), jnp.int32),  # root + all nodes default DECISION
        size=jnp.int32(1),
    )

    def simulate(s: IntScalar, tree: _Tree) -> _Tree:
        # DETERMINIZE: sample a world and seed the root descent over it.
        walk = _determinize(cfg, keys[s + 1], layout, view, player)

        # SELECT (root): the Sequential-Halving action for this simulation.
        world_legal = _legal_mask(layout, walk.state)
        a_root = _root_select(
            tree, cfg, gumbel, num_considered, root_legal, world_legal, player
        )

        # SELECT (descend): walk to an unexpanded leaf (no value work in the loop).
        walk = _descend(cfg, tree, a_root, walk, layout, player)

        # EVALUATE: score the leaf once, here, not per descent step.
        value, mover = _evaluate(cfg, layout, walk, player)

        # EXPAND: attach the new leaf node (its interior prior is a forward on the
        # leaf state; a no-op when the descent grew no node).
        leaf_prior = _interior_logits(cfg, layout, walk.state, player)
        tree = _expand(tree, walk, value, mover, leaf_prior)

        # BACKUP: add the leaf value to every edge on the path.
        return _backup(tree, walk, value, cfg.max_depth)

    tree = cast(_Tree, jax.lax.fori_loop(0, cfg.num_simulations, simulate, tree))
    # action_weights = softmax(root_logits + completed_Q) over the legal set.
    cq, legal_logits = _completed_q(tree, jnp.int32(0), root_legal, player)
    weights = jax.nn.softmax(legal_logits + cq)
    # root value = visit-weighted mean of the root edges' backed-up values (the
    # root mover is the searcher, so w[0] is already in the searcher's frame).
    root_q = tree.w[0].sum() / jnp.maximum(tree.n[0].sum(), 1.0)
    return weights, root_q


# --- factory ---


def make_tree(
    value: ValueFunction,
    prior: PolicyPrior | None,
    *,
    num_simulations: int,
    max_depth: int,
    max_considered: int,
    value_scale: float,
    expected_rolls: bool = True,
    chance_nodes: bool = False,
    dev_chance: bool = True,
) -> TreeSearch:
    """Build one SO-ISMCTS tree (a :data:`TreeSearch` over :func:`_run`).
    ``prior`` (when given) is the *interior* node prior — a learned policy head;
    otherwise interior nodes use the tier table. The root prior is supplied per
    call (``root_logits``). The returned function is pure (the caller
    ``jit``/``vmap``s it).

    ``expected_rolls`` scores a dice-edge leaf by the exact 11-roll expectation
    (variance reduction at 11x the value-fn calls); False uses the single sampled
    post-roll state already produced by the step (much cheaper, noisier leaf).

    ``chance_nodes`` resolves stochastic transitions through explicit chance nodes
    in the tree (dice always, dev-card buys when ``dev_chance``) — nature's move
    sampled at its true probability and applied via the engine's forced-outcome
    seam, so the search can plan *past* a roll. It supersedes ``expected_rolls``
    (a leaf-only roll EV), so the two are mutually exclusive."""
    cfg = _Cfg(
        value=value,
        prior=prior,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_considered=max_considered,
        value_scale=value_scale,
        expected_rolls=expected_rolls and not chance_nodes,
        chance_nodes=chance_nodes,
        dev_chance=dev_chance,
        n_nodes=num_simulations + 1,
        table=jnp.asarray(_considered_table(max_considered, num_simulations)),
    )
    return cast(TreeSearch, partial(_run, cfg))
