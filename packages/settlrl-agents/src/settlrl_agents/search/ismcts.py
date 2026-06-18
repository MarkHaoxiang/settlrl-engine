"""True Single-Observer ISMCTS over the engine (prototype, single-board).

The mctx search (``search/__init__.py``) integrates the belief by re-sampling a
world per simulation, but its *statistics* live on mctx's fixed dense action
axis with visit-count selection and a root-only legality mask -- an action
illegal under a given simulation's world is still a selectable edge that
no-ops, and there are no availability counts. That is the half of ISMCTS mctx
cannot express (Cowling, Powley & Whitehouse 2012; the Canopy reference builds a
custom tree for exactly this reason).

This is that custom tree. Each simulation determinizes once (``sample_world``)
and descends *forward*, stepping the engine so the legal set at every node comes
from ``flat_available`` on the live determinized state -- selection therefore
only ever considers actions legal in *this* world (true per-simulation legality,
no no-op edges). Selection is PUCT with the prior renormalized over the legal
set; the availability count (how often an action was legal here) is tracked per
Cowling. The value frame is the same two-sided *paranoid* reduction the mctx
search uses (searcher vs the table; exact zero-sum at 2p): every node stores the
searcher-frame value and selects ``sign * Q + U`` with ``sign = +1`` at the
searcher's nodes, ``-1`` at the opponents'.

Prototype scope: a host-driven tree calling jitted single-board engine ops (not
vmapped over lanes like mctx). It is built additively beside the mctx search so
the two can be strength-compared before the mctx path is retired. The leaf and
prior come from any :class:`ValueFunction` (the heuristic now; the learned AZ
net once it ships).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar, KeyScalar
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

_Facts = tuple[Array, IntScalar, Array, Value]  # legal mask, mover, terminal, leaf v
_Advance = tuple[BoardState, Array, IntScalar, Array, Value]


@functools.lru_cache(maxsize=8)
def _ops(
    value: ValueFunction, value_scale: float, prior_scale: float
) -> tuple[Callable[..., _Facts], Callable[..., _Advance], Callable[..., Array]]:
    """The jitted single-board engine ops, **compiled once per ``value``** (the
    cache is the fix for recompiling every move). ``facts`` reads a node's public
    state + searcher-frame leaf value; ``advance`` fuses one step with the child's
    facts (one host sync per descent edge, child state kept on device); ``prior``
    is the one-step value sweep blended with greedy's tier table (the interior
    prior, replaced by a learned head when ``value`` is one)."""

    @jax.jit
    def facts(layout: BoardLayout, state: BoardState, player: IntScalar) -> _Facts:
        legal = flat_available_for(layout, state)
        term = _terminal(state)
        win = _winner(state) == player
        v = jnp.tanh(value(layout, state, player) / value_scale)
        leaf = jnp.where(term, jnp.where(win, 1.0, -1.0), v)
        return legal, agent_selection_single(state), term, leaf

    @jax.jit
    def advance(
        layout: BoardLayout, state: BoardState, action: IntScalar, player: IntScalar
    ) -> _Advance:
        atype, aparams = flat_to_action(action)
        avail = action_available(layout, state, atype, aparams)
        nxt, _ = apply_action(layout, state, atype, aparams, avail)
        return nxt, *facts(layout, nxt, player)

    @jax.jit
    def prior(
        layout: BoardLayout, state: BoardState, player: IntScalar, mask: FlatMask
    ) -> Float[Array, f"flat={N_FLAT}"]:
        # Soft prior logits: greedy's tier table (~2 nats apart by design) biased
        # by the squashed one-step value. The raw value sweep as logits is far too
        # peaked -- its softmax is ~one-hot, which starves PUCT of exploration.
        succ, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, ROW_TYPE, ROW_PARAMS, mask
        )
        vals = jax.vmap(value, in_axes=(None, 0, None))(layout, succ, player)
        return _TIER_LOGITS + jnp.tanh(vals / value_scale) / prior_scale

    return facts, advance, prior


def _softmax_legal(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Prior over the legal actions (0 elsewhere)."""
    masked = np.where(legal > 0, logits, -np.inf)
    masked -= masked.max()
    p = np.where(legal > 0, np.exp(masked), 0.0)
    return np.asarray(p / p.sum(), dtype=np.float64)


@dataclass
class _Node:
    mover: int
    terminal: bool
    prior: np.ndarray  # (N_FLAT,) over legal actions, 0 elsewhere
    children: dict[int, int] = field(default_factory=dict)
    visits: int = 0
    n: np.ndarray = field(default_factory=lambda: np.zeros(N_FLAT))  # edge visits
    w: np.ndarray = field(default_factory=lambda: np.zeros(N_FLAT))  # searcher-frame
    navail: np.ndarray = field(default_factory=lambda: np.zeros(N_FLAT))  # availability


def _select(node: _Node, legal: np.ndarray, player: int) -> int:
    """PUCT over the actions legal in this determinization. Prior renormalized
    over the legal set; ``sign`` makes opponents minimize the searcher value.
    Unvisited actions take first-play-urgency Q = the node's own mean value, so a
    visited action's backed-up value is no permanent head start over unexplored
    siblings (the fix for the search collapsing onto one action)."""
    sign = 1.0 if node.mover == player else -1.0
    node_v = node.w.sum() / max(node.visits, 1)  # searcher-frame node mean
    fpu = node_v - _FPU * sign  # unvisited: node mean, pessimistic in mover frame
    q = np.where(node.n > 0, node.w / np.maximum(node.n, 1), fpu)
    p = node.prior * legal
    psum = p.sum()
    p = p / psum if psum > 0 else legal / max(int(legal.sum()), 1)
    u = _C_PUCT * p * np.sqrt(node.visits + 1) / (1.0 + node.n)
    score = np.where(legal, sign * q + u, -np.inf)
    return int(np.argmax(score))


def _run(
    key: KeyScalar,
    layout: BoardLayout,
    view: BeliefView,
    player: int,
    root_mask: np.ndarray,
    *,
    num_simulations: int,
    max_depth: int,
    value: ValueFunction,
    value_scale: float,
    prior_scale: float,
) -> np.ndarray:
    """Grow one SO-ISMCTS tree; return the root edge-visit counts (the improved
    policy, the AlphaZero target)."""
    facts, advance, prior = _ops(value, value_scale, prior_scale)  # type: ignore[arg-type]
    pj = jnp.int32(player)
    keys = jax.random.split(key, num_simulations + 1)

    def facts_h(state: BoardState) -> tuple[np.ndarray, int, bool, float]:
        legal, mover, term, leaf = facts(layout, state, pj)
        return np.asarray(legal), int(mover), bool(term), float(leaf)

    def expand(mover: int, term: bool, state: BoardState, legal: np.ndarray) -> _Node:
        p = (
            np.zeros(N_FLAT)
            if term
            else _softmax_legal(
                np.asarray(prior(layout, state, pj, jnp.asarray(legal))), legal
            )
        )
        return _Node(mover=mover, terminal=term, prior=p)

    # Root from one sampled world (its prior only seeds the first expansions;
    # per-simulation determinization carries the rest).
    root_state = sample_world(keys[0], view, pj)
    r_legal, r_mover, r_term, _ = facts_h(root_state)
    nodes = [expand(r_mover, r_term, root_state, r_legal)]

    for sim in range(num_simulations):
        state = sample_world(keys[sim + 1], view, pj)
        legal, _, term, leaf = facts_h(state)  # root facts under this world
        path: list[tuple[int, int]] = []  # (node id, action)
        node_id = 0
        for _ in range(max_depth):
            node = nodes[node_id]
            if term or int(legal.sum()) == 0:
                break
            a = _select(node, legal, player)
            node.navail[legal > 0] += 1
            path.append((node_id, a))
            state, legal_d, mover_d, term_d, leaf_d = advance(
                layout, state, jnp.int32(a), pj
            )
            legal, leaf, term = np.asarray(legal_d), float(leaf_d), bool(term_d)
            if a in node.children:
                node_id = node.children[a]
                continue
            nodes.append(expand(int(mover_d), term, state, legal))  # leaf: expand once
            node.children[a] = len(nodes) - 1
            break

        for nid, a in path:  # backup the searcher-frame leaf value
            n = nodes[nid]
            n.visits += 1
            n.n[a] += 1
            n.w[a] += leaf

    return nodes[0].n.copy()


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
    root_mask = np.asarray(mask)
    counts = _run(
        key, layout, view, int(player), root_mask,
        num_simulations=num_simulations, max_depth=max_depth,
        value=value, value_scale=value_scale, prior_scale=prior_scale,
    )  # fmt: skip
    total = counts.sum()
    return counts / total if total > 0 else root_mask / max(int(root_mask.sum()), 1)


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
