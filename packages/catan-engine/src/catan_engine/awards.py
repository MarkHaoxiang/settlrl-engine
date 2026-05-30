"""Longest Road and Largest Army computation and award reassignment.

The longest-road length is the hard piece: it is the longest *trail* (no repeated
edge) in the player's road subgraph that may not pass *through* an opponent-owned
vertex (it may start or end there). It is implemented as an explicit-stack
iterative DFS via ``lax.while_loop`` so it stays fully traceable / vmappable.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.layout import EDGE_V, MAX_VERTEX_DEGREE, N_EDGES, N_VERTICES
from catan_engine.resources import N_PLAYERS
from catan_engine.state import NO_INDEX, BoardState

# int32 "unclaimed" award marker (the uint8 NO_INDEX as stored on BoardState).
# This is the award-holder sentinel, unrelated to any geometry padding.
_NONE = jnp.int32(NO_INDEX)

# Generous bound for the longest-road DFS stack. We seed 2 * N_EDGES frames
# (both directions of every edge) and the live DFS frontier adds only
# depth * (maxdeg - 1) ~= 15 * 2 on top; this leaves a wide margin.
STACK_CAP = 2 * N_EDGES + 128

# Sentinel-free CSR adjacency derived from the COO edge_index (EDGE_V), built
# once at import. Each undirected edge is listed under both endpoints; entries
# are grouped by source vertex so vertex v owns the slice
# [_ADJ_INDPTR[v], _ADJ_INDPTR[v + 1]). The DFS reads slots by degree mask
# (slot < deg), never a padding value.
_E = np.asarray(EDGE_V)
_src = np.concatenate([_E[:, 0], _E[:, 1]])
_order = np.argsort(_src, kind="stable")
_src = _src[_order]
_counts = np.bincount(_src, minlength=N_VERTICES)
_ADJ_INDPTR = jnp.asarray(np.concatenate([[0], np.cumsum(_counts)]), dtype=jnp.int32)
_ADJ_EDGE = jnp.asarray(
    np.concatenate([np.arange(N_EDGES), np.arange(N_EDGES)])[_order], dtype=jnp.int32
)
_ADJ_NBR = jnp.asarray(
    np.concatenate([_E[:, 1], _E[:, 0]])[_order], dtype=jnp.int32
)


def longest_road_length(
    edge_road: jax.Array, vertex_owner: jax.Array, player: jax.Array
) -> jax.Array:
    """Length of the player's longest continuous road (trail).

    A trail may not reuse an edge and may not pass *through* a vertex occupied
    by an opponent (it may start or end there). Implemented as an explicit-stack
    iterative DFS so it is fully traceable / vmappable.

    Seeding both directions of every edge (a length-1 frame having already
    traversed that edge, landing at the far vertex) and only *expanding* from
    passable vertices handles the opponent-as-endpoint rule and lets a trail
    pass through an empty interior vertex whose two endpoints are opponents.
    Non-owned edges are seeded as harmless length-0 frames.
    """
    target = player + 1
    mine = edge_road == target  # (N_EDGES,) bool

    # Seed frames: for each edge in each direction, land on the far vertex.
    onehot = jnp.eye(N_EDGES, dtype=jnp.bool_) & mine[:, None]  # empty if not mine
    seed_vertex = jnp.concatenate([EDGE_V[:, 1], EDGE_V[:, 0]])  # (2N,)
    seed_len = jnp.concatenate([mine, mine]).astype(jnp.int32)  # 1 if mine else 0
    seed_mask = jnp.concatenate([onehot, onehot], axis=0)  # (2N, N_EDGES)
    n_seed = 2 * N_EDGES

    stack_v = jnp.zeros((STACK_CAP,), jnp.int32).at[:n_seed].set(seed_vertex)
    stack_len = jnp.zeros((STACK_CAP,), jnp.int32).at[:n_seed].set(seed_len)
    stack_mask = jnp.zeros((STACK_CAP, N_EDGES), jnp.bool_).at[:n_seed].set(seed_mask)
    sp = jnp.int32(n_seed)
    best = jnp.int32(0)

    def cond(carry: tuple) -> jax.Array:
        return cast(jax.Array, carry[3] > 0)

    def body(carry: tuple) -> tuple:
        stack_v, stack_len, stack_mask, sp, best = carry
        sp = sp - 1
        v = stack_v[sp]
        m = stack_mask[sp]
        length = stack_len[sp]
        best = jnp.maximum(best, length)
        owner = vertex_owner[v]
        can = (owner == 0) | (owner == target)

        start = _ADJ_INDPTR[v]
        deg = _ADJ_INDPTR[v + 1] - start
        st = (stack_v, stack_len, stack_mask, sp)
        for slot in range(MAX_VERTEX_DEGREE):
            sv, sl, sm, sp_i = st
            idx = jnp.clip(start + slot, 0, 2 * N_EDGES - 1)
            e = _ADJ_EDGE[idx]
            w = _ADJ_NBR[idx]
            valid = (slot < deg) & can & mine[e] & ~m[e]
            new_mask = m.at[e].set(True)
            sv = sv.at[sp_i].set(jnp.where(valid, w, sv[sp_i]))
            sl = sl.at[sp_i].set(jnp.where(valid, length + 1, sl[sp_i]))
            sm = sm.at[sp_i].set(jnp.where(valid, new_mask, sm[sp_i]))
            sp_i = sp_i + valid.astype(jnp.int32)
            st = (sv, sl, sm, sp_i)

        stack_v, stack_len, stack_mask, sp = st
        return (stack_v, stack_len, stack_mask, sp, best)

    carry = jax.lax.while_loop(cond, body, (stack_v, stack_len, stack_mask, sp, best))
    return cast(jax.Array, carry[4])


def _reassign_award(counts: jax.Array, owner: jax.Array, threshold: int) -> jax.Array:
    """Award holder given per-player ``counts``: need >= threshold; ties to holder."""
    qualifies = counts >= threshold
    any_q = jnp.any(qualifies)
    top = jnp.max(jnp.where(qualifies, counts, -1))
    leaders = qualifies & (counts == top)
    holder_leads = jnp.where(
        owner == _NONE, False, leaders[jnp.clip(owner, 0, N_PLAYERS - 1)]
    )
    first_leader = jnp.argmax(leaders).astype(jnp.int32)
    new_owner = jnp.where(
        any_q, jnp.where(holder_leads, owner.astype(jnp.int32), first_leader), _NONE
    )
    return new_owner.astype(jnp.uint8)


def recompute_longest_road(state: BoardState) -> BoardState:
    """Reassign Longest Road (need >= 5; current holder wins ties)."""
    lengths = jnp.stack(
        [
            longest_road_length(state.edge_road, state.vertex_owner, jnp.int32(p))
            for p in range(N_PLAYERS)
        ]
    )
    new_owner = _reassign_award(lengths, state.longest_road_owner, 5)
    qualifies_any = jnp.any(lengths >= 5)
    new_len = jnp.where(
        qualifies_any, lengths[jnp.clip(new_owner, 0, N_PLAYERS - 1)], 0
    ).astype(jnp.uint8)
    return state._replace(longest_road_owner=new_owner, longest_road_len=new_len)


def recompute_largest_army(state: BoardState) -> BoardState:
    """Reassign Largest Army (need >= 3; current holder wins ties)."""
    new_owner = _reassign_award(
        state.knights_played.astype(jnp.int32), state.largest_army_owner, 3
    )
    return state._replace(largest_army_owner=new_owner)
