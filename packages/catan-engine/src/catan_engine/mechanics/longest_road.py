"""The longest-road length: an explicit-stack DFS over road trails.

The length is the longest *trail* (no repeated edge; vertices may repeat) in
the player's road subgraph that may not pass *through* an opponent-owned
vertex (it may start or end there).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from catan_engine.board.layout import EDGE_V, MAX_VERTEX_DEGREE, N_EDGES, N_VERTICES
from catan_engine.board.state import (
    MAX_ROADS,
    BoolScalar,
    EdgeRoadVec,
    IntScalar,
    VertexOwnerVec,
)

# Tight bound for the DFS stack, assuming the rule invariant n_owned <=
# MAX_ROADS (= 15). Seeding puts at most 2 * n_owned <= 30 frames on the stack.
# Each pop pushes at most deg - 1 <= 2 children (the arrival edge is always in
# the frame's used set), so an expansion grows the stack by at most +1 net; a
# trail's length is capped at n_owned, so at most MAX_ROADS - 1 = 14 such
# expansions can be live along the current DFS path. Peak sp is therefore
# 2 * MAX_ROADS + (MAX_ROADS - 1) = 44, plus one scratch slot (_DUMP) above
# any live frame, where dropped seeds park (never popped).
STACK_CAP = 2 * MAX_ROADS + (MAX_ROADS - 1) + 1
_DUMP = STACK_CAP - 1

# Sentinel-free CSR adjacency derived from the COO edge_index (EDGE_V), built
# once at import. Each undirected edge is listed under both endpoints; entries
# are grouped by source vertex so vertex v owns the slice
# [ADJ_INDPTR[v], ADJ_INDPTR[v] + ADJ_DEG[v]). Readers mask slots by degree
# (slot < deg), never a padding value.
_E = np.asarray(EDGE_V)
_src = np.concatenate([_E[:, 0], _E[:, 1]])
_order = np.argsort(_src, kind="stable")
_src = _src[_order]
_counts = np.bincount(_src, minlength=N_VERTICES)
ADJ_INDPTR = jnp.asarray(np.concatenate([[0], np.cumsum(_counts)]), dtype=jnp.int32)
ADJ_EDGE = jnp.asarray(
    np.concatenate([np.arange(N_EDGES), np.arange(N_EDGES)])[_order], dtype=jnp.int32
)
ADJ_NBR = jnp.asarray(np.concatenate([_E[:, 1], _E[:, 0]])[_order], dtype=jnp.int32)
ADJ_DEG = jnp.asarray(_counts, dtype=jnp.int32)


_StackVec = Int[Array, f"stack_cap={STACK_CAP}"]
"""One frame field across the DFS stack's slots."""


class _DfsState(NamedTuple):
    """The DFS ``while_loop`` carry: the frame stack (a frame is one partial
    trail -- tip vertex, length, used-edge set), the stack pointer delimiting
    the live frames ``[0, sp)``, and the best length popped so far."""

    stack_tip: _StackVec  # vertex the trail's tip stands on
    stack_len: _StackVec  # road pieces used so far
    stack_used: _StackVec  # used-edge set, bit owned_rank[e]
    sp: IntScalar
    best: IntScalar


def longest_road_length(
    edge_road: EdgeRoadVec,
    vertex_owner: VertexOwnerVec,
    player: IntScalar,
    needed: BoolScalar | bool = True,
) -> IntScalar:
    """Length of the player's longest continuous road (trail).

    A trail may not reuse an edge and may not pass *through* a vertex occupied
    by an opponent (it may start or end there). Fully traceable / vmappable.
    When ``needed`` is False the result is 0 and the DFS is seeded empty, so
    under ``vmap`` a masked-off lane adds no loop iterations.
    """
    owner_code = player + 1  # edge_road / vertex_owner store player + 1, 0 = empty
    mine = (edge_road == owner_code) & needed
    mine_i = mine.astype(jnp.int32)
    passable = (vertex_owner == 0) | (vertex_owner == owner_code)
    u, v = EDGE_V[:, 0], EDGE_V[:, 1]

    # owned_rank[e]: rank of edge e among the player's owned edges -- its bit
    # position in the int32 used-edge mask (a player owns <= MAX_ROADS = 15 < 32
    # edges). Non-owned edges get a clamped junk rank that is never read.
    owned_rank = jnp.clip(jnp.cumsum(mine_i) - 1, 0, 31)

    # Endpoint-only seeding: a maximal trail can only terminate at a vertex of
    # odd owned-degree (a pass-through consumes edges in pairs) or at an
    # impassable vertex -- unless it is closed. Seed direction u -> v only when
    # u is such a start; edges with neither endpoint a start keep their forward
    # seed, covering closed trails (rotatable to begin anywhere, either way).
    owned_deg = jnp.zeros((N_VERTICES,), jnp.int32).at[u].add(mine_i).at[v].add(mine_i)
    is_start = (owned_deg % 2 == 1) | ~passable
    fwd_keep = mine & is_start[u]
    bwd_keep = mine & is_start[v]
    fwd_keep = fwd_keep | (mine & ~fwd_keep & ~bwd_keep)  # closed-trail fallback

    # Kept seeds compact to the front of the stack; dropped directions scatter
    # to _DUMP. Each seed is a length-1 trail with only its own edge's bit set;
    # the two directions differ only in the tip (the walked-to endpoint).
    keep = jnp.concatenate([fwd_keep, bwd_keep])
    slot = jnp.where(keep, jnp.cumsum(keep.astype(jnp.int32)) - 1, _DUMP)

    def seed_stack(val: jax.Array) -> jax.Array:
        """A stack array with per-direction ``val`` scattered to the seed slots."""
        return jnp.zeros((STACK_CAP,), jnp.int32).at[slot].set(val)

    init = _DfsState(
        stack_tip=seed_stack(jnp.concatenate([v, u])),
        stack_len=seed_stack(jnp.ones((2 * N_EDGES,), jnp.int32)),
        stack_used=seed_stack(jnp.tile(jnp.int32(1) << owned_rank, 2)),
        sp=jnp.sum(keep.astype(jnp.int32)),
        best=jnp.int32(0),
    )

    def cond(dfs: _DfsState) -> BoolScalar:
        return dfs.sp > 0

    def body(dfs: _DfsState) -> _DfsState:
        stack_tip, stack_len, stack_used = dfs.stack_tip, dfs.stack_len, dfs.stack_used
        sp = dfs.sp - 1
        tip = stack_tip[sp]
        used = stack_used[sp]
        length = stack_len[sp]
        best = jnp.maximum(dfs.best, length)

        start = ADJ_INDPTR[tip]
        deg = ADJ_DEG[tip]
        for slot in range(MAX_VERTEX_DEGREE):
            idx = jnp.clip(start + slot, 0, 2 * N_EDGES - 1)
            e, nbr = ADJ_EDGE[idx], ADJ_NBR[idx]
            edge_bit = jnp.int32(1) << owned_rank[e]
            valid = (slot < deg) & passable[tip] & mine[e] & ((used & edge_bit) == 0)

            stack_tip = stack_tip.at[sp].set(jnp.where(valid, nbr, stack_tip[sp]))
            stack_len = stack_len.at[sp].set(
                jnp.where(valid, length + 1, stack_len[sp])
            )
            stack_used = stack_used.at[sp].set(
                jnp.where(valid, used | edge_bit, stack_used[sp])
            )
            sp = sp + valid.astype(jnp.int32)

        return _DfsState(stack_tip, stack_len, stack_used, sp, best)

    final = jax.lax.while_loop(cond, body, init)
    return final.best
