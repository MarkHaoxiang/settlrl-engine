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

# Frames popped (and expanded together) per while_loop iteration.
_POP_K = 32

# Proven peak occupancy plus one scratch slot (_DUMP, never popped); see
# docs/longest-road-stack-bound.html for the proof.
STACK_CAP = 2 * MAX_ROADS + min(2 * MAX_ROADS, _POP_K) + (MAX_ROADS - 3) * _POP_K + 1
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


# A frame packs one partial trail into a single int32: bits [0, MAX_ROADS)
# hold the used-edge set (bit owned_rank[e]), the bits above hold the tip
# vertex (the vertex the trail's tip stands on). The trail's length is the
# popcount of the used set, so it needs no field of its own.
_TIP_SHIFT = MAX_ROADS
_USED_MASK = (1 << MAX_ROADS) - 1

_StackVec = Int[Array, f"stack_cap={STACK_CAP}"]
"""The packed frames across the DFS stack's slots."""


class _DfsState(NamedTuple):
    """The DFS ``while_loop`` carry: the packed-frame stack, the stack pointer
    delimiting the live frames ``[0, sp)``, and the best length popped so
    far."""

    stack: _StackVec
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
    # position in the frame's used-edge mask (a player owns <= MAX_ROADS
    # edges). Non-owned edges get a clamped junk rank that is never read.
    owned_rank = jnp.clip(jnp.cumsum(mine_i) - 1, 0, MAX_ROADS - 1)

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
    tip = jnp.concatenate([v, u])
    edge_bit = jnp.tile(jnp.int32(1) << owned_rank, 2)

    init = _DfsState(
        stack=jnp.zeros((STACK_CAP,), jnp.int32)
        .at[slot]
        .set((tip << _TIP_SHIFT) | edge_bit),
        sp=jnp.sum(keep.astype(jnp.int32)),
        best=jnp.int32(0),
    )

    def cond(dfs: _DfsState) -> BoolScalar:
        return dfs.sp > 0

    def body(dfs: _DfsState) -> _DfsState:
        # Pop the top min(_POP_K, sp) frames as one block.
        pop_idx = dfs.sp - 1 - jnp.arange(_POP_K)
        live = pop_idx >= 0
        frame = dfs.stack[jnp.clip(pop_idx, 0)]
        tip = frame >> _TIP_SHIFT
        used = frame & _USED_MASK
        length = jnp.where(live, jax.lax.population_count(used), 0)
        best = jnp.maximum(dfs.best, jnp.max(length))
        base = dfs.sp - jnp.sum(live.astype(jnp.int32))

        # Expand all popped frames at once: a (_POP_K, MAX_VERTEX_DEGREE) grid
        # of candidate children.
        cols = jnp.arange(MAX_VERTEX_DEGREE)
        idx = jnp.clip(ADJ_INDPTR[tip][:, None] + cols, 0, 2 * N_EDGES - 1)
        e, nbr = ADJ_EDGE[idx], ADJ_NBR[idx]
        edge_bit = jnp.int32(1) << owned_rank[e]
        valid = (
            live[:, None]
            & (cols < ADJ_DEG[tip][:, None])
            & passable[tip][:, None]
            & mine[e]
            & ((used[:, None] & edge_bit) == 0)
        )
        child = (nbr << _TIP_SHIFT) | used[:, None] | edge_bit

        # Push the valid children contiguously above the new base; invalid
        # candidates scatter to _DUMP.
        valid_f = valid.ravel()
        valid_i = valid_f.astype(jnp.int32)
        offset = jnp.cumsum(valid_i) - valid_i
        slot = jnp.where(valid_f, base + offset, _DUMP)
        return _DfsState(
            stack=dfs.stack.at[slot].set(child.ravel()),
            sp=base + jnp.sum(valid_i),
            best=best,
        )

    final = jax.lax.while_loop(cond, body, init)
    return final.best
