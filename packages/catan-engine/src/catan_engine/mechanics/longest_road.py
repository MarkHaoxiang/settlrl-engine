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
    BoolScalar,
    EdgeRoadVec,
    IntScalar,
    VertexOwnerVec,
)

# Generous bound for the DFS stack. We seed at most 2 * MAX_ROADS (=30)
# owned-edge frames and the live DFS frontier adds only depth * (maxdeg-1)
# ~= 15 * 2 on top; this leaves a wide margin. _DUMP is a scratch slot above any
# live frame where non-owned edges park their (never-popped) seeds.
STACK_CAP = 2 * N_EDGES + 128
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
ADJ_DEG = jnp.asarray(_counts, dtype=jnp.int32)  # per-vertex degree (slice width)


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
    mine = (edge_road == owner_code) & needed  # (N_EDGES,) bool
    passable = (vertex_owner == 0) | (vertex_owner == owner_code)  # (N_VERTICES,)

    # owned_rank[e]: rank of edge e among the player's owned edges -- the k-th
    # owned edge, in edge-id order, has rank k-1. A rank doubles as the edge's
    # bit position in the int32 used-edge mask (a player owns <= MAX_ROADS = 15
    # < 32 edges) and as its seed slot. Non-owned edges get a clamped junk rank
    # the `mine[e]` guard never reads, and seed into a throwaway slot that is
    # never popped.
    owned_rank = jnp.clip(jnp.cumsum(mine.astype(jnp.int32)) - 1, 0, 31)  # (N_EDGES,)
    n_owned = jnp.sum(mine.astype(jnp.int32))

    # Seed only owned edges, compacted to the front: forward seeds fill slots
    # [0, n_owned), backward seeds fill [n_owned, 2 * n_owned); each starts a
    # length-1 trail with only its own bit set. Non-owned edges scatter to slot
    # _DUMP (> any live frame, never popped).
    fwd_slot = jnp.where(mine, owned_rank, _DUMP)
    bwd_slot = jnp.where(mine, n_owned + owned_rank, _DUMP)

    def seed_stack(fwd_val: jax.Array, bwd_val: jax.Array) -> jax.Array:
        """A stack array seeded per edge: ``fwd_val[e]`` at ``fwd_slot[e]``,
        ``bwd_val[e]`` at ``bwd_slot[e]``."""
        zeros = jnp.zeros((STACK_CAP,), jnp.int32)
        return zeros.at[fwd_slot].set(fwd_val).at[bwd_slot].set(bwd_val)

    # The two directions of an edge differ only in the tip (which endpoint was
    # walked to).
    seed_len = jnp.where(mine, jnp.int32(1), jnp.int32(0))
    seed_used = jnp.where(mine, jnp.int32(1) << owned_rank, jnp.int32(0))
    init = _DfsState(
        stack_tip=seed_stack(EDGE_V[:, 1], EDGE_V[:, 0]),
        stack_len=seed_stack(seed_len, seed_len),
        stack_used=seed_stack(seed_used, seed_used),
        sp=jnp.int32(2) * n_owned,
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
