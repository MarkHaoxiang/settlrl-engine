"""Longest Road and Largest Army computation and award reassignment.

The longest-road length is the hard piece: it is the longest *trail* (no repeated
edge) in the player's road subgraph that may not pass *through* an opponent-owned
vertex (it may start or end there). It is implemented as an explicit-stack
iterative DFS via ``lax.while_loop`` so it stays fully traceable / vmappable;
each frame carries its used-edge set as a packed int32 bitmask.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.board.layout import EDGE_V, MAX_VERTEX_DEGREE, N_EDGES, N_VERTICES
from catan_engine.board.resources import N_PLAYERS
from catan_engine.board.state import (
    NO_INDEX,
    VICTORY_POINTS_TO_WIN,
    BoardState,
    BoolScalar,
    EdgeRoadVec,
    IntScalar,
    VertexOwnerVec,
)
from catan_engine.mechanics.common import (
    GAME_COMPLETE,
    SUCCESS,
    player_total_vp,
)

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
    edge_road: EdgeRoadVec, vertex_owner: VertexOwnerVec, player: IntScalar
) -> IntScalar:
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

    # The used-edge set rides each frame as a packed int32 bitmask. A player owns
    # at most MAX_ROADS (=15) edges, so a running count over `mine` maps the owned
    # edges to bit positions 0.. -- a single int32 word always suffices (15 < 32).
    # Non-owned edges get a clamped junk index that the `mine[e]` guard never reads.
    local_of = jnp.clip(jnp.cumsum(mine.astype(jnp.int32)) - 1, 0, 31)  # (N_EDGES,)

    # Seed frames: for each edge in each direction, land on the far vertex with
    # only that edge's bit set (a length-1 trail); non-owned edges seed mask 0.
    seed_word = jnp.where(mine, jnp.int32(1) << local_of, jnp.int32(0))  # (N_EDGES,)
    seed_vertex = jnp.concatenate([EDGE_V[:, 1], EDGE_V[:, 0]])  # (2N,)
    seed_len = jnp.concatenate([mine, mine]).astype(jnp.int32)  # 1 if mine else 0
    seed_mask = jnp.concatenate([seed_word, seed_word])  # (2N,) int32
    n_seed = 2 * N_EDGES

    stack_v = jnp.zeros((STACK_CAP,), jnp.int32).at[:n_seed].set(seed_vertex)
    stack_len = jnp.zeros((STACK_CAP,), jnp.int32).at[:n_seed].set(seed_len)
    stack_mask = jnp.zeros((STACK_CAP,), jnp.int32).at[:n_seed].set(seed_mask)
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
            bit = jnp.int32(1) << local_of[e]
            valid = (slot < deg) & can & mine[e] & ((m & bit) == 0)
            new_mask = m | bit
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
    """Award holder given per-player ``counts`` (need >= ``threshold``).

    Follows the rulebook tie rule (Almanac, "Longest Road", p.9): the current
    holder keeps the card while still tied for the lead; if it is beaten, a
    *single* new leader takes it, but a 2+ way tie among non-holders leaves the
    card unheld (``NO_INDEX``). With no qualifier the card is unheld.
    """
    qualifies = counts >= threshold
    any_q = jnp.any(qualifies)
    top = jnp.max(jnp.where(qualifies, counts, -1))
    leaders = qualifies & (counts == top)
    n_leaders = jnp.sum(leaders.astype(jnp.int32))
    holder_leads = jnp.where(
        owner == _NONE, False, leaders[jnp.clip(owner, 0, N_PLAYERS - 1)]
    )
    first_leader = jnp.argmax(leaders).astype(jnp.int32)
    taken = jnp.where(
        holder_leads,
        owner.astype(jnp.int32),
        jnp.where(n_leaders == 1, first_leader, _NONE),
    )
    new_owner = jnp.where(any_q, taken, _NONE)
    return new_owner.astype(jnp.uint8)


def recompute_longest_road(state: BoardState) -> BoardState:
    """Reassign Longest Road (need >= 5; see ``_reassign_award`` for the tie rule)."""
    lengths = jnp.stack(
        [
            longest_road_length(state.edge_road, state.vertex_owner, jnp.int32(p))
            for p in range(N_PLAYERS)
        ]
    )
    new_owner = _reassign_award(lengths, state.longest_road_owner, 5)
    has_owner = new_owner != jnp.uint8(NO_INDEX)
    new_len = jnp.where(
        has_owner, lengths[jnp.clip(new_owner.astype(jnp.int32), 0, N_PLAYERS - 1)], 0
    ).astype(jnp.uint8)
    return state._replace(longest_road_owner=new_owner, longest_road_len=new_len)


def recompute_largest_army(state: BoardState) -> BoardState:
    """Reassign Largest Army (need >= 3; see ``_reassign_award`` for the tie rule).

    In real play knights never decrease, so a holder is always among the leaders
    and the "tie among non-holders" branch is unreachable here.
    """
    new_owner = _reassign_award(
        state.knights_played.astype(jnp.int32), state.largest_army_owner, 3
    )
    return state._replace(largest_army_owner=new_owner)


# ===========================================================================
# Step resolution (stage 2 of an action)
#
# The award reassignment + win check are factored out of the per-action cores so
# the expensive Longest Road sweep runs *once* per step rather than once per
# ``jax.lax.switch`` branch -- every branch executes under ``vmap``, so leaving
# ``recompute_longest_road`` inside the BuildRoad / BuildSettlement branches paid
# the DFS for both on every action. ``apply_action`` calls this once after the
# switch; the standalone ``*_step`` wrappers whose action can change an award or
# win the game route their core output through it.
# ===========================================================================


def recompute_awards(state: BoardState) -> BoardState:
    """Recompute both award holders (Longest Road, then Largest Army)."""
    return recompute_largest_army(recompute_longest_road(state))


def _any_player_won(state: BoardState) -> BoolScalar:
    """True if any player's total VP has reached the win threshold."""
    vps = jnp.stack(
        [player_total_vp(state, jnp.int32(p)) for p in range(N_PLAYERS)]
    )
    return jnp.any(vps >= VICTORY_POINTS_TO_WIN)


def resolve_step(
    state: BoardState, result: IntScalar
) -> tuple[BoardState, IntScalar]:
    """Stage 2 of an action: recompute awards, then resolve the win.

    Recomputes the Longest Road / Largest Army holders for the post-core state
    and upgrades a ``SUCCESS`` result to ``GAME_COMPLETE`` when the move brought a
    player to the win threshold (an ``INVALID`` move left the board unchanged, so
    the recompute is a no-op and the code is preserved).
    """
    state = recompute_awards(state)
    won = _any_player_won(state)
    upgraded = jnp.where((result == SUCCESS) & won, GAME_COMPLETE, result)
    return state, upgraded


resolve_step_b = jax.jit(jax.vmap(resolve_step))
"""Batched (per-lane) :func:`resolve_step` for the standalone ``*_step`` wrappers."""
