"""Longest Road and Largest Army award reassignment and step resolution.

The longest-road *length* itself (the trail DFS) lives in
:mod:`catan_engine.mechanics.longest_road`; this module turns per-player
lengths/knights into award holders and resolves the win.
"""

from __future__ import annotations

from itertools import combinations
from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.board.layout import MAX_VERTEX_DEGREE, N_EDGES, N_VERTICES
from catan_engine.board.state import (
    NO_INDEX,
    VICTORY_POINTS_TO_WIN,
    BoardState,
    BoolScalar,
    IntScalar,
)
from catan_engine.mechanics.common import (
    GAME_COMPLETE,
    SUCCESS,
    player_total_vp,
)
from catan_engine.mechanics.longest_road import (
    ADJ_DEG,
    ADJ_EDGE,
    ADJ_INDPTR,
    longest_road_length,
)

# int32 "unclaimed" award marker (the uint8 NO_INDEX as stored on BoardState).
# This is the award-holder sentinel, unrelated to any geometry padding.
_NONE = jnp.int32(NO_INDEX)


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
        owner == _NONE, False, leaders[jnp.clip(owner, 0, counts.shape[0] - 1)]
    )
    first_leader = jnp.argmax(leaders).astype(jnp.int32)
    taken = jnp.where(
        holder_leads,
        owner.astype(jnp.int32),
        jnp.where(n_leaders == 1, first_leader, _NONE),
    )
    new_owner = jnp.where(any_q, taken, _NONE)
    return new_owner.astype(jnp.uint8)


def recompute_longest_road(
    state: BoardState, needed: BoolScalar | bool = True
) -> BoardState:
    """Reassign Longest Road (need >= 5; see ``_reassign_award`` for the tie rule).

    When ``needed`` is False the stored holder/length are kept unchanged (and the
    DFS is seeded empty -- see :func:`longest_road_length`); callers pass False
    for actions that cannot change any road length.
    """
    n = state.n_players
    lengths = jax.vmap(longest_road_length, in_axes=(None, None, 0, None))(
        state.edge_road, state.vertex_owner, jnp.arange(n, dtype=jnp.int32), needed
    )
    new_owner = _reassign_award(lengths, state.longest_road_owner, 5)
    has_owner = new_owner != jnp.uint8(NO_INDEX)
    new_len = jnp.where(
        has_owner, lengths[jnp.clip(new_owner.astype(jnp.int32), 0, n - 1)], 0
    ).astype(jnp.uint8)
    return state._replace(
        longest_road_owner=jnp.where(needed, new_owner, state.longest_road_owner),
        longest_road_len=jnp.where(needed, new_len, state.longest_road_len),
    )


def recompute_largest_army(state: BoardState) -> BoardState:
    """Reassign Largest Army (need >= 3; see ``_reassign_award`` for the tie rule)."""
    # Knights never decrease, so a holder is always among the leaders and the
    # "tie among non-holders" branch is unreachable here.
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
#
# The Longest Road DFS is further gated per lane (``longest_road_needed``): only
# a successful BuildRoad can extend a length and only a successful
# BuildSettlement can break one (setup placements stay below the 5-road
# threshold, and edges never disappear), so every other lane keeps its stored
# holder/length and contributes zero iterations to the vmapped while_loop. The
# build gates below tighten this further: a trail's length is bounded by the
# builder's road count, and a settlement severs a trail only where it passed
# *through* the vertex, which takes two same-opponent incident edges (one edge
# makes the vertex a trail endpoint, which an opponent building may legally be).
# ===========================================================================


def road_build_gate(state: BoardState, player: IntScalar) -> BoolScalar:
    """Whether a road just built by ``player`` could change the Longest Road.

    True once the builder owns at least 5 roads (the award threshold); below
    that no trail of theirs can qualify and nobody else's length moved.
    """
    target = (player + 1).astype(state.edge_road.dtype)
    return jnp.sum((state.edge_road == target).astype(jnp.int32)) >= 5


def settlement_break_gate(
    state: BoardState, vertex: IntScalar, player: IntScalar
) -> BoolScalar:
    """Whether a settlement just built on ``vertex`` by ``player`` could break
    an opponent's road: some single opponent owns >= 2 of the vertex's edges."""
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    start = ADJ_INDPTR[v]
    deg = ADJ_DEG[v]
    target = player.astype(jnp.int32) + 1
    owners = []
    for slot in range(MAX_VERTEX_DEGREE):
        idx = jnp.clip(start + slot, 0, 2 * N_EDGES - 1)
        o = state.edge_road[ADJ_EDGE[idx]].astype(jnp.int32)
        owners.append(jnp.where(slot < deg, o, 0))
    hit = jnp.bool_(False)
    for a, b in combinations(owners, 2):
        hit = hit | ((a == b) & (a != 0) & (a != target))
    return cast(jax.Array, hit)


def road_build_needed(state: BoardState, result: IntScalar) -> BoolScalar:
    """Longest Road recompute gate for a just-applied BuildRoad: the build
    succeeded and the builder (the current player) passes
    :func:`road_build_gate`."""
    player = state.current_player.astype(jnp.int32)
    return cast(jax.Array, (result == SUCCESS) & road_build_gate(state, player))


def settlement_break_needed(
    state: BoardState, vertex: IntScalar, result: IntScalar
) -> BoolScalar:
    """Longest Road recompute gate for a just-applied BuildSettlement on
    ``vertex``: the build succeeded and passes :func:`settlement_break_gate`
    for the builder (the current player)."""
    player = state.current_player.astype(jnp.int32)
    return cast(
        jax.Array, (result == SUCCESS) & settlement_break_gate(state, vertex, player)
    )


road_build_needed_b = jax.jit(jax.vmap(road_build_needed))
"""Batched (per-lane) :func:`road_build_needed`."""

settlement_break_needed_b = jax.jit(jax.vmap(settlement_break_needed))
"""Batched (per-lane) :func:`settlement_break_needed`."""


def recompute_awards(
    state: BoardState, longest_road_needed: BoolScalar | bool = True
) -> BoardState:
    """Recompute both award holders (Longest Road, then Largest Army)."""
    return recompute_largest_army(recompute_longest_road(state, longest_road_needed))


def current_player_won(state: BoardState) -> BoolScalar:
    """True if the state's current player has reached the win threshold.

    The rulebook (p.5) only lets a player win *during their own turn*: an
    opponent crowned with Longest Road by a settlement break may sit at 10+ VP
    while play continues. Checking the *post-step* current player implements
    this exactly -- END_TURN's rotation makes it the turn-start claim of an
    off-turn 10.
    """
    cur = state.current_player.astype(jnp.int32)
    return player_total_vp(state, cur) >= VICTORY_POINTS_TO_WIN


def resolve_step(
    state: BoardState,
    result: IntScalar,
    longest_road_needed: BoolScalar | bool = True,
) -> tuple[BoardState, IntScalar]:
    """Stage 2 of an action: recompute awards, then resolve the win.

    Recomputes the Longest Road / Largest Army holders for the post-core state
    and upgrades a ``SUCCESS`` result to ``GAME_COMPLETE`` when the move left
    the (post-step) current player at the win threshold (see
    :func:`current_player_won`; an ``INVALID`` move left the board unchanged,
    so the recompute is a no-op and the code is preserved).
    ``longest_road_needed`` gates the Longest Road recompute (see
    :func:`recompute_longest_road`); pass False for actions that cannot change
    any road length.
    """
    state = recompute_awards(state, longest_road_needed)
    won = current_player_won(state)
    upgraded = jnp.where((result == SUCCESS) & won, GAME_COMPLETE, result)
    return state, upgraded


resolve_step_b = jax.jit(jax.vmap(resolve_step))
"""Batched (per-lane) :func:`resolve_step` for the standalone ``*_step`` wrappers."""
