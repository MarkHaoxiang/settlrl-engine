"""Traceable, vmappable single-game rule helpers.

Every function here operates on **one unbatched game**: ``vertex_owner`` has
shape ``(N_VERTICES,)``, ``player_resources`` has shape
``(N_PLAYERS, N_RESOURCES)``, scalars such as ``current_player`` are 0-d arrays,
and so on. Batch by wrapping the caller with ``jax.vmap`` -- ``BoardState`` /
``BoardLayout`` are NamedTuples (pytrees), so vmap maps over their leading batch
axis automatically.

These are the JAX-native replacements for ``catan_engine.rules`` (NumPy,
single-game on batch index 0). They are written without any Python control flow
on traced values: no ``int(...)``, ``np.asarray``, ``if`` on values, or
``.item()``. Index parameters are clamped with ``jnp.clip`` and ``NO_INDEX``
sentinels are handled by masking + clipped gathers.

Player convention follows state.py: players are 0-indexed; ``vertex_owner`` /
``edge_road`` store ``player + 1`` with 0 meaning empty.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.dev_cards import DEV_CARD_COST, N_DEV_CARD_TYPES, DevCard
from catan_engine.layout import (
    MAX_VERTEX_DEGREE,
    NO_INDEX,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    BoardLayout,
    _edge_vertex_map,
    _tile_vertex_map,
    _vertex_edge_map,
    _vertex_neighbour_map,
    _vertex_port_map,
    _vertex_tile_map,
)
from catan_engine.port import Port
from catan_engine.resources import (
    BANK_INITIAL,
    CITY_COST,
    N_PLAYERS,
    N_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
)
from catan_engine.state import MAX_ROADS, VICTORY_POINTS_TO_WIN, BoardState
from catan_engine.tile import Tile

# --- Static geometry as int32 jnp arrays for traceable gather/index ---------
EDGE_V = _edge_vertex_map.astype(jnp.int32)  # (N_EDGES, 2)
V_EDGES = _vertex_edge_map.astype(jnp.int32)  # (N_VERTICES, MAX_VERTEX_DEGREE)
V_NBR = _vertex_neighbour_map.astype(jnp.int32)  # (N_VERTICES, MAX_VERTEX_DEGREE)
V_TILES = _vertex_tile_map.astype(jnp.int32)  # (N_VERTICES, MAX_VERTEX_DEGREE)
V_PORT = _vertex_port_map.astype(jnp.int32)  # (N_VERTICES,)
TILE_V = _tile_vertex_map.astype(jnp.int32)  # (N_TILES, 6)

NO_IDX = jnp.int32(NO_INDEX)

# Build-cost vectors in resource order [sheep, wheat, wood, brick, ore].
ROAD_COST_ARR = jnp.array(ROAD_COST, dtype=jnp.int32)
SETTLEMENT_COST_ARR = jnp.array(SETTLEMENT_COST, dtype=jnp.int32)
CITY_COST_ARR = jnp.array(CITY_COST, dtype=jnp.int32)
DEV_CARD_COST_ARR = jnp.array(DEV_CARD_COST, dtype=jnp.int32)

# Generous bound for the longest-road DFS stack. We seed 2 * N_EDGES frames
# (both directions of every edge) and the live DFS frontier adds only
# depth * (maxdeg - 1) ~= 15 * 2 on top; this leaves a wide margin.
STACK_CAP = 2 * N_EDGES + 128

# Setup placement order over 2 * N_PLAYERS settlements (snake / boustrophedon),
# as a traceable int32 array so SetupRoad can advance the turn branchlessly.
SETUP_ORDER = list(range(N_PLAYERS)) + list(range(N_PLAYERS - 1, -1, -1))
SETUP_ORDER_ARR = jnp.array(SETUP_ORDER, dtype=jnp.int32)  # (2 * N_PLAYERS,)
N_SETUP = len(SETUP_ORDER)


def tree_select(mask: jax.Array, a: BoardState, b: BoardState) -> BoardState:
    """Per-leaf ``where(mask, a, b)`` over two single-game states (mask scalar)."""
    return cast(
        BoardState, jax.tree_util.tree_map(lambda x, y: jnp.where(mask, x, y), a, b)
    )


# ---------------------------------------------------------------------------
# Counts / economy
# ---------------------------------------------------------------------------


def count_roads(edge_road: jax.Array, player: jax.Array) -> jax.Array:
    return jnp.sum(edge_road == player + 1).astype(jnp.int32)


def count_settlements(
    vertex_owner: jax.Array, vertex_type: jax.Array, player: jax.Array
) -> jax.Array:
    return jnp.sum((vertex_owner == player + 1) & (vertex_type == 1)).astype(jnp.int32)


def count_cities(
    vertex_owner: jax.Array, vertex_type: jax.Array, player: jax.Array
) -> jax.Array:
    return jnp.sum((vertex_owner == player + 1) & (vertex_type == 2)).astype(jnp.int32)


def roads_left(edge_road: jax.Array, player: jax.Array) -> jax.Array:
    return MAX_ROADS - count_roads(edge_road, player)


def can_afford(resources_row: jax.Array, cost_arr: jax.Array) -> jax.Array:
    """True if a single player's resource row covers ``cost_arr``."""
    return jnp.all(resources_row.astype(jnp.int32) >= cost_arr)


def pay(
    player_resources: jax.Array, player: jax.Array, cost_arr: jax.Array
) -> jax.Array:
    """Subtract ``cost_arr`` from ``player``'s row (clipped at 0), returning uint8."""
    updated = player_resources.astype(jnp.int32).at[player].add(-cost_arr)
    return jnp.clip(updated, 0, 255).astype(jnp.uint8)


def bank_stock(player_resources: jax.Array, resource: jax.Array) -> jax.Array:
    held = player_resources[:, resource].astype(jnp.int32).sum()
    return BANK_INITIAL - held


def player_total_vp(state: BoardState, player: jax.Array) -> jax.Array:
    """Building VP + awards + hidden Victory Point cards for ``player``."""
    total = state.victory_points[player].astype(jnp.int32)
    total += jnp.where(state.longest_road_owner == player, 2, 0)
    total += jnp.where(state.largest_army_owner == player, 2, 0)
    total += state.dev_hand[player, DevCard.VICTORY_POINT].astype(jnp.int32)
    return total


# ---------------------------------------------------------------------------
# Placement legality
# ---------------------------------------------------------------------------


def distance_rule_ok(vertex_owner: jax.Array, vertex: jax.Array) -> jax.Array:
    """Vertex empty and no adjacent vertex carries a building."""
    nbr = V_NBR[vertex]  # (MAX_VERTEX_DEGREE,)
    valid = nbr != NO_IDX
    occ = vertex_owner[jnp.where(valid, nbr, 0)]
    blocked = jnp.any(valid & (occ != 0))
    return (vertex_owner[vertex] == 0) & ~blocked


def settlement_connected(
    edge_road: jax.Array, player: jax.Array, vertex: jax.Array
) -> jax.Array:
    """Player owns a road incident to ``vertex`` (required outside setup)."""
    e = V_EDGES[vertex]
    valid = e != NO_IDX
    roads = edge_road[jnp.where(valid, e, 0)]
    return jnp.any(valid & (roads == player + 1))


def road_placeable(
    edge_road: jax.Array, vertex_owner: jax.Array, player: jax.Array, edge: jax.Array
) -> jax.Array:
    """Edge empty and connects to the player's network at a non-blocked end."""
    target = player + 1
    empty = edge_road[edge] == 0

    def end_ok(v: jax.Array) -> jax.Array:
        own_here = vertex_owner[v] == target
        blocked = (vertex_owner[v] != 0) & ~own_here  # opponent building blocks
        e2 = V_EDGES[v]
        valid = (e2 != NO_IDX) & (e2 != edge)
        roads = edge_road[jnp.where(valid, e2, 0)]
        has_own_adj = jnp.any(valid & (roads == target))
        return own_here | (~blocked & has_own_adj)

    return empty & (end_ok(EDGE_V[edge, 0]) | end_ok(EDGE_V[edge, 1]))


# ---------------------------------------------------------------------------
# Longest road / largest army
# ---------------------------------------------------------------------------


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

        st = (stack_v, stack_len, stack_mask, sp)
        for slot in range(MAX_VERTEX_DEGREE):
            sv, sl, sm, sp_i = st
            e = V_EDGES[v, slot]
            w = V_NBR[v, slot]
            e_c = jnp.clip(e, 0, N_EDGES - 1)
            valid = (e != NO_IDX) & can & mine[e_c] & ~m[e_c]
            new_mask = m.at[e_c].set(True)
            sv = sv.at[sp_i].set(jnp.where(valid, w, sv[sp_i]))
            sl = sl.at[sp_i].set(jnp.where(valid, length + 1, sl[sp_i]))
            sm = sm.at[sp_i].set(jnp.where(valid, new_mask, sm[sp_i]))
            sp_i = sp_i + valid.astype(jnp.int32)
            st = (sv, sl, sm, sp_i)

        stack_v, stack_len, stack_mask, sp = st
        return (stack_v, stack_len, stack_mask, sp, best)

    carry = jax.lax.while_loop(
        cond, body, (stack_v, stack_len, stack_mask, sp, best)
    )
    return cast(jax.Array, carry[4])


def _reassign_award(
    counts: jax.Array, owner: jax.Array, threshold: int
) -> jax.Array:
    """Award holder given per-player ``counts``: need >= threshold; ties to holder."""
    qualifies = counts >= threshold
    any_q = jnp.any(qualifies)
    top = jnp.max(jnp.where(qualifies, counts, -1))
    leaders = qualifies & (counts == top)
    holder_leads = jnp.where(
        owner == NO_IDX, False, leaders[jnp.clip(owner, 0, N_PLAYERS - 1)]
    )
    first_leader = jnp.argmax(leaders).astype(jnp.int32)
    new_owner = jnp.where(
        any_q, jnp.where(holder_leads, owner.astype(jnp.int32), first_leader), NO_IDX
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


# ---------------------------------------------------------------------------
# Ports / maritime trade
# ---------------------------------------------------------------------------


def port_ratio(
    vertex_owner: jax.Array,
    port_allocation: jax.Array,
    player: jax.Array,
    give: jax.Array,
) -> jax.Array:
    """Best maritime ratio for giving ``give``: 4, or 3 (general), or 2 (match)."""
    owns = vertex_owner == player + 1
    is_port = V_PORT != NO_IDX
    ptype = port_allocation[jnp.where(is_port, V_PORT, 0)]
    my_port = owns & is_port
    general = jnp.any(my_port & (ptype == Port.GENERAL))
    match = jnp.any(my_port & (ptype == give))
    return jnp.where(match, 2, jnp.where(general, 3, 4)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Dice, production, theft
# ---------------------------------------------------------------------------


def roll_dice(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return (advanced key, two-dice sum 2..12)."""
    key, k1, k2 = jax.random.split(key, 3)
    d1 = jax.random.randint(k1, (), 1, 7)
    d2 = jax.random.randint(k2, (), 1, 7)
    return key, (d1 + d2).astype(jnp.int32)


def distribute_resources(
    layout: BoardLayout, state: BoardState, roll: jax.Array
) -> BoardState:
    """Pay out resources for ``roll`` to building owners, honouring the bank.

    Bank rule: if demand for a resource exceeds the bank and more than one
    player is owed it, no one receives it; if exactly one player is owed it,
    they receive whatever the bank has left.
    """
    owner = state.vertex_owner
    kind = state.vertex_type
    res = state.player_resources.astype(jnp.int32)  # (P, R)

    produces = (
        (layout.tile_number == roll)
        & (jnp.arange(N_TILES) != state.robber)
        & (layout.tile_resource != Tile.DESERT)
    )  # (N_TILES,)

    c_owner = owner[TILE_V]  # (N_TILES, 6)
    c_kind = kind[TILE_V]
    amt = jnp.where(c_kind == 1, 1, jnp.where(c_kind == 2, 2, 0)) * produces[:, None]
    amt = jnp.where(c_owner > 0, amt, 0).astype(jnp.int32)
    pl = jnp.clip(c_owner.astype(jnp.int32) - 1, 0, N_PLAYERS - 1)
    res_idx = jnp.broadcast_to(
        layout.tile_resource[:, None].astype(jnp.int32), (N_TILES, 6)
    )
    gains = jnp.zeros((N_PLAYERS, N_RESOURCES), jnp.int32).at[
        pl.reshape(-1), res_idx.reshape(-1)
    ].add(amt.reshape(-1))

    bank = BANK_INITIAL - res.sum(axis=0)  # (R,)
    total = gains.sum(axis=0)  # (R,)
    n_claim = (gains > 0).sum(axis=0)  # (R,)
    enough = total <= bank
    single = n_claim == 1
    granted = jnp.where(
        enough, gains, jnp.where(single, jnp.minimum(gains, bank), 0)
    )
    new_res = (res + granted).astype(jnp.uint8)
    return state._replace(player_resources=new_res)


def steal(state: BoardState, thief: jax.Array, victim: jax.Array) -> BoardState:
    """Move one random resource card from ``victim`` to ``thief`` (no-op if empty)."""
    res = state.player_resources.astype(jnp.int32)
    hand = res[victim]  # (R,)
    total = hand.sum()
    key, sub = jax.random.split(state.key)
    probs = jnp.where(
        total > 0,
        hand / jnp.maximum(total, 1),
        jnp.full((N_RESOURCES,), 1.0 / N_RESOURCES),
    )
    choice = jax.random.choice(sub, N_RESOURCES, p=probs)
    do = (total > 0).astype(jnp.int32)
    res = res.at[victim, choice].add(-do)
    res = res.at[thief, choice].add(do)
    return state._replace(
        player_resources=res.astype(jnp.uint8), key=key
    )


def robber_victim_mask(
    state: BoardState, tile: jax.Array, current: jax.Array
) -> jax.Array:
    """(N_PLAYERS,) bool: players != current with a building on ``tile`` and cards."""
    o = state.vertex_owner[TILE_V[tile]]  # (6,)
    pl = jnp.clip(o.astype(jnp.int32) - 1, 0, N_PLAYERS - 1)
    present = jnp.zeros((N_PLAYERS,), jnp.bool_).at[pl].max(o > 0)
    has_cards = state.player_resources.astype(jnp.int32).sum(axis=1) > 0
    return present & has_cards & (jnp.arange(N_PLAYERS) != current)


def grant_setup_resources(
    layout: BoardLayout, state: BoardState, vertex: jax.Array, player: jax.Array
) -> BoardState:
    """Grant one resource per (non-desert) tile adjacent to a 2nd settlement."""
    res = state.player_resources.astype(jnp.int32)
    bank = BANK_INITIAL - res.sum(axis=0)  # (R,)
    tiles = V_TILES[vertex]  # (MAX_VERTEX_DEGREE,)
    for i in range(MAX_VERTEX_DEGREE):
        t = tiles[i]
        t_c = jnp.clip(t, 0, N_TILES - 1)
        resource = layout.tile_resource[t_c].astype(jnp.int32)
        ok = (t != NO_IDX) & (resource != Tile.DESERT) & (bank[resource] > 0)
        add = ok.astype(jnp.int32)
        res = res.at[player, resource].add(add)
        bank = bank.at[resource].add(-add)
    return state._replace(player_resources=res.astype(jnp.uint8))


# ---------------------------------------------------------------------------
# Development cards
# ---------------------------------------------------------------------------


def playable_dev(state: BoardState, player: jax.Array, card: int) -> jax.Array:
    """True if ``player`` holds a playable copy of ``card`` (not bought this turn)."""
    held = state.dev_hand[player, card].astype(jnp.int32)
    bought = state.dev_bought[card].astype(jnp.int32)
    return held - bought > 0


def draw_dev_card(key: jax.Array, dev_deck: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Draw one card type from ``dev_deck`` weighted by remaining counts.

    Returns ``(advanced key, card index)``. The probabilities fall back to
    uniform when the deck is empty so the draw is always well defined under a
    trace; callers gate the actual application on deck availability.
    """
    deck = dev_deck.astype(jnp.float32)
    total = deck.sum()
    probs = jnp.where(
        total > 0,
        deck / jnp.maximum(total, 1.0),
        jnp.full((N_DEV_CARD_TYPES,), 1.0 / N_DEV_CARD_TYPES),
    )
    key, sub = jax.random.split(key)
    card = jax.random.choice(sub, N_DEV_CARD_TYPES, p=probs)
    return key, card.astype(jnp.int32)


# Re-export win threshold for callers building action transitions.
__all__ = [
    "VICTORY_POINTS_TO_WIN",
]
