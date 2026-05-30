"""Trusted NumPy single-game rule reference (test oracle only).

This is the original NumPy single-game rule implementation, kept in the test
suite as the differential reference that ``test_rules_vec.py`` validates the
JAX-native ``catan_engine.rules_vec`` against. It is intentionally NOT part of
the shipped package -- the engine uses ``rules_vec`` exclusively.

Everything here operates on one game (batch index ``b``, default 0). The board
state stays a batched ``BoardState``; these helpers read the requested game via
NumPy, apply the rule with ordinary control flow, and write the result back with
``BoardState._replace``. They are deliberately *not* JAX-traceable -- correctness
and completeness first (see the "single-game" decision in CLAUDE-level notes).

Player convention follows state.py: players are 0-indexed; vertex_owner /
edge_road store ``player + 1`` with 0 meaning empty.
"""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.dev_cards import DevCard
from catan_engine.layout import (
    NO_INDEX,
    N_EDGES,
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
from catan_engine.resources import BANK_INITIAL, N_PLAYERS, N_RESOURCES
from catan_engine.state import BoardState, MAX_ROADS
from catan_engine.tile import Tile

# Static geometry as NumPy for fast indexing in the helpers below.
_EDGE_V = np.asarray(_edge_vertex_map)  # (N_EDGES, 2)
_V_EDGES = np.asarray(_vertex_edge_map)  # (N_VERTICES, MAX_DEGREE)
_V_NBR = np.asarray(_vertex_neighbour_map)  # (N_VERTICES, MAX_DEGREE)
_V_TILES = np.asarray(_vertex_tile_map)  # (N_VERTICES, MAX_DEGREE)
_V_PORT = np.asarray(_vertex_port_map)  # (N_VERTICES,)
_TILE_V = np.asarray(_tile_vertex_map)  # (N_TILES, 6)

# Setup placement order over 2 * N_PLAYERS settlements (snake / boustrophedon).
SETUP_ORDER: list[int] = list(range(N_PLAYERS)) + list(range(N_PLAYERS - 1, -1, -1))


# ---------------------------------------------------------------------------
# Counts and resources
# ---------------------------------------------------------------------------


def count_roads(state: BoardState, player: int, b: int = 0) -> int:
    return int(np.sum(np.asarray(state.edge_road[b]) == player + 1))


def count_settlements(state: BoardState, player: int, b: int = 0) -> int:
    owner = np.asarray(state.vertex_owner[b])
    kind = np.asarray(state.vertex_type[b])
    return int(np.sum((owner == player + 1) & (kind == 1)))


def count_cities(state: BoardState, player: int, b: int = 0) -> int:
    owner = np.asarray(state.vertex_owner[b])
    kind = np.asarray(state.vertex_type[b])
    return int(np.sum((owner == player + 1) & (kind == 2)))


def can_afford(
    state: BoardState, player: int, cost: tuple[int, ...], b: int = 0
) -> bool:
    res = np.asarray(state.player_resources[b, player])
    return all(int(res[i]) >= cost[i] for i in range(N_RESOURCES))


def _pay(state: BoardState, player: int, cost: tuple[int, ...], b: int) -> BoardState:
    res = np.asarray(state.player_resources[b]).copy()
    res[player] -= np.array(cost, dtype=res.dtype)
    pr = state.player_resources.at[b].set(jnp.asarray(res))
    return state._replace(player_resources=pr)


def bank_stock(state: BoardState, resource: int, b: int = 0) -> int:
    held = int(np.asarray(state.player_resources[b, :, resource]).sum())
    return BANK_INITIAL - held


# ---------------------------------------------------------------------------
# Victory points
# ---------------------------------------------------------------------------


def player_total_vp(state: BoardState, player: int, b: int = 0) -> int:
    """Total VP for ``player``: buildings + awards + hidden Victory Point cards."""
    total = int(state.victory_points[b, player])
    if int(state.longest_road_owner[b]) == player:
        total += 2
    if int(state.largest_army_owner[b]) == player:
        total += 2
    total += int(state.dev_hand[b, player, DevCard.VICTORY_POINT])
    return total


# ---------------------------------------------------------------------------
# Placement legality
# ---------------------------------------------------------------------------


def distance_rule_ok(state: BoardState, vertex: int, b: int = 0) -> bool:
    """Vertex is empty and no adjacent vertex carries a building."""
    owner = np.asarray(state.vertex_owner[b])
    if owner[vertex] != 0:
        return False
    for w in _V_NBR[vertex]:
        if w != NO_INDEX and owner[int(w)] != 0:
            return False
    return True


def settlement_connected(
    state: BoardState, player: int, vertex: int, b: int = 0
) -> bool:
    """Player owns a road incident to ``vertex`` (required outside setup)."""
    road = np.asarray(state.edge_road[b])
    for e in _V_EDGES[vertex]:
        if e != NO_INDEX and road[int(e)] == player + 1:
            return True
    return False


def road_placeable(state: BoardState, player: int, edge: int, b: int = 0) -> bool:
    """Edge is empty and connects to the player's network at a non-blocked end."""
    road = np.asarray(state.edge_road[b])
    owner = np.asarray(state.vertex_owner[b])
    if road[edge] != 0:
        return False
    target = player + 1
    for v in (int(_EDGE_V[edge, 0]), int(_EDGE_V[edge, 1])):
        if owner[v] == target:  # own building at this end
            return True
        if owner[v] != 0:  # opponent building blocks routing through this end
            continue
        for e2 in _V_EDGES[v]:
            if e2 != NO_INDEX and int(e2) != edge and road[int(e2)] == target:
                return True
    return False


# ---------------------------------------------------------------------------
# Longest road / largest army
# ---------------------------------------------------------------------------


def longest_road_length(state: BoardState, player: int, b: int = 0) -> int:
    """Length of the player's longest continuous road.

    A path may not reuse an edge and may not pass *through* a vertex occupied by
    an opponent (it may start or end there). The board is small (<= 15 roads),
    so an exhaustive DFS over each starting edge is fine.
    """
    road = np.asarray(state.edge_road[b])
    owner = np.asarray(state.vertex_owner[b])
    target = player + 1
    my_edges = [e for e in range(N_EDGES) if road[e] == target]
    if not my_edges:
        return 0

    incident: dict[int, list[int]] = defaultdict(list)
    for e in my_edges:
        incident[int(_EDGE_V[e, 0])].append(e)
        incident[int(_EDGE_V[e, 1])].append(e)

    best = 0

    def passable(vertex: int) -> bool:
        return bool(owner[vertex] == 0 or owner[vertex] == target)

    def dfs(vertex: int, used: set[int], length: int) -> None:
        nonlocal best
        best = max(best, length)
        if not passable(vertex):  # cannot continue through an opponent building
            return
        for e in incident[vertex]:
            if e in used:
                continue
            a, b2 = int(_EDGE_V[e, 0]), int(_EDGE_V[e, 1])
            nxt = b2 if a == vertex else a
            used.add(e)
            dfs(nxt, used, length + 1)
            used.discard(e)

    for e in my_edges:
        a, b2 = int(_EDGE_V[e, 0]), int(_EDGE_V[e, 1])
        for start, other in ((a, b2), (b2, a)):
            dfs(other, {e}, 1)
    return best


def recompute_longest_road(state: BoardState, b: int = 0) -> BoardState:
    """Reassign the Longest Road card (needs >= 5; current holder wins ties)."""
    lengths = [longest_road_length(state, p, b) for p in range(N_PLAYERS)]
    owner = int(state.longest_road_owner[b])
    qualifying = [p for p in range(N_PLAYERS) if lengths[p] >= 5]
    if not qualifying:
        new_owner, new_len = NO_INDEX, 0
    else:
        top = max(lengths[p] for p in qualifying)
        leaders = [p for p in qualifying if lengths[p] == top]
        new_owner = owner if owner in leaders else leaders[0]
        new_len = lengths[new_owner]
    return state._replace(
        longest_road_owner=state.longest_road_owner.at[b].set(new_owner),
        longest_road_len=state.longest_road_len.at[b].set(new_len),
    )


def recompute_largest_army(state: BoardState, b: int = 0) -> BoardState:
    """Reassign the Largest Army card (needs >= 3; current holder wins ties)."""
    knights = np.asarray(state.knights_played[b])
    owner = int(state.largest_army_owner[b])
    qualifying = [p for p in range(N_PLAYERS) if int(knights[p]) >= 3]
    if not qualifying:
        new_owner = NO_INDEX
    else:
        top = max(int(knights[p]) for p in qualifying)
        leaders = [p for p in qualifying if int(knights[p]) == top]
        new_owner = owner if owner in leaders else leaders[0]
    return state._replace(
        largest_army_owner=state.largest_army_owner.at[b].set(new_owner)
    )


# ---------------------------------------------------------------------------
# Ports / maritime trade
# ---------------------------------------------------------------------------


def port_ratio(
    state: BoardState, layout: BoardLayout, player: int, give: int, b: int = 0
) -> int:
    """Best maritime trade ratio the player can use to give away ``give``.

    4 by default; 3 with any general (3:1) port; 2 with the matching 2:1 port.
    """
    owner = np.asarray(state.vertex_owner[b])
    alloc = np.asarray(layout.port_allocation[b])
    ratio = 4
    for v in range(N_VERTICES):
        if owner[v] != player + 1 or _V_PORT[v] == NO_INDEX:
            continue
        ptype = int(alloc[int(_V_PORT[v])])
        if ptype == Port.GENERAL:
            ratio = min(ratio, 3)
        elif ptype == give:
            ratio = min(ratio, 2)
    return ratio


# ---------------------------------------------------------------------------
# Dice, production, theft
# ---------------------------------------------------------------------------


def roll_dice(state: BoardState, b: int = 0) -> tuple[BoardState, int]:
    key, k1, k2 = jax.random.split(state.key[b], 3)
    d1 = int(jax.random.randint(k1, (), 1, 7))
    d2 = int(jax.random.randint(k2, (), 1, 7))
    return state._replace(key=state.key.at[b].set(key)), d1 + d2


def distribute_resources(
    layout: BoardLayout, state: BoardState, roll: int, b: int = 0
) -> BoardState:
    """Pay out resources for ``roll`` to building owners, honouring the bank.

    If demand for a resource exceeds the bank and more than one player is owed
    it, no one receives that resource; if exactly one player is owed it, they
    receive whatever the bank has left (official rule).
    """
    tile_number = np.asarray(layout.tile_number[b])
    tile_resource = np.asarray(layout.tile_resource[b])
    owner = np.asarray(state.vertex_owner[b])
    kind = np.asarray(state.vertex_type[b])
    robber = int(state.robber[b])
    res = np.asarray(state.player_resources[b]).astype(np.int64)

    gains = np.zeros((N_PLAYERS, N_RESOURCES), dtype=np.int64)
    for tile in range(len(tile_number)):
        if tile == robber or int(tile_number[tile]) != roll:
            continue
        resource = int(tile_resource[tile])
        if resource == Tile.DESERT:
            continue
        for v in _TILE_V[tile]:
            o = int(owner[int(v)])
            if o == 0:
                continue
            gains[o - 1, resource] += 1 if kind[int(v)] == 1 else 2

    bank = BANK_INITIAL - res.sum(axis=0)
    for r in range(N_RESOURCES):
        demand = gains[:, r]
        if demand.sum() <= bank[r]:
            res[:, r] += demand
        elif np.count_nonzero(demand) == 1:
            p = int(np.argmax(demand))
            res[p, r] += min(int(demand[p]), int(bank[r]))
        # else: nobody receives this resource

    pr = state.player_resources.at[b].set(jnp.asarray(res.astype(np.uint8)))
    return state._replace(player_resources=pr)


def steal(state: BoardState, thief: int, victim: int, b: int = 0) -> BoardState:
    """Move one random resource card from ``victim`` to ``thief``."""
    res = np.asarray(state.player_resources[b]).astype(np.int64)
    hand = res[victim]
    total = int(hand.sum())
    if total == 0:
        return state
    key, sub = jax.random.split(state.key[b])
    probs = jnp.asarray(hand.astype(np.float64) / total)
    choice = int(jax.random.choice(sub, N_RESOURCES, p=probs))
    res[victim, choice] -= 1
    res[thief, choice] += 1
    return state._replace(
        player_resources=state.player_resources.at[b].set(
            jnp.asarray(res.astype(np.uint8))
        ),
        key=state.key.at[b].set(key),
    )


def robber_victims(state: BoardState, tile: int, current: int, b: int = 0) -> list[int]:
    """Players (other than ``current``) with a building on ``tile`` and cards."""
    owner = np.asarray(state.vertex_owner[b])
    res = np.asarray(state.player_resources[b])
    victims: set[int] = set()
    for v in _TILE_V[tile]:
        o = int(owner[int(v)])
        if o != 0 and o - 1 != current and int(res[o - 1].sum()) > 0:
            victims.add(o - 1)
    return sorted(victims)


def grant_setup_resources(
    layout: BoardLayout, state: BoardState, vertex: int, player: int, b: int = 0
) -> BoardState:
    """Grant one resource per (non-desert) tile adjacent to a 2nd settlement."""
    tile_resource = np.asarray(layout.tile_resource[b])
    res = np.asarray(state.player_resources[b]).astype(np.int64)
    for t in _V_TILES[vertex]:
        if t == NO_INDEX:
            continue
        resource = int(tile_resource[int(t)])
        if resource != Tile.DESERT and bank_stock(state, resource, b) > 0:
            res[player, resource] += 1
    pr = state.player_resources.at[b].set(jnp.asarray(res.astype(np.uint8)))
    return state._replace(player_resources=pr)


def roads_left(state: BoardState, player: int, b: int = 0) -> int:
    return MAX_ROADS - count_roads(state, player, b)
