"""Plain-numpy point of view: one observation unpacked for host-side logic.

``Pov`` unpacks one seat's host-fetched observation (see
:data:`~catan_agents.shared.policy.HostObservation`) plus its legality mask,
and the module-level tables
restate the static board graph (vertex adjacency, incident tiles, the flat
action table) in host-side form. Everything here is read-only convenience for
the decision tree; no game rule is re-implemented beyond what scoring needs —
legality always comes from the mask.
"""

from __future__ import annotations

import functools

import numpy as np
from catan_engine.board.dev_cards import DEV_CARD_COST, DevCard
from catan_engine.board.layout import (
    EDGE_V,
    N_VERTICES,
    PORT_V,
    TILE_V,
)
from catan_engine.board.port import Port
from catan_engine.board.resources import (
    CITY_COST,
    N_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
)
from catan_engine.board.state import CITY, GamePhase
from catan_engine.env import ActionType

from catan_agents.shared.policy import HostFlatMask, HostObservation
from catan_agents.shared.rows import (
    ROW_IDX as ROW_IDX,
)
from catan_agents.shared.rows import (
    ROW_TARGET as ROW_TARGET,
)
from catan_agents.shared.rows import (
    ROWS_OF_TYPE as ROWS_OF_TYPE,
)
from catan_agents.shared.rows import (
    flat_row as flat_row,
)

# -- Static board graph, host-side ------------------------------------------

EDGE_ENDPOINTS: np.ndarray = np.asarray(EDGE_V)  # (N_EDGES, 2)
TILE_CORNERS: np.ndarray = np.asarray(TILE_V)  # (N_TILES, 6)
_PORTS: np.ndarray = np.asarray(PORT_V)  # (N_PORTS, 2)

VERTEX_EDGES: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(e) for e in np.flatnonzero(np.equal(EDGE_ENDPOINTS, v).any(axis=1)))
    for v in range(N_VERTICES)
)
"""Edges incident to each vertex."""

VERTEX_NEIGHBORS: tuple[tuple[int, ...], ...] = tuple(
    tuple(
        int(EDGE_ENDPOINTS[e, 1] if EDGE_ENDPOINTS[e, 0] == v else EDGE_ENDPOINTS[e, 0])
        for e in VERTEX_EDGES[v]
    )
    for v in range(N_VERTICES)
)
"""Vertices one edge away from each vertex."""

VERTEX_TILES: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(t) for t in np.flatnonzero(np.equal(TILE_CORNERS, v).any(axis=1)))
    for v in range(N_VERTICES)
)
"""Tiles cornered by each vertex."""

# The flat-action table's host decode (ROW_IDX / ROW_TARGET / ROWS_OF_TYPE /
# flat_row) is re-exported above from ``catan_agents.shared.rows``.

# Build costs in resource order [sheep, wheat, wood, brick, ore].
COST_ROAD: np.ndarray = np.asarray(ROAD_COST, dtype=np.int64)
COST_SETTLEMENT: np.ndarray = np.asarray(SETTLEMENT_COST, dtype=np.int64)
COST_CITY: np.ndarray = np.asarray(CITY_COST, dtype=np.int64)
COST_DEV: np.ndarray = np.asarray(DEV_CARD_COST, dtype=np.int64)


def _pips(tile_number: np.ndarray) -> np.ndarray:
    """Pip count per tile (chances in 36 of its number; 0 for the desert)."""
    return np.asarray(
        np.clip(6 - np.abs(7 - tile_number.astype(np.int64)), 0, None)
        * (tile_number > 0)
    )


class Pov:
    """One seat's view of one game, as numpy, with the derived sets the
    decision tree keys on (own pieces, production, legal rows by type)."""

    def __init__(self, obs: HostObservation, mask: HostFlatMask) -> None:
        o = obs
        # Shapes follow the Observation contract (mask: (N_FLAT,) bool;
        # hand/bank/trade_*: (N_RESOURCES,)).
        self.mask: np.ndarray = mask
        self.me: int = int(o["self"])
        self.hand: np.ndarray = o["self_resources"].astype(np.int64)
        self.dev_hand: np.ndarray = o["self_dev_hand"]
        self.tile_resource: np.ndarray = o["tile_resource"]
        self.tile_number: np.ndarray = o["tile_number"]
        self.port_allocation: np.ndarray = o["port_allocation"]
        self.vertex_owner: np.ndarray = o["vertex_owner"]
        self.vertex_type: np.ndarray = o["vertex_type"]
        self.edge_road: np.ndarray = o["edge_road"]
        self.robber: int = int(o["robber"])
        self.victory_points: np.ndarray = o["victory_points"].astype(np.int64)
        self.knights_played: np.ndarray = o["knights_played"].astype(np.int64)
        self.longest_road_owner: int = int(o["longest_road_owner"])
        self.largest_army_owner: int = int(o["largest_army_owner"])
        self.longest_road_len: int = int(o["longest_road_len"])
        self.hand_size: np.ndarray = o["hand_size"].astype(np.int64)
        self.dev_card_count: np.ndarray = o["dev_card_count"].astype(np.int64)
        self.phase: int = int(o["phase"])
        self.current_player: int = int(o["current_player"])
        self.dice_roll: int = int(o["dice_roll"])
        self.has_rolled: bool = bool(o["has_rolled"])
        self.trade_partner: int = int(o["trade_partner"])
        self.pending_discard: int = int(o["self_pending_discard"])
        self.trade_give: np.ndarray = o["trade_give"].astype(np.int64)
        self.trade_receive: np.ndarray = o["trade_receive"].astype(np.int64)
        self.bank: np.ndarray = o["bank"].astype(np.int64)
        self.n_players: int = int(o["victory_points"].shape[0])

    # -- Legality ------------------------------------------------------------

    def legal_rows(self, atype: ActionType) -> np.ndarray:
        """The legal flat rows of one action type (table order)."""
        rows = ROWS_OF_TYPE[int(atype)]
        return np.asarray(rows[self.mask[rows]])

    def legal(self, atype: ActionType, idx: int = 0, target: int = 0) -> int | None:
        """The move's flat row if it is legal right now, else None."""
        row = flat_row(atype, idx, target)
        return row if self.mask[row] else None

    @property
    def my_turn_main(self) -> bool:
        """Acting in our own MAIN phase with the dice rolled (the build/trade
        window — the only window the plan executor runs in)."""
        return (
            self.phase == GamePhase.MAIN
            and self.has_rolled
            and self.current_player == self.me
        )

    # -- Own pieces and production --------------------------------------------

    @functools.cached_property
    def my_vertices(self) -> np.ndarray:
        return np.flatnonzero(self.vertex_owner == self.me + 1)

    @functools.cached_property
    def my_settlements(self) -> np.ndarray:
        return np.flatnonzero(
            (self.vertex_owner == self.me + 1) & (self.vertex_type != CITY)
        )

    @functools.cached_property
    def my_edges(self) -> np.ndarray:
        return np.flatnonzero(self.edge_road == self.me + 1)

    @functools.cached_property
    def tile_pips(self) -> np.ndarray:
        """Pips per tile, robber-blind."""
        return _pips(self.tile_number)

    def port_kind(self, vertex: int) -> int | None:
        """The port kind on ``vertex`` (a ``Port`` value), or None off-port."""
        for p, (a, b) in enumerate(_PORTS):
            if vertex in (int(a), int(b)):
                return int(self.port_allocation[p])
        return None

    def vertex_production(self, vertex: int) -> np.ndarray:
        """Per-resource pips one building at ``vertex`` would earn."""
        out = np.zeros(N_RESOURCES, dtype=np.int64)
        for t in VERTEX_TILES[vertex]:
            r = int(self.tile_resource[t])
            if r < N_RESOURCES:
                out[r] += int(self.tile_pips[t])
        return out

    @functools.cached_property
    def my_production(self) -> np.ndarray:
        """Per-resource pips of our current buildings (cities count double)."""
        out = np.zeros(N_RESOURCES, dtype=np.int64)
        for v in self.my_vertices:
            out += self.vertex_production(int(v)) * int(self.vertex_type[v])
        return out

    @functools.cached_property
    def port_ratio(self) -> np.ndarray:
        """Our maritime ratio per resource (4, 3 with a general port, 2 with
        the matching port) — mirrors the engine's ``port_ratio``."""
        ratio = np.full(N_RESOURCES, 4, dtype=np.int64)
        for p, (a, b) in enumerate(_PORTS):
            if self.me + 1 in (self.vertex_owner[a], self.vertex_owner[b]):
                kind = int(self.port_allocation[p])
                if kind == Port.GENERAL:
                    ratio = np.minimum(ratio, 3)
                else:
                    ratio[kind] = 2
        return ratio

    @property
    def my_total_vp(self) -> int:
        """Building VP + held awards + own (hidden) Victory Point cards."""
        total = int(self.victory_points[self.me])
        total += 2 * (self.longest_road_owner == self.me)
        total += 2 * (self.largest_army_owner == self.me)
        return total + int(self.dev_hand[DevCard.VICTORY_POINT])

    def my_longest_trail(self) -> tuple[int, set[int]]:
        """Our longest road's length and the end vertices it can grow from
        (ends sitting on an opponent's building are excluded: a road cannot
        continue through one). DFS over our <= 15 edges, as the rules count
        a trail: no edge reused, no passing through an opponent's building."""
        adj: dict[int, list[int]] = {}
        for e in self.my_edges:
            for v in EDGE_ENDPOINTS[int(e)]:
                adj.setdefault(int(v), []).append(int(e))
        if not adj:
            return 0, set()
        mine = self.me + 1
        best, ends = 0, set()

        def dfs(v: int, used: set[int], length: int) -> None:
            nonlocal best, ends
            if length > best:
                best, ends = length, {v}
            elif length == best:
                ends.add(v)
            if self.vertex_owner[v] not in (0, mine):  # may end here, not pass
                return
            for e in adj.get(v, []):
                if e in used:
                    continue
                a, b = int(EDGE_ENDPOINTS[e, 0]), int(EDGE_ENDPOINTS[e, 1])
                used.add(e)
                dfs(b if a == v else a, used, length + 1)
                used.discard(e)

        for e in self.my_edges:
            a, b = int(EDGE_ENDPOINTS[int(e), 0]), int(EDGE_ENDPOINTS[int(e), 1])
            dfs(a, {int(e)}, 1)
            dfs(b, {int(e)}, 1)
        grow = {v for v in ends if int(self.vertex_owner[v]) in (0, self.me + 1)}
        return best, grow

    # -- Expansion search ------------------------------------------------------

    def settleable(self, vertex: int) -> bool:
        """Empty and distance-rule clear right now (build legality minus the
        own-road requirement — the planner's road path supplies that)."""
        if self.vertex_owner[vertex] != 0:
            return False
        return all(self.vertex_owner[n] == 0 for n in VERTEX_NEIGHBORS[vertex])

    def expansion_paths(self, max_roads: int) -> list[tuple[int, list[int]]]:
        """Every settleable vertex reachable from our network within
        ``max_roads`` new roads, with the new edges to build (BFS, so a
        shortest path each). An opponent's building blocks passage through
        its vertex (rulebook road rule)."""
        dist: dict[int, list[int]] = {}  # vertex -> new edges built to reach it
        frontier: list[int] = []
        for e in self.my_edges:
            for v in (int(EDGE_ENDPOINTS[e, 0]), int(EDGE_ENDPOINTS[e, 1])):
                owner = int(self.vertex_owner[v])
                if owner in (0, self.me + 1) and v not in dist:
                    dist[v] = []
                    frontier.append(v)
        for v in self.my_vertices:
            if int(v) not in dist:
                dist[int(v)] = []
                frontier.append(int(v))
        out: list[tuple[int, list[int]]] = []
        seen_targets: set[int] = set()
        for _ in range(max_roads):
            next_frontier: list[int] = []
            for v in frontier:
                for e in VERTEX_EDGES[v]:
                    if self.edge_road[e] != 0:
                        continue
                    n = int(
                        EDGE_ENDPOINTS[e, 1]
                        if EDGE_ENDPOINTS[e, 0] == v
                        else EDGE_ENDPOINTS[e, 0]
                    )
                    if n in dist:
                        continue
                    dist[n] = [*dist[v], e]
                    owner = int(self.vertex_owner[n])
                    if owner == 0:
                        next_frontier.append(n)
                        if n not in seen_targets and self.settleable(n):
                            seen_targets.add(n)
                            out.append((n, dist[n]))
            frontier = next_frontier
        return out
