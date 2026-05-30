"""Text rendering of a Catan board, for human-readable expect-test snapshots.

Shared by the board tests and the per-action tests. ``BoardRenderer`` turns a
``(BoardLayout, BoardState)`` pair into an ASCII hex map (tiles, number tokens,
robber, ports, settlements/cities, roads) plus ``tabulate`` summary tables. This
is a *test-only* utility: the engine keeps no screen geometry, so the vertex /
tile positions are re-derived here from the engine's own index ordering.

The geometry mirrors ``layout._generate_mappings`` (cube coordinates generated in
tile order) and projects every vertex onto a character grid. A vertex cube coord
``(a, b, c)`` sums to +/-1; a tile centre sums to 0. The projection
``row = 3a - (a+b+c)``, ``col = c - b`` lays the standard Catan hexagon out as a
regular honeycomb. The edge list comes straight from ``_edge_vertex_map`` so road
rendering uses the exact same index ordering as ``BoardState.edge_road``.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
from tabulate import tabulate

from catan_engine.dev_cards import N_DEV_CARD_TYPES, DevCard
from catan_engine.layout import (
    NO_INDEX,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    BoardLayout,
    _edge_vertex_map,
    _port_vertices_map,
)
from catan_engine.port import Port
from catan_engine.resources import N_PLAYERS, compute_bank_resources
from catan_engine.state import BoardState, GamePhase
from catan_engine.tile import Tile
from tests.reference import player_total_vp

_VERTEX_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _build_geometry() -> tuple[
    list[tuple[int, int, int]],  # tile centre cube coords, in tile order
    dict[int, tuple[int, int, int]],  # vertex index -> cube coord
]:
    vertices: dict[tuple[int, int, int], int] = {}
    centres: list[tuple[int, int, int]] = []
    nxt = 0
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(s) <= 2:
                centres.append((q, r, s))
                for dq, dr, ds in _VERTEX_DIRS:
                    v = (q + dq, r + dr, s + ds)
                    if v not in vertices:
                        vertices[v] = nxt
                        nxt += 1
    inv = {idx: cube for cube, idx in vertices.items()}
    return centres, inv


_TILE_CENTRES, _VERTEX_CUBE = _build_geometry()
# Canonical edge -> (vertex, vertex) ordering, owned by the engine.
_EDGE_VERTICES = [(int(a), int(b)) for a, b in np.asarray(_edge_vertex_map).tolist()]

# Character-grid scale: each cube step is this many rows / columns.
_ROW_SCALE = 2
_COL_SCALE = 4


def _vertex_xy(idx: int) -> tuple[int, int]:
    a, b, c = _VERTEX_CUBE[idx]
    return (3 * a - (a + b + c)) * _ROW_SCALE, (c - b) * _COL_SCALE


def _tile_xy(tile: int) -> tuple[int, int]:
    a, b, c = _TILE_CENTRES[tile]
    return 3 * a * _ROW_SCALE, (c - b) * _COL_SCALE


class BoardRenderer:
    """Renders a full Catan board (layout + state) as text.

    Produces an ASCII hex map showing tiles, number tokens, the robber, ports,
    settlements/cities and roads, followed by ``tabulate`` summary tables for
    the state that cannot live on the map: turn/phase, player hands, dev cards,
    awards, victory points and the bank.
    """

    _MARGIN = 4

    def __init__(
        self, layout: BoardLayout, state: BoardState, batch_idx: int = 0
    ) -> None:
        self._state = state
        self._b = batch_idx
        self._resource = np.asarray(layout.tile_resource[batch_idx])
        self._number = np.asarray(layout.tile_number[batch_idx])
        self._port = np.asarray(layout.port_allocation[batch_idx])
        self._vertex_owner = np.asarray(state.vertex_owner[batch_idx])
        self._vertex_type = np.asarray(state.vertex_type[batch_idx])
        self._edge_road = np.asarray(state.edge_road[batch_idx])
        self._robber = int(np.asarray(state.robber[batch_idx]))
        self._resources = np.asarray(state.player_resources[batch_idx])
        self._dev_hand = np.asarray(state.dev_hand[batch_idx])
        self._dev_deck = np.asarray(state.dev_deck[batch_idx])
        self._knights = np.asarray(state.knights_played[batch_idx])

    # -- ASCII map --------------------------------------------------------

    def _blank_canvas(self) -> tuple[list[list[str]], int, int]:
        pts = [_vertex_xy(i) for i in range(N_VERTICES)]
        pts += [_tile_xy(t) for t in range(N_TILES)]
        min_r = min(r for r, _ in pts) - self._MARGIN
        min_c = min(c for _, c in pts) - self._MARGIN
        height = max(r for r, _ in pts) - min_r + 1 + self._MARGIN
        width = max(c for _, c in pts) - min_c + 1 + self._MARGIN
        grid = [[" "] * width for _ in range(height)]
        return grid, min_r, min_c

    def render_map(self) -> str:
        grid, min_r, min_c = self._blank_canvas()
        height, width = len(grid), len(grid[0])

        def put(r: int, c: int, ch: str) -> None:
            rr, cc = r - min_r, c - min_c
            if 0 <= rr < height and 0 <= cc < width:
                grid[rr][cc] = ch

        def put_str(r: int, c: int, text: str) -> None:
            for j, ch in enumerate(text):
                put(r, c + j, ch)

        self._draw_edges(put)
        self._draw_tiles(put_str)
        self._draw_vertices(put)
        self._draw_ports(put_str)

        return "\n".join("".join(row).rstrip() for row in grid)

    def _draw_edges(self, put: Any) -> None:
        for edge_idx, (a, b) in enumerate(_EDGE_VERTICES):
            r1, c1 = _vertex_xy(a)
            r2, c2 = _vertex_xy(b)
            dr, dc = r2 - r1, c2 - c1
            owner = int(self._edge_road[edge_idx])
            if dc == 0:
                line = "|"
            elif dr * dc > 0:
                line = "\\"
            else:
                line = "/"
            ch = str(owner) if owner else line
            steps = max(abs(dr), abs(dc))
            for t in range(1, steps):
                put(round(r1 + dr * t / steps), round(c1 + dc * t / steps), ch)

    def _draw_tiles(self, put_str: Any) -> None:
        for tile in range(N_TILES):
            r, c = _tile_xy(tile)
            resource = Tile(int(self._resource[tile]))
            put_str(r - 1, c - 1, str(resource))
            if resource != Tile.DESERT:
                put_str(r, c - 1, f"{int(self._number[tile]):>2}")
            if self._robber == tile:
                put_str(r + 1, c - 1, "<R>")

    def _draw_vertices(self, put: Any) -> None:
        for idx in range(N_VERTICES):
            r, c = _vertex_xy(idx)
            owner = int(self._vertex_owner[idx])
            if owner == 0:
                put(r, c, "o")
            elif int(self._vertex_type[idx]) == 2:  # city
                put(r, c, "ABCD"[owner - 1])
            else:  # settlement
                put(r, c, str(owner))

    def _draw_ports(self, put_str: Any) -> None:
        for port_idx in range(N_PORTS):
            v1, v2 = (int(v) for v in _port_vertices_map[port_idx])
            r1, c1 = _vertex_xy(v1)
            r2, c2 = _vertex_xy(v2)
            mid_r, mid_c = (r1 + r2) / 2, (c1 + c2) / 2
            norm = math.hypot(mid_r, mid_c) or 1.0
            out_r = round(mid_r + mid_r / norm * 2.2)
            out_c = round(mid_c + mid_c / norm * 5.0)
            label = str(Port(int(self._port[port_idx])))
            put_str(out_r, out_c - 1, label)

    # -- Tables -----------------------------------------------------------

    def render_tables(self) -> str:
        return "\n\n".join(
            [
                self._status_line(),
                self._players_table(),
                self._awards_line(),
                self._dev_deck_line(),
                self._bank_table(),
                self._robber_line(),
            ]
        )

    def _status_line(self) -> str:
        s, b = self._state, self._b
        phase = GamePhase(int(s.phase[b]))
        cur = int(s.current_player[b])
        dice = int(s.dice_roll[b])
        fields = [
            f"Phase {phase}",
            f"Current player {cur + 1}",
            f"Dice {dice if dice else '-'}",
        ]
        if phase in (GamePhase.SETUP_SETTLEMENT, GamePhase.SETUP_ROAD):
            fields.append(f"Setup {int(s.setup_index[b])}/{2 * N_PLAYERS}")
        return "  |  ".join(fields)

    def _players_table(self) -> str:
        s, b = self._state, self._b
        cur = int(s.current_player[b])
        lr_owner = int(s.longest_road_owner[b])
        la_owner = int(s.largest_army_owner[b])
        rows = []
        for p in range(N_PLAYERS):
            player = p + 1
            hand = self._resources[p]
            settlements = int(
                np.sum((self._vertex_owner == player) & (self._vertex_type == 1))
            )
            cities = int(
                np.sum((self._vertex_owner == player) & (self._vertex_type == 2))
            )
            roads = int(np.sum(self._edge_road == player))
            awards = " ".join(
                tag for tag, owner in (("LR", lr_owner), ("LA", la_owner)) if owner == p
            )
            rows.append(
                [
                    ">" if p == cur else "",
                    player,
                    *(int(x) for x in hand),
                    int(hand.sum()),
                    int(self._dev_hand[p].sum()),
                    int(self._knights[p]),
                    settlements,
                    cities,
                    roads,
                    player_total_vp(s, p, b),
                    awards,
                ]
            )
        headers = [
            "",
            "P",
            "Shp",
            "Wht",
            "Wod",
            "Brk",
            "Ore",
            "Hand",
            "Dev",
            "Knt",
            "Set",
            "Cit",
            "Rd",
            "VP",
            "Awd",
        ]
        return "Players\n" + tabulate(rows, headers=headers, tablefmt="grid")

    def _awards_line(self) -> str:
        s, b = self._state, self._b
        lr_owner = int(s.longest_road_owner[b])
        la_owner = int(s.largest_army_owner[b])
        if lr_owner == NO_INDEX:
            lr = "unclaimed"
        else:
            lr = f"player {lr_owner + 1} (length {int(s.longest_road_len[b])})"
        if la_owner == NO_INDEX:
            la = "unclaimed"
        else:
            la = f"player {la_owner + 1} ({int(self._knights[la_owner])} knights)"
        return f"Longest Road: {lr}    Largest Army: {la}"

    def _dev_deck_line(self) -> str:
        parts = [
            f"{DevCard(i)}:{int(self._dev_deck[i])}" for i in range(N_DEV_CARD_TYPES)
        ]
        return "Dev deck  " + "  ".join(parts)

    def _bank_table(self) -> str:
        bank = np.asarray(
            compute_bank_resources(jnp.asarray(self._resources[None, ...]))[0]
        )
        headers = ["Sheep", "Wheat", "Wood", "Brick", "Ore"]
        return "Bank\n" + tabulate(
            [[int(x) for x in bank]], headers=headers, tablefmt="grid"
        )

    def _robber_line(self) -> str:
        tile = self._robber
        resource = Tile(int(self._resource[tile]))
        token = "desert" if resource == Tile.DESERT else str(int(self._number[tile]))
        return f"Robber: tile {tile} ({resource}, {token})"

    _LEGEND = (
        "Legend: o=vertex  1-4=settlement(player)  A-D=city(player)  "
        "digit on edge=road(player)  <R>=robber"
    )

    def __str__(self) -> str:
        return "\n".join(
            [
                "Catan Board",
                "=" * 60,
                "",
                self.render_map(),
                "",
                self._LEGEND,
                "",
                self.render_tables(),
                "",
            ]
        )
