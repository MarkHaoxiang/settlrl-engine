import math
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from expecttest import TestCase
from hypothesis import given, settings
from hypothesis import strategies as st
from tabulate import tabulate

from catan_engine.dev_cards import N_DEV_CARD_TYPES, DevCard
from catan_engine.layout import (
    NO_INDEX,
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    BoardLayout,
    _edge_vertex_map,
    _port_vertices_map,
    _tile_vertex_map,
    make_layout,
)
from catan_engine.port import Port
from catan_engine.resources import N_PLAYERS, compute_bank_resources
from tests.reference import player_total_vp
from catan_engine.state import BoardState, GamePhase, make_board_state
from catan_engine.tile import Tile

# ---------------------------------------------------------------------------
# Board geometry
#
# The engine stores tiles / vertices / edges as flat arrays but keeps no screen
# geometry, so we re-derive the *positions* here purely for rendering. We
# reconstruct the same vertex numbering used by ``layout._generate_mappings``
# (cube coordinates, generated in tile order) and project every vertex onto a
# character grid.
#
# A vertex cube coord (a, b, c) sums to +/-1; a tile centre sums to 0. The
# projection ``row = 3a - (a+b+c)``, ``col = c - b`` lays the standard Catan
# hexagon out as a regular honeycomb. The edge list itself comes straight from
# the engine's ``_edge_vertex_map`` so that road rendering uses the exact same
# index ordering as ``BoardState.edge_road``.
# ---------------------------------------------------------------------------

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
            f"Turn {int(s.turn_number[b])}",
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


class TestMappings(TestCase):
    """Tests for the static tile-vertex and port-vertex mappings."""

    def test_unique_vertex_count(self) -> None:
        vertices = np.asarray(_tile_vertex_map)
        assert len(set(vertices.flatten().tolist())) == N_VERTICES

    def test_tile_vertex_uniqueness(self) -> None:
        vertices = np.asarray(_tile_vertex_map)
        for tile_idx in range(N_TILES):
            tile_verts = vertices[tile_idx].tolist()
            assert len(set(tile_verts)) == 6, f"tile {tile_idx} has duplicate vertices"

    def test_port_vertices_distinct(self) -> None:
        ports = np.asarray(_port_vertices_map)
        for i, (v1, v2) in enumerate(ports.tolist()):
            assert v1 != v2, f"port {i} has identical vertices"

    def test_edge_map_matches_geometry(self) -> None:
        # Every engine edge must join two cube-adjacent vertices (abs diff sorts
        # to (0, 1, 1)), which is exactly what the renderer assumes when drawing.
        assert len(_EDGE_VERTICES) == N_EDGES
        for a, b in _EDGE_VERTICES:
            ca, cb = _VERTEX_CUBE[a], _VERTEX_CUBE[b]
            diff = tuple(sorted(abs(ca[i] - cb[i]) for i in range(3)))
            assert diff == (0, 1, 1), f"edge ({a}, {b}) is not cube-adjacent"


class TestBoardGenerator(TestCase):
    board: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.board = make_layout(batch_size=1, key=jax.random.key(0))

    @given(st.integers(min_value=1, max_value=8))
    @settings(deadline=None)
    def test_output_shapes(self, batch_size: int) -> None:
        board = make_layout(batch_size=batch_size, key=jax.random.key(0))
        assert board.tile_resource.shape == (batch_size, N_TILES)
        assert board.tile_number.shape == (batch_size, N_TILES)
        assert board.port_allocation.shape == (batch_size, 9)

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_tile_resource_counts(self, seed: int) -> None:
        board = make_layout(batch_size=1, key=jax.random.key(seed))
        resources = np.asarray(board.tile_resource[0])
        unique, counts = np.unique(resources, return_counts=True)
        summary = sorted(f"{Tile(int(t))!s}: {int(c)}" for t, c in zip(unique, counts))
        self.assertExpectedInline(
            "\n".join(summary),
            """\
BRK: 3
DST: 1
ORE: 3
SHP: 4
WHT: 4
WOD: 4""",
        )

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_tile_number_distribution(self, seed: int) -> None:
        board = make_layout(batch_size=1, key=jax.random.key(seed))
        resources = np.asarray(board.tile_resource[0])
        numbers = np.asarray(board.tile_number[0])
        non_desert = resources != Tile.DESERT.value
        self.assertExpectedInline(
            str(sorted(numbers[non_desert].tolist())),
            """[2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]""",
        )

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_desert_has_no_number(self, seed: int) -> None:
        board = make_layout(batch_size=1, key=jax.random.key(seed))
        resources = np.asarray(board.tile_resource[0])
        numbers = np.asarray(board.tile_number[0])
        self.assertExpectedInline(
            str(numbers[resources == Tile.DESERT.value].tolist()), """[0]"""
        )

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_port_counts(self, seed: int) -> None:
        board = make_layout(batch_size=1, key=jax.random.key(seed))
        ports = np.asarray(board.port_allocation[0])
        unique, counts = np.unique(ports, return_counts=True)
        summary = sorted(f"{Port(int(p))!s}: {int(c)}" for p, c in zip(unique, counts))
        self.assertExpectedInline(
            "\n".join(summary),
            """\
3:1: 4
BRK: 1
ORE: 1
SHP: 1
WHT: 1
WOD: 1""",
        )


def _sample_board() -> tuple[BoardLayout, BoardState]:
    """A deterministic mid-game board with hand-placed state, for snapshots."""
    layout = make_layout(batch_size=1, key=jax.random.key(0))
    state = make_board_state(batch_size=1)

    vertex_owner = np.asarray(state.vertex_owner).copy()
    vertex_type = np.asarray(state.vertex_type).copy()
    edge_road = np.asarray(state.edge_road).copy()
    resources = np.asarray(state.player_resources).copy()
    vp = np.asarray(state.victory_points).copy()
    dev_hand = np.asarray(state.dev_hand).copy()
    dev_deck = np.asarray(state.dev_deck).copy()
    knights = np.asarray(state.knights_played).copy()

    # player 1: settlement @5, city @44; player 2: city @12; player 3: settlement @30
    for vertex, owner, kind in [(5, 1, 1), (44, 1, 2), (12, 2, 2), (30, 3, 1)]:
        vertex_owner[0, vertex] = owner
        vertex_type[0, vertex] = kind
    for edge, owner in [(0, 1), (3, 1), (10, 2), (20, 3), (40, 1)]:
        edge_road[0, edge] = owner
    resources[0, 0] = [2, 1, 0, 3, 1]
    resources[0, 1] = [0, 0, 2, 1, 0]
    resources[0, 2] = [1, 1, 1, 0, 2]
    vp[0] = [4, 2, 1, 0]

    # dev cards: player 1 holds a VP card + Monopoly, player 2 holds a Knight.
    dev_hand[0, 0, int(DevCard.VICTORY_POINT)] = 1
    dev_hand[0, 0, int(DevCard.MONOPOLY)] = 1
    dev_hand[0, 1, int(DevCard.KNIGHT)] = 1
    for card in (DevCard.VICTORY_POINT, DevCard.MONOPOLY, DevCard.KNIGHT):
        dev_deck[0, int(card)] -= 1
    knights[0] = [3, 0, 1, 0]

    def u8(value: int) -> jax.Array:
        return jnp.asarray(np.array([value], dtype=np.uint8))

    state = state._replace(
        vertex_owner=jnp.asarray(vertex_owner),
        vertex_type=jnp.asarray(vertex_type),
        edge_road=jnp.asarray(edge_road),
        robber=u8(9),
        player_resources=jnp.asarray(resources),
        victory_points=jnp.asarray(vp),
        dev_hand=jnp.asarray(dev_hand),
        dev_deck=jnp.asarray(dev_deck),
        knights_played=jnp.asarray(knights),
        current_player=u8(0),
        phase=u8(int(GamePhase.MAIN)),
        turn_number=jnp.asarray(np.array([12], dtype=np.uint16)),
        dice_roll=u8(8),
        longest_road_owner=u8(0),
        longest_road_len=u8(5),
        largest_army_owner=u8(0),
    )
    return layout, state


class TestBoardRenderer(TestCase):
    def test_edge_index_in_range(self) -> None:
        for a, b in _EDGE_VERTICES:
            assert 0 <= a < N_VERTICES and 0 <= b < N_VERTICES

    def test_empty_board_snapshot(self) -> None:
        layout = make_layout(batch_size=1, key=jax.random.key(0))
        state = make_board_state(batch_size=1)
        self.assertExpectedInline(
            str(BoardRenderer(layout, state)),
            """\
Catan Board
============================================================




          ORE             3:1
               /o\\     /o\\     /o\\
              /   \\   /   \\   /   \\
            o/     \\o/     \\o/     \\o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |  <R>  |
           /o\\     /o\\     /o\\     /o\\   3:1
          /   \\   /   \\   /   \\   /   \\
        o/     \\o/     \\o/     \\o/     \\o
  WOD   |  WHT  |  WOD  |  WOD  |  SHP  |
        |   9   |   2   |  10   |  11   |
        |       |       |       |       |
       /o\\     /o\\     /o\\     /o\\     /o\\
      /   \\   /   \\   /   \\   /   \\   /   \\
    o/     \\o/     \\o/     \\o/     \\o/     \\o
    |  ORE  |  SHP  |  WOD  |  DST  |  WHT  |
    |   8   |   4   |   3   |       |  12   |   3
    |       |       |       |       |       |
    o\\     /o\\     /o\\     /o\\     /o\\     /o
      \\   /   \\   /   \\   /   \\   /   \\   /
       \\o/     \\o/     \\o/     \\o/     \\o/
        |  SHP  |  ORE  |  BRK  |  BRK  |
        |   8   |   3   |  11   |   6   |
  3:1   |       |       |       |       |
        o\\     /o\\     /o\\     /o\\     /o
          \\   /   \\   /   \\   /   \\   /
           \\o/     \\o/     \\o/     \\o/   BRK
            |  WHT  |  WHT  |  WOD  |
            |   4   |   9   |   5   |
            |       |       |       |
            o\\     /o\\     /o\\     /o
              \\   /   \\   /   \\   /
               \\o/     \\o/     \\o/
          SHP             WHT




Legend: o=vertex  1-4=settlement(player)  A-D=city(player)  digit on edge=road(player)  <R>=robber

Turn 0  |  Phase SETUP_SETTLEMENT  |  Current player 1  |  Dice -  |  Setup 0/8

Players
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   P |   Shp |   Wht |   Wod |   Brk |   Ore |   Hand |   Dev |   Knt |   Set |   Cit |   Rd |   VP | Awd   |
+====+=====+=======+=======+=======+=======+=======+========+=======+=======+=======+=======+======+======+=======+
| >  |   1 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   2 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   3 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   4 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+

Longest Road: unclaimed    Largest Army: unclaimed

Dev deck  KNT:14  RDB:2  YOP:2  MNP:2  VPT:5

Bank
+---------+---------+--------+---------+-------+
|   Sheep |   Wheat |   Wood |   Brick |   Ore |
+=========+=========+========+=========+=======+
|      19 |      19 |     19 |      19 |    19 |
+---------+---------+--------+---------+-------+

Robber: tile 0 (BRK, 10)
""",
        )

    def test_full_board_snapshot(self) -> None:
        layout, state = _sample_board()
        self.assertExpectedInline(
            str(BoardRenderer(layout, state)),
            """\
Catan Board
============================================================




          ORE             3:1
               /o\\     /o\\     1o\\
              /   \\   /   \\   1   \\
            B/     \\o/     \\o1     \\o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |       |
           3o\\     /o2     /1\\     1o\\   3:1
          3   \\   /   2   /   \\   1   \\
        o3     \\o/     2o/     \\o1     \\o
  WOD   |  WHT  |  WOD  |  WOD  |  SHP  |
        |   9   |   2   |  10   |  11   |
        |       |       |       |       |
       /o\\     /o\\     /o\\     /o\\     /o\\
      /   \\   /   \\   /   \\   /   \\   /   \\
    o/     \\o/     \\o/     \\o/     \\o/     \\o
    |  ORE  |  SHP  |  WOD  |  DST  |  WHT  1
    |   8   |   4   |   3   |       |  12   1   3
    |       |       |  <R>  |       |       1
    o\\     /o\\     /o\\     /3\\     /o\\     /o
      \\   /   \\   /   \\   /   \\   /   \\   /
       \\o/     \\o/     \\o/     \\o/     \\o/
        |  SHP  |  ORE  |  BRK  |  BRK  |
        |   8   |   3   |  11   |   6   |
  3:1   |       |       |       |       |
        o\\     /A\\     /o\\     /o\\     /o
          \\   /   \\   /   \\   /   \\   /
           \\o/     \\o/     \\o/     \\o/   BRK
            |  WHT  |  WHT  |  WOD  |
            |   4   |   9   |   5   |
            |       |       |       |
            o\\     /o\\     /o\\     /o
              \\   /   \\   /   \\   /
               \\o/     \\o/     \\o/
          SHP             WHT




Legend: o=vertex  1-4=settlement(player)  A-D=city(player)  digit on edge=road(player)  <R>=robber

Turn 12  |  Phase MAIN  |  Current player 1  |  Dice 8

Players
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   P |   Shp |   Wht |   Wod |   Brk |   Ore |   Hand |   Dev |   Knt |   Set |   Cit |   Rd |   VP | Awd   |
+====+=====+=======+=======+=======+=======+=======+========+=======+=======+=======+=======+======+======+=======+
| >  |   1 |     2 |     1 |     0 |     3 |     1 |      7 |     2 |     3 |     1 |     1 |    3 |    9 | LR LA |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   2 |     0 |     0 |     2 |     1 |     0 |      3 |     1 |     0 |     0 |     1 |    1 |    2 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   3 |     1 |     1 |     1 |     0 |     2 |      5 |     0 |     1 |     1 |     0 |    1 |    1 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   4 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+

Longest Road: player 1 (length 5)    Largest Army: player 1 (3 knights)

Dev deck  KNT:13  RDB:2  YOP:2  MNP:1  VPT:4

Bank
+---------+---------+--------+---------+-------+
|   Sheep |   Wheat |   Wood |   Brick |   Ore |
+=========+=========+========+=========+=======+
|      16 |      17 |     16 |      15 |    16 |
+---------+---------+--------+---------+-------+

Robber: tile 9 (WOD, 3)
""",
        )
