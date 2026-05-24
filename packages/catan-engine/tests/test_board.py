from typing import Any, ClassVar

import jax
import numpy as np
from expecttest import TestCase
from hypothesis import given, settings
from hypothesis import strategies as st

from catan_engine.board import (
    N_TILES,
    N_VERTICES,
    _port_vertices_map,
    _tile_vertex_map,
    make_board,
)
from catan_engine.port import Port
from catan_engine.tile import Tile

_EXPECTED_NUMBERS = sorted([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
_EXPECTED_RESOURCES = {
    Tile.SHEEP: 4, Tile.WHEAT: 4, Tile.WOOD: 4,
    Tile.BRICK: 3, Tile.ORE: 3, Tile.DESERT: 1,
}
_EXPECTED_PORTS = {
    Port.SHEEP: 1, Port.WHEAT: 1, Port.WOOD: 1,
    Port.BRICK: 1, Port.ORE: 1, Port.GENERAL: 4,
}


class AsciiBoard:
    """ASCII renderer for a single Catan board from a make_board batch."""

    # Tile indices per display row, q = -2 .. +2, r increasing within each row.
    _ROWS = [
        [0, 1, 2],
        [3, 4, 5, 6],
        [7, 8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
    ]

    def __init__(self, board: dict, batch_idx: int = 0) -> None:
        self._resource = np.asarray(board["tile_resource"][batch_idx])
        self._number = np.asarray(board["tile_number"][batch_idx])

    def __str__(self) -> str:
        lines = ["Catan Board", "=" * 50, ""]
        for row in self._ROWS:
            indent = " " * ((5 - len(row)) * 5)
            cells = []
            for i in row:
                res = Tile(int(self._resource[i]))
                num = int(self._number[i])
                cell = (
                    f"[{res!s}   ]"
                    if res == Tile.DESERT
                    else f"[{res!s} {num:2d}]"
                )
                cells.append(cell)
            lines.append(indent + "  ".join(cells))
        lines.append("")
        return "\n".join(lines) + "\n"


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


class TestBoardGenerator(TestCase):
    board: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.board = make_board(batch_size=1, key=jax.random.key(0))

    @given(st.integers(min_value=1, max_value=8))
    @settings(deadline=None)
    def test_output_shapes(self, batch_size: int) -> None:
        board = make_board(batch_size=batch_size, key=jax.random.key(0))
        assert board["tile_resource"].shape == (batch_size, N_TILES)
        assert board["tile_number"].shape == (batch_size, N_TILES)
        assert board["port_allocation"].shape == (batch_size, 9)

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_tile_resource_counts(self, seed: int) -> None:
        board = make_board(batch_size=1, key=jax.random.key(seed))
        resources = np.asarray(board["tile_resource"][0])
        for tile, expected in _EXPECTED_RESOURCES.items():
            assert int((resources == tile.value).sum()) == expected, tile

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_tile_number_distribution(self, seed: int) -> None:
        board = make_board(batch_size=1, key=jax.random.key(seed))
        resources = np.asarray(board["tile_resource"][0])
        numbers = np.asarray(board["tile_number"][0])
        non_desert = resources != Tile.DESERT.value
        assert sorted(numbers[non_desert].tolist()) == _EXPECTED_NUMBERS

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_desert_has_no_number(self, seed: int) -> None:
        board = make_board(batch_size=1, key=jax.random.key(seed))
        resources = np.asarray(board["tile_resource"][0])
        numbers = np.asarray(board["tile_number"][0])
        desert_numbers = numbers[resources == Tile.DESERT.value].tolist()
        assert desert_numbers == [0]

    @given(st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_port_counts(self, seed: int) -> None:
        board = make_board(batch_size=1, key=jax.random.key(seed))
        ports = np.asarray(board["port_allocation"][0])
        for port, expected in _EXPECTED_PORTS.items():
            assert int((ports == port.value).sum()) == expected, port

    @given(st.integers(min_value=2, max_value=6), st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=20, deadline=None)
    def test_batch_independence(self, batch_size: int, seed: int) -> None:
        board = make_board(batch_size=batch_size, key=jax.random.key(seed))
        resources = np.asarray(board["tile_resource"])
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                assert not np.array_equal(resources[i], resources[j]), (
                    f"boards {i} and {j} have identical resource layouts"
                )

    def test_ascii_snapshot(self) -> None:
        self.assertExpectedInline(
            str(AsciiBoard(self.board)),
            """\
Catan Board
==================================================

          [BRK 10]  [ORE  6]  [SHP  5]
     [SHP 11]  [WOD 10]  [WOD  2]  [WHT  9]
[WHT 12]  [DST   ]  [WOD  3]  [SHP  4]  [ORE  8]
     [BRK  6]  [BRK 11]  [ORE  3]  [SHP  8]
          [WOD  5]  [WHT  9]  [WHT  4]

""",
        )
