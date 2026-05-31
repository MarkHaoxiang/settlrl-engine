from typing import Any, ClassVar

import jax
import numpy as np
from expecttest import TestCase
from hypothesis import given, settings
from hypothesis import strategies as st

from catan_engine.board.layout import (
    EDGE_V,
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    PORT_V,
    TILE_V,
    edge_cubes,
    edge_index,
    make_layout,
    tile_cube,
    tile_index,
    vertex_cube,
    vertex_index,
)
from catan_engine.board.port import Port
from catan_engine.board.tile import Tile
from tests.render import _EDGE_VERTICES, _VERTEX_CUBE


class TestMappings(TestCase):
    """Tests for the static incidence maps generated in ``layout``."""

    def test_map_shapes(self) -> None:
        assert TILE_V.shape == (N_TILES, 6)
        assert PORT_V.shape == (N_PORTS, 2)
        assert EDGE_V.shape == (N_EDGES, 2)

    def test_unique_vertex_count(self) -> None:
        vertices = np.asarray(TILE_V)
        assert len(set(vertices.flatten().tolist())) == N_VERTICES

    def test_tile_vertex_uniqueness(self) -> None:
        vertices = np.asarray(TILE_V)
        for tile_idx in range(N_TILES):
            tile_verts = vertices[tile_idx].tolist()
            assert len(set(tile_verts)) == 6, f"tile {tile_idx} has duplicate vertices"

    def test_port_vertices_distinct(self) -> None:
        ports = np.asarray(PORT_V)
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


class TestCubeCoords(TestCase):
    """Cube-coordinate <-> index lookup helpers."""

    def test_vertex_roundtrip(self) -> None:
        for v in range(N_VERTICES):
            cube = vertex_cube(v)
            assert sum(cube) in (1, -1), f"vertex {v} cube {cube} sums to {sum(cube)}"
            assert vertex_index(cube) == v

    def test_tile_roundtrip(self) -> None:
        for t in range(N_TILES):
            cube = tile_cube(t)
            assert sum(cube) == 0, f"tile {t} centre {cube} does not sum to 0"
            assert tile_index(cube) == t

    def test_edge_roundtrip_matches_edge_v(self) -> None:
        ev = np.asarray(EDGE_V)
        for e in range(N_EDGES):
            ca, cb = edge_cubes(e)
            assert edge_index(ca, cb) == e
            assert {vertex_index(ca), vertex_index(cb)} == {
                int(ev[e, 0]),
                int(ev[e, 1]),
            }

    def test_edge_index_order_independent(self) -> None:
        ca, cb = edge_cubes(0)
        assert edge_index(ca, cb) == edge_index(cb, ca)

    def test_unknown_coord_raises(self) -> None:
        with self.assertRaises(KeyError):
            vertex_index((9, 9, 9))
        with self.assertRaises(KeyError):
            tile_index((9, 9, 9))
