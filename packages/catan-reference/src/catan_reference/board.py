"""The standard Catan board, generated from cube coordinates.

The geometry is built from first principles (canonical hex-grid cube-coordinate
math), independently of ``catan-engine``. The only thing shared with the engine
is the cube-coordinate *convention* — which is a description of the physical
board, not a rule — so that a board can be translated between the two by cube
coordinate (see the engine's ``board/layout.py`` host-side lookups and the test
conversion layer). Reference vertex/edge/tile indices are this module's own and
need not match the engine's.

Cube coordinates ``(q, r, s)`` are integer triples. Tile centres satisfy
``q + r + s == 0`` and ``|q|, |r|, |s| <= 2`` (the 19 hexes). A vertex is a tile
centre offset by one unit along a single axis, so its coordinates sum to ``+1``
or ``-1`` (the 54 intersections). An edge joins a ``+1`` vertex to an adjacent
``-1`` vertex (the 72 paths).
"""

from __future__ import annotations

from dataclasses import dataclass

from catan_reference.types import PortType, Resource

Cube = tuple[int, int, int]

N_TILES = 19
N_VERTICES = 54
N_EDGES = 72

# Unit offsets from a tile centre to its six corner vertices.
_VERTEX_DIRS: tuple[Cube, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

# A ``+1`` vertex connects to a ``-1`` vertex when their difference is one of
# these (each shares the two tiles either side of the path).
_EDGE_DIFFS: tuple[Cube, ...] = ((1, 1, 0), (1, 0, 1), (0, 1, 1))


def _sub(a: Cube, b: Cube) -> Cube:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _build() -> tuple[list[Cube], list[Cube], list[tuple[int, int]]]:
    """Return (tile centres, vertices, edges), each in a stable sorted order."""
    tiles: list[Cube] = sorted(
        (q, r, -q - r) for q in range(-2, 3) for r in range(-2, 3) if abs(-q - r) <= 2
    )

    vertex_set: set[Cube] = set()
    for centre in tiles:
        for d in _VERTEX_DIRS:
            vertex_set.add((centre[0] + d[0], centre[1] + d[1], centre[2] + d[2]))
    vertices: list[Cube] = sorted(vertex_set)
    index = {cube: i for i, cube in enumerate(vertices)}

    edge_set: set[tuple[int, int]] = set()
    for cube in vertices:
        if sum(cube) != 1:
            continue
        for diff in _EDGE_DIFFS:
            other = _sub(cube, diff)
            if other in index:
                a, b = index[cube], index[other]
                edge_set.add((min(a, b), max(a, b)))
    edges: list[tuple[int, int]] = sorted(edge_set)
    return tiles, vertices, edges


_TILE_CUBES, _VERTEX_CUBES, _EDGES = _build()

_CUBE_TO_VERTEX: dict[Cube, int] = {c: i for i, c in enumerate(_VERTEX_CUBES)}
_CUBE_TO_TILE: dict[Cube, int] = {c: i for i, c in enumerate(_TILE_CUBES)}
_PAIR_TO_EDGE: dict[frozenset[int], int] = {
    frozenset(pair): e for e, pair in enumerate(_EDGES)
}

# Adjacency, derived once. Indices are this module's own.
TILE_VERTICES: list[list[int]] = [
    sorted(
        _CUBE_TO_VERTEX[(c[0] + d[0], c[1] + d[1], c[2] + d[2])] for d in _VERTEX_DIRS
    )
    for c in _TILE_CUBES
]
VERTEX_EDGES: list[list[int]] = [[] for _ in range(N_VERTICES)]
VERTEX_NEIGHBORS: list[list[int]] = [[] for _ in range(N_VERTICES)]
for _e, (_a, _b) in enumerate(_EDGES):
    VERTEX_EDGES[_a].append(_e)
    VERTEX_EDGES[_b].append(_e)
    VERTEX_NEIGHBORS[_a].append(_b)
    VERTEX_NEIGHBORS[_b].append(_a)
VERTEX_TILES: list[list[int]] = [[] for _ in range(N_VERTICES)]
for _t, _corners in enumerate(TILE_VERTICES):
    for _v in _corners:
        VERTEX_TILES[_v].append(_t)


def vertex_cube(vertex: int) -> Cube:
    return _VERTEX_CUBES[vertex]


def cube_to_vertex(cube: Cube) -> int:
    return _CUBE_TO_VERTEX[cube]


def tile_cube(tile: int) -> Cube:
    return _TILE_CUBES[tile]


def cube_to_tile(cube: Cube) -> int:
    return _CUBE_TO_TILE[cube]


def edge_vertices(edge: int) -> tuple[int, int]:
    return _EDGES[edge]


def edge_between(a: int, b: int) -> int:
    """Index of the edge joining vertices ``a`` and ``b`` (must be adjacent)."""
    return _PAIR_TO_EDGE[frozenset((a, b))]


@dataclass(frozen=True)
class Port:
    """A harbour: its trade type and the two coastal vertices that control it."""

    type: PortType
    vertices: tuple[int, int]


@dataclass(frozen=True)
class Layout:
    """The variable board: per-tile terrain + number token, and the harbours.

    ``tile_resource[t]`` is ``None`` for the desert; ``tile_number[t]`` is ``0``
    there. Geometry (which vertices/edges/tiles are adjacent) is fixed and lives
    in this module; only this allocation changes between games.
    """

    tile_resource: tuple[Resource | None, ...]  # len N_TILES
    tile_number: tuple[int, ...]  # len N_TILES, 0 for the desert
    ports: tuple[Port, ...]

    def port_at_vertex(self, vertex: int) -> Port | None:
        for port in self.ports:
            if vertex in port.vertices:
                return port
        return None
