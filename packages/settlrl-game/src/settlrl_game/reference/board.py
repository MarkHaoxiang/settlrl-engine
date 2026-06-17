"""The standard Settlrl board, generated from cube coordinates.

The geometry is built from first principles (canonical hex-grid cube-coordinate
math), independently of ``settlrl-engine``. The only thing shared with the engine
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
from random import Random
from typing import Literal

from settlrl_game.reference.types import PortType, Resource

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


# --- random board generation -----------------------------------------------

# The standard base-game allotment (rulebook): 19 tiles, 18 number tokens (the
# desert takes none), 9 harbours. Terrain and tokens shuffle freely over the
# tiles; harbour *types* shuffle over the fixed coastal positions below.
_TERRAIN: tuple[Resource | None, ...] = (
    None,  # desert
    *(Resource.WOOD,) * 4,
    *(Resource.SHEEP,) * 4,
    *(Resource.WHEAT,) * 4,
    *(Resource.BRICK,) * 3,
    *(Resource.ORE,) * 3,
)
# A single 2 and 12, every other pip 3..11 twice -- but no 7 (that is the robber).
_NUMBER_TOKENS: tuple[int, ...] = (
    2,
    12,
    *(p for p in (3, 4, 5, 6, 8, 9, 10, 11) for _ in (0, 1)),
)
_PORT_TYPES: tuple[PortType, ...] = (
    *(PortType.GENERIC,) * 4,
    PortType.SHEEP,
    PortType.WHEAT,
    PortType.WOOD,
    PortType.BRICK,
    PortType.ORE,
)
# The nine harbour positions are part of the physical board (like the cube
# convention itself), given as their two coastal vertices' cube coordinates.
_PORT_VERTICES: tuple[tuple[int, int], ...] = tuple(
    (cube_to_vertex(a), cube_to_vertex(b))
    for a, b in (
        ((3, 0, -2), (2, 0, -3)),
        ((-3, 2, 0), (-2, 3, 0)),
        ((0, -2, 3), (0, -3, 2)),
        ((-1, -1, 3), (-2, -1, 2)),
        ((-2, 1, 2), (-3, 1, 1)),
        ((1, -3, 1), (2, -2, 1)),
        ((2, -2, -1), (3, -1, -1)),
        ((1, 2, -2), (1, 1, -3)),
        ((-1, 2, -2), (-1, 3, -1)),
    )
)


# Rulebook Almanac variable set-up: number tokens A..R laid alphabetically along
# a counterclockwise spiral from a corner toward the centre, skipping the
# desert. The A..R letters map to this fixed number sequence:
SPIRAL_NUMBERS = (5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11)


def _spiral_tile_order() -> tuple[int, ...]:
    """Tile indices spiralling in from a corner (outer ring counterclockwise,
    then the middle ring, then the centre); consecutive tiles are cube-adjacent.
    Which corner it starts from is distributionally irrelevant -- the terrain
    shuffle is rotation/reflection-invariant -- so one fixed path suffices."""
    dirs = ((-1, 1, 0), (0, 1, -1), (1, 0, -1), (1, -1, 0), (0, -1, 1), (-1, 0, 1))
    order: list[int] = []
    for radius in (2, 1):
        cube = (0, -radius, radius)
        for d in dirs:
            for _ in range(radius):
                order.append(cube_to_tile(cube))
                cube = (cube[0] + d[0], cube[1] + d[1], cube[2] + d[2])
    order.append(cube_to_tile((0, 0, 0)))
    return tuple(order)


_SPIRAL_TILE_ORDER = _spiral_tile_order()


def desert_tile(layout: Layout) -> int:
    """The desert tile index (where the robber starts)."""
    return layout.tile_resource.index(None)


def _place_numbers(
    terrain: list[Resource | None], rng: Random, placement: Literal["random", "spiral"]
) -> list[int]:
    """Number tokens over the non-desert tiles: shuffled, or along the spiral."""
    numbers = [0] * N_TILES
    if placement == "spiral":
        token = 0
        for t in _SPIRAL_TILE_ORDER:
            if terrain[t] is not None:
                numbers[t] = SPIRAL_NUMBERS[min(token, len(SPIRAL_NUMBERS) - 1)]
                token += 1
    else:
        tokens = list(_NUMBER_TOKENS)
        rng.shuffle(tokens)
        supply = iter(tokens)
        for t in range(N_TILES):
            if terrain[t] is not None:
                numbers[t] = next(supply)
    return numbers


def random_layout(
    rng: Random, number_placement: Literal["random", "spiral"] = "random"
) -> Layout:
    """A random standard board: terrain and harbour types shuffled over the fixed
    geometry, number tokens laid by ``number_placement`` -- shuffled (``random``)
    or along the rulebook spiral (``spiral``). Terrain and ports depend only on
    ``rng``, so a seed's map is identical across modes; only the numbers differ.
    """
    terrain = list(_TERRAIN)
    rng.shuffle(terrain)
    types = list(_PORT_TYPES)
    rng.shuffle(types)
    ports = tuple(Port(t, vs) for t, vs in zip(types, _PORT_VERTICES, strict=True))
    numbers = _place_numbers(terrain, rng, number_placement)
    return Layout(tuple(terrain), tuple(numbers), ports)
