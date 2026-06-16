from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Key, UInt8

from settlrl_engine.board.port import Port
from settlrl_engine.board.tile import Tile

N_TILES = 19
N_VERTICES = 54
N_EDGES = 72  # V - E + F = 2, F = N_TILES + 1 outer face
N_PORTS = 9

# Maximum number of edges/neighbours/tiles incident to a single vertex.
MAX_VERTEX_DEGREE = 3

TileResourceArray = UInt8[Array, f"batch tiles={N_TILES}"]
TileNumberArray = UInt8[Array, f"batch tiles={N_TILES}"]
PortAllocationArray = UInt8[Array, f"batch ports={N_PORTS}"]

# Single-game (un-batched) views, used by the rule modules.
PortAllocVec = UInt8[Array, f"ports={N_PORTS}"]
TileNumberVec = UInt8[Array, f"tiles={N_TILES}"]


class BoardLayout(NamedTuple):
    """Immutable board geometry: tile resources, number tokens, and port types."""

    tile_resource: TileResourceArray
    tile_number: TileNumberArray
    port_allocation: PortAllocationArray


# Two vertices share an edge iff one cube coord sums to +1, the other to -1,
# and their difference is one of these (matches the renderer's geometry).
_EDGE_DIFFS = ((1, 1, 0), (1, 0, 1), (0, 1, 1))


Cube = tuple[int, int, int]


def _generate_mappings() -> tuple[
    Int[Array, "tiles 6"],
    Int[Array, "edges 2"],
    Int[Array, "ports 2"],
    dict[Cube, int],
    list[Cube],
]:
    tile_vertex_mapping: list[list[int]] = []
    tile_centres: list[Cube] = []  # cube coord of each tile centre, in tile order
    vertices: dict[Cube, int] = {}

    vertex_dirs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    port_vertices = (
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

    # Axial coordinates for the vertices of a Settlrl board
    i = 0
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(s) <= 2:
                # Valid tile
                tile_centres.append((q, r, s))
                tile_vertex_mapping.append([])
                for dq, dr, ds in vertex_dirs:
                    vertex = (q + dq, r + dr, s + ds)
                    if vertex not in vertices:
                        vertices[vertex] = i
                        i += 1
                    tile_vertex_mapping[-1].append(vertices[vertex])

    ports = [[vertices[v1], vertices[v2]] for v1, v2 in port_vertices]

    # Edges: connect each cube-sum +1 vertex to its cube-sum -1 neighbours.
    # Sorted to give a stable, canonical edge index for BoardState.edge_road.
    edge_list: list[tuple[int, int]] = []
    for cube, idx in vertices.items():
        if sum(cube) == 1:
            for dq, dr, ds in _EDGE_DIFFS:
                other = (cube[0] - dq, cube[1] - dr, cube[2] - ds)
                if other in vertices:
                    a, b = idx, vertices[other]
                    edge_list.append((min(a, b), max(a, b)))
    edge_list.sort()

    # int32 so the traceable rule helpers can gather/index through them directly
    # under a JAX trace.
    return (
        jnp.array(tile_vertex_mapping, dtype=jnp.int32),
        jnp.array(ports, dtype=jnp.int32),
        jnp.array(edge_list, dtype=jnp.int32),
        vertices,
        tile_centres,
    )


# Static board geometry as int32 ``jnp`` arrays, shared by the engine's traceable
# rule modules and the test reference / renderer. These are the dense,
# padding-free incidence maps: ``EDGE_V`` is the COO ``edge_index`` (every edge
# has exactly two endpoints) and ``TILE_V`` / ``PORT_V`` are dense hyperedge
# maps. The rule modules derive vertex-incidence on the fly by scattering over
# them (see e.g. ``placement`` / ``trade`` / ``dice``), so no ragged vertex->*
# reverse maps or padding sentinels are stored here. Shapes checked in
# ``tests/test_layout.py``.
(
    TILE_V,  # (N_TILES, 6) tile -> corner vertices
    PORT_V,  # (N_PORTS, 2) port -> its two vertices
    EDGE_V,  # (N_EDGES, 2) edge -> endpoint vertices (the COO edge_index)
    _CUBE_TO_VERTEX,  # cube coord -> vertex index
    _TILE_CUBE,  # tile index -> centre cube coord
) = _generate_mappings()

# Host-side cube-coordinate <-> index lookups, for human-readable inspection of
# the board (e.g. read vertex_owner at a known corner). Cube coords are
# (q, r, s) integer triples: tile centres sum to 0, vertices to +/-1. These are
# plain Python lookups built once at import -- NOT for use inside a JAX trace.
_VERTEX_CUBE: dict[int, Cube] = {idx: cube for cube, idx in _CUBE_TO_VERTEX.items()}
_CUBE_TO_TILE: dict[Cube, int] = {cube: t for t, cube in enumerate(_TILE_CUBE)}
_VPAIR_TO_EDGE: dict[frozenset[int], int] = {
    frozenset((int(a), int(b))): e for e, (a, b) in enumerate(EDGE_V.tolist())
}


def _cube_key(cube: Cube) -> Cube:
    return (int(cube[0]), int(cube[1]), int(cube[2]))


def vertex_index(cube: Cube) -> int:
    """Vertex index at cube coord ``(q, r, s)`` (must sum to +/-1)."""
    key = _cube_key(cube)
    if key not in _CUBE_TO_VERTEX:
        raise KeyError(f"no vertex at cube {key}")
    return _CUBE_TO_VERTEX[key]


def vertex_cube(index: int) -> Cube:
    """Cube coord ``(q, r, s)`` of vertex ``index``."""
    return _VERTEX_CUBE[index]


def edge_index(cube_a: Cube, cube_b: Cube) -> int:
    """Edge index joining the two vertices at the given cube coords."""
    pair = frozenset((vertex_index(cube_a), vertex_index(cube_b)))
    if pair not in _VPAIR_TO_EDGE:
        raise KeyError(
            f"no edge between cubes {_cube_key(cube_a)} and {_cube_key(cube_b)}"
        )
    return _VPAIR_TO_EDGE[pair]


def edge_cubes(index: int) -> tuple[Cube, Cube]:
    """The two endpoint cube coords of edge ``index``."""
    a, b = (int(x) for x in EDGE_V[index])
    return _VERTEX_CUBE[a], _VERTEX_CUBE[b]


def tile_index(cube: Cube) -> int:
    """Tile index at centre cube coord ``(q, r, s)`` (must sum to 0)."""
    key = _cube_key(cube)
    if key not in _CUBE_TO_TILE:
        raise KeyError(f"no tile at cube {key}")
    return _CUBE_TO_TILE[key]


def tile_cube(index: int) -> Cube:
    """Centre cube coord ``(q, r, s)`` of tile ``index``."""
    return _TILE_CUBE[index]


# -- Spiral ("Set-up, Variable") number placement ------------------------------
#
# Rulebook Almanac, "Set-up, Variable": sort the number tokens letter side up
# and place them on the terrain hexes in alphabetical order, starting at a
# corner of the island and proceeding counterclockwise, spiralling toward the
# centre; the desert is skipped (it never gets a token). The letters A..R on
# the token backs correspond to this fixed number sequence:
SPIRAL_NUMBERS = (5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11)


def _spiral_tile_order() -> tuple[int, ...]:
    """Tile indices in spiral order: the outer ring counterclockwise from the
    rightmost corner (on the renderer's projection), then the middle ring, then
    the centre. Consecutive tiles are always cube-adjacent (checked in tests).

    Which corner the spiral starts from (and its handedness) is irrelevant to
    the distribution of generated boards -- the terrain shuffle is invariant
    under rotation/reflection -- so a single fixed path suffices.
    """
    # Counterclockwise ring steps, in order, starting upward from the right.
    dirs = ((-1, 1, 0), (0, 1, -1), (1, 0, -1), (1, -1, 0), (0, -1, 1), (-1, 0, 1))
    order: list[int] = []
    for radius in (2, 1):
        cube = (0, -radius, radius)  # ring corner in line with the start corner
        for d in dirs:
            for _ in range(radius):
                order.append(tile_index(cube))
                cube = (cube[0] + d[0], cube[1] + d[1], cube[2] + d[2])
    order.append(tile_index((0, 0, 0)))
    return tuple(order)


SPIRAL_TILE_ORDER = _spiral_tile_order()

# Device-side forms: spiral position -> tile index, its inverse permutation
# (tile index -> spiral position), and the A..R token values.
_SPIRAL_ORDER_ARR = jnp.array(SPIRAL_TILE_ORDER, dtype=jnp.int32)
_SPIRAL_POS_ARR = jnp.argsort(_SPIRAL_ORDER_ARR)
_SPIRAL_NUMBERS_ARR = jnp.array(SPIRAL_NUMBERS, dtype=jnp.uint8)


def make_layout(
    batch_size: int = 1,
    key: Key[Array, ""] | None = None,
    number_placement: Literal["random", "spiral"] = "random",
) -> BoardLayout:
    """Generate a batched random :class:`BoardLayout`.

    Terrain hexes and harbors are always shuffled uniformly. ``number_placement``
    selects how the number tokens are laid on the terrain: ``"random"`` (default)
    shuffles them uniformly, while ``"spiral"`` follows the rulebook's variable
    set-up -- tokens A..R placed alphabetically along a counterclockwise spiral
    from a corner toward the centre, skipping the desert -- the balanced
    placement commonly used in tournament play.
    """
    if number_placement not in ("random", "spiral"):
        raise ValueError(
            f"number_placement must be 'random' or 'spiral', got {number_placement!r}"
        )
    B = batch_size
    key = key if key is not None else jax.random.key(0)
    key, k1, k2, k3 = jax.random.split(key, 4)

    tile_number = jnp.array(
        [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
        dtype=jnp.uint8,
    )
    tile_number = jnp.tile(tile_number, (B, 1))
    keys = jax.random.split(k1, B)
    allocation_idxs = jax.vmap(lambda k: jax.random.permutation(k, 18))(keys)
    batch_idx = jnp.arange(B)[:, None]
    tile_number = tile_number[batch_idx, allocation_idxs]
    tile_number = jnp.concatenate(  # Concatenate desert tile with no number
        [tile_number, jnp.zeros((B, 1), dtype=jnp.uint8)], axis=1
    )

    tile_resource = jnp.zeros((B, N_TILES), dtype=jnp.uint8)
    tile_resource = tile_resource.at[:, :4].set(Tile.SHEEP.value)
    tile_resource = tile_resource.at[:, 4:8].set(Tile.WHEAT.value)
    tile_resource = tile_resource.at[:, 8:12].set(Tile.WOOD.value)
    tile_resource = tile_resource.at[:, 12:15].set(Tile.BRICK.value)
    tile_resource = tile_resource.at[:, 15:18].set(Tile.ORE.value)
    tile_resource = tile_resource.at[:, 18].set(Tile.DESERT.value)
    keys = jax.random.split(k2, B)
    allocation_idxs = jax.vmap(lambda k: jax.random.permutation(k, N_TILES))(keys)
    tile_resource = tile_resource[batch_idx, allocation_idxs]
    tile_number = tile_number[batch_idx, allocation_idxs]

    if number_placement == "spiral":
        # Deal the tokens A..R along the spiral, skipping the desert. ``token``
        # is the exclusive running count of non-desert tiles in spiral order --
        # the alphabetical token index each tile receives (clamped only to keep
        # the desert's discarded gather in bounds).
        not_desert = tile_resource[:, _SPIRAL_ORDER_ARR] != Tile.DESERT.value
        token = jnp.cumsum(not_desert, axis=1) - not_desert
        vals = _SPIRAL_NUMBERS_ARR[jnp.minimum(token, len(SPIRAL_NUMBERS) - 1)]
        on_spiral = jnp.where(not_desert, vals, 0).astype(jnp.uint8)
        tile_number = on_spiral[:, _SPIRAL_POS_ARR]

    port_allocation = jnp.array(
        [
            Port.SHEEP.value,
            Port.WHEAT.value,
            Port.WOOD.value,
            Port.BRICK.value,
            Port.ORE.value,
            Port.GENERAL.value,
            Port.GENERAL.value,
            Port.GENERAL.value,
            Port.GENERAL.value,
        ],
        dtype=jnp.uint8,
    )
    port_allocation = jnp.tile(port_allocation, (B, 1))
    keys = jax.random.split(k3, B)
    allocation_idxs = jax.vmap(lambda k: jax.random.permutation(k, N_PORTS))(keys)
    port_allocation = port_allocation[batch_idx, allocation_idxs]

    return BoardLayout(
        tile_resource=tile_resource,
        tile_number=tile_number,
        port_allocation=port_allocation,
    )


def desert_tile(tile_resource: TileResourceArray) -> UInt8[Array, "batch"]:
    """Per-lane index of the desert tile -- the robber's starting position.

    Settlrl starts the robber on the desert (where it blocks no production). The
    desert is the only ``Tile.DESERT`` tile; ``make_layout`` shuffles tile
    positions, so the location must be read off the generated ``tile_resource``.
    """
    return jnp.argmax(tile_resource == Tile.DESERT.value, axis=1).astype(jnp.uint8)
