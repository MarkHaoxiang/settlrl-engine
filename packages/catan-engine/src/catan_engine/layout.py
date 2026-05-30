from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, UInt8

from catan_engine.port import Port
from catan_engine.tile import Tile

N_TILES = 19
N_VERTICES = 54
N_EDGES = 72  # V - E + F = 2, F = N_TILES + 1 outer face
N_PORTS = 9

# Maximum number of edges/neighbours/tiles incident to a single vertex.
MAX_VERTEX_DEGREE = 3
# Sentinel stored in the padded incidence maps to mean "no entry".
NO_INDEX = 255

TileResourceArray = UInt8[Array, f"batch tiles={N_TILES}"]
TileNumberArray = UInt8[Array, f"batch tiles={N_TILES}"]
PortAllocationArray = UInt8[Array, f"batch ports={N_PORTS}"]


class BoardLayout(NamedTuple):
    """Immutable board geometry: tile resources, number tokens, and port types."""

    tile_resource: jax.Array
    tile_number: jax.Array
    port_allocation: jax.Array


# Two vertices share an edge iff one cube coord sums to +1, the other to -1,
# and their difference is one of these (matches the renderer's geometry).
_EDGE_DIFFS = ((1, 1, 0), (1, 0, 1), (0, 1, 1))


def _generate_mappings() -> tuple[jax.Array, ...]:
    tile_vertex_mapping: list[list[int]] = []
    vertices: dict[tuple[int, int, int], int] = {}

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

    # Axial coordinates for the vertices of a Catan board
    i = 0
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(s) <= 2:
                # Valid tile
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

    n_v = len(vertices)
    pad = [NO_INDEX] * MAX_VERTEX_DEGREE

    # vertex -> incident edge indices, and the vertex across each of those edges.
    vertex_edges = [pad.copy() for _ in range(n_v)]
    vertex_neighbours = [pad.copy() for _ in range(n_v)]
    for e, (a, b) in enumerate(edge_list):
        for v, w in ((a, b), (b, a)):
            slot = vertex_edges[v].index(NO_INDEX)
            vertex_edges[v][slot] = e
            vertex_neighbours[v][slot] = w

    # vertex -> incident tiles (inverse of tile_vertex_mapping).
    vertex_tiles = [pad.copy() for _ in range(n_v)]
    for t, corners in enumerate(tile_vertex_mapping):
        for v in corners:
            vertex_tiles[v][vertex_tiles[v].index(NO_INDEX)] = t

    # vertex -> port index (NO_INDEX if the vertex is not a port vertex).
    vertex_port = [NO_INDEX] * n_v
    for p, (v1, v2) in enumerate(ports):
        vertex_port[v1] = p
        vertex_port[v2] = p

    return (
        jnp.array(tile_vertex_mapping, dtype=jnp.uint8),
        jnp.array(ports, dtype=jnp.uint8),
        jnp.array(edge_list, dtype=jnp.uint8),
        jnp.array(vertex_edges, dtype=jnp.uint8),
        jnp.array(vertex_neighbours, dtype=jnp.uint8),
        jnp.array(vertex_tiles, dtype=jnp.uint8),
        jnp.array(vertex_port, dtype=jnp.uint8),
    )


(
    _tile_vertex_map,
    _port_vertices_map,
    _edge_vertex_map,
    _vertex_edge_map,
    _vertex_neighbour_map,
    _vertex_tile_map,
    _vertex_port_map,
) = _generate_mappings()
assert _tile_vertex_map.shape == (N_TILES, 6)
assert _port_vertices_map.shape == (N_PORTS, 2)
assert _edge_vertex_map.shape == (N_EDGES, 2)
assert _vertex_edge_map.shape == (N_VERTICES, MAX_VERTEX_DEGREE)
assert _vertex_neighbour_map.shape == (N_VERTICES, MAX_VERTEX_DEGREE)
assert _vertex_tile_map.shape == (N_VERTICES, MAX_VERTEX_DEGREE)
assert _vertex_port_map.shape == (N_VERTICES,)


def make_layout(
    batch_size: int = 1,
    key: jax.Array | None = None,
) -> BoardLayout:
    B = batch_size
    key = key if key is not None else jax.random.key(0)
    key, k1, k2, k3 = jax.random.split(key, 4)

    tile_number = jnp.array(
        [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
        dtype=jnp.uint8,
    )
    tile_number = jnp.tile(tile_number, (B, 1))
    keys = jax.random.split(k1, B)
    allocation_idxs = jnp.stack([jax.random.permutation(k, 18) for k in keys])
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
    allocation_idxs = jnp.stack([jax.random.permutation(k, N_TILES) for k in keys])
    tile_resource = tile_resource[batch_idx, allocation_idxs]
    tile_number = tile_number[batch_idx, allocation_idxs]

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
    allocation_idxs = jnp.stack([jax.random.permutation(k, N_PORTS) for k in keys])
    port_allocation = port_allocation[batch_idx, allocation_idxs]

    return BoardLayout(
        tile_resource=tile_resource,
        tile_number=tile_number,
        port_allocation=port_allocation,
    )
