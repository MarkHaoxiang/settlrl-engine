import jax
import jax.numpy as jnp

from catan_engine.port import Port
from catan_engine.tile import Tile

N_TILES = 19
N_VERTICES = 54


def _generate_mappings() -> tuple[jax.Array, jax.Array]:
    tile_vertex_mapping: list[list[int]] = []
    vertices = {}

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

    return (
        jnp.array(tile_vertex_mapping, dtype=jnp.uint8),
        jnp.array(ports, dtype=jnp.uint8),
    )


_tile_vertex_map, _port_vertices_map = _generate_mappings()
assert _tile_vertex_map.shape == (N_TILES, 6)
assert _port_vertices_map.shape == (9, 2)


def make_board(
    batch_size: int = 1,
    key: jax.Array | None = None,
) -> dict[str, jax.Array]:
    B = batch_size
    key = key if key is not None else jax.random.key(0)
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Create board tiles
    # Generate tile numbers
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
    # Assign resources to tiles
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

    # Generate ports
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
    allocation_idxs = jnp.stack([jax.random.permutation(k, 9) for k in keys])
    port_allocation = port_allocation[batch_idx, allocation_idxs]

    return {
        "tile_resource": tile_resource,
        "tile_number": tile_number,
        "port_allocation": port_allocation,
    }
