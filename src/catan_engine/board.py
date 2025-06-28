import torch
from enum import IntEnum

N_TILES = 19
N_VERTICES = 54


def _generate_mappings():
    tile_vertex_mapping: list[list] = []
    vertices = {}

    vertex_dirs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    port_vertices = (
        ((3, 0, -2), (2, 0, -3)),
        ((-3, -2, 0), (-2, 3, 0)),
        ((0, -2, 3), (0, -3, 2)),
        ((-1, -1, 3), (-2, -1, 2)),
        ((-2, 1, 2), (-3, 1, 1)),
        ((1, -3, 1), (2, -2, 1)),
        ((2, -2, -1), (3, -1, 1)),
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
        torch.tensor(tile_vertex_mapping, dtype=torch.uint8),
        torch.tensor(ports, dtype=torch.uint8),
    )


_tile_vertex_map, _port_vertices_map = _generate_mappings()
assert _tile_vertex_map.shape == (N_TILES, 6)
assert _port_vertices_map.shape == (9, 2)


class Tile(IntEnum):
    SHEEP = 0
    WHEAT = 1
    WOOD = 2
    BRICK = 3
    ORE = 4
    DESERT = 5


class Port(IntEnum):
    SHEEP = Tile.SHEEP.value
    WHEAT = Tile.WHEAT.value
    WOOD = Tile.WOOD.value
    BRICK = Tile.BRICK.value
    ORE = Tile.ORE.value
    GENERAL = 5  # 3:1 Port


def make_board(
    batch_size: int = 1,
    rng: torch.Generator | None = None,
    device: torch.device = torch.device("cpu"),
):
    B = batch_size
    rng = rng if rng is not None else torch.Generator()

    # Create board tiles
    # Generate tile numbers
    tile_number = torch.tensor(
        [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
        dtype=torch.uint8,
        device=device,
    ).repeat(B, 1)
    allocation_idxs = torch.stack(
        [torch.randperm(N_TILES, generator=rng) for _ in range(B)]
    )
    tile_number = torch.gather(tile_number, 1, allocation_idxs)
    tile_number = torch.concatenate(  # Concatenate desert tile with no number
        [tile_number, torch.zeros(B, 1, dtype=torch.uint8, device=device)], dim=1
    )
    # Assign resources to tiles
    tile_resource = torch.zeros(batch_size, N_TILES, dtype=torch.uint8, device=device)
    tile_resource[:, :4] = Tile.SHEEP.value
    tile_resource[:, 4:8] = Tile.WHEAT.value
    tile_resource[:, 8:12] = Tile.WOOD.value
    tile_resource[:, 12:15] = Tile.BRICK.value
    tile_resource[:, 15:18] = Tile.ORE.value
    tile_resource[:, 18] = Tile.DESERT.value
    allocation_idxs = torch.stack(
        [torch.randperm(N_TILES, generator=rng) for _ in range(B)]
    )
    tile_resource = torch.gather(tile_resource, 1, allocation_idxs)
    tile_number = torch.gather(tile_number, 1, allocation_idxs)

    # Generate ports
    port_allocation = torch.tensor(
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
        dtype=torch.uint8,
        device=device,
    ).repeat(B, 1)
    allocation_idxs = torch.stack([torch.randperm(9, generator=rng) for _ in range(B)])
