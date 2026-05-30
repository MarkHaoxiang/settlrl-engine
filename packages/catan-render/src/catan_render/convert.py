"""Convert a ``catan_engine`` board into the renderer's wire model.

The engine stores a board as a ``(BoardLayout, BoardState)`` pair of batched JAX
arrays; the renderer needs a small, JSON-friendly :class:`BoardModel`. This
module bridges the two so the FastAPI server can serve a board produced by the
engine.

Geometry note: the engine assigns tile / vertex / edge indices implicitly by the
order it generates them in ``catan_engine.layout._generate_mappings``. Those maps
are private, so we reproduce the same deterministic construction here to recover
the index -> coordinate tables the renderer needs. The asserts below pin our
reconstruction to the engine's published counts.
"""

from typing import Literal

from catan_engine.board import Board
from catan_engine.layout import N_EDGES, N_PORTS, N_TILES, N_VERTICES
from catan_engine.port import Port
from catan_engine.tile import Tile

from .models import (
    BoardModel,
    BuildingModel,
    CubeModel,
    HexModel,
    PlayerModel,
    PortModel,
    RoadModel,
    Terrain,
    TileModel,
)

PortResource = Literal["sheep", "wheat", "wood", "brick", "ore"]

# -- Geometry (mirrors catan_engine.layout) --------------------------------

# Directions from a tile centre to its six corner vertices, and the three
# edge-difference vectors, exactly as in catan_engine.layout.
_VERTEX_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_EDGE_DIFFS = ((1, 1, 0), (1, 0, 1), (0, 1, 1))

Cube = tuple[int, int, int]


def _build_geometry() -> tuple[list[Cube], list[Cube], list[tuple[int, int]]]:
    """Recreate the engine's tile / vertex / edge index tables.

    Returns:
        tile_coords: tile index -> axial (q, r, s) tile centre.
        vertex_coords: vertex index -> cube (q, r, s) corner coordinate.
        edge_vertices: edge index -> (vertex index, vertex index).
    """
    tile_coords: list[Cube] = []
    vertices: dict[Cube, int] = {}
    i = 0
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(s) <= 2:
                tile_coords.append((q, r, s))
                for dq, dr, ds in _VERTEX_DIRS:
                    v = (q + dq, r + dr, s + ds)
                    if v not in vertices:
                        vertices[v] = i
                        i += 1

    edge_set: set[tuple[int, int]] = set()
    for cube, idx in vertices.items():
        if sum(cube) == 1:
            for dq, dr, ds in _EDGE_DIFFS:
                other = (cube[0] - dq, cube[1] - dr, cube[2] - ds)
                if other in vertices:
                    a, b = idx, vertices[other]
                    edge_set.add((min(a, b), max(a, b)))
    edge_vertices = sorted(edge_set)

    vertex_coords: list[Cube] = [(0, 0, 0)] * len(vertices)
    for cube, idx in vertices.items():
        vertex_coords[idx] = cube

    return tile_coords, vertex_coords, edge_vertices


_TILE_CUBES, VERTEX_COORDS, EDGE_VERTICES = _build_geometry()
assert len(_TILE_CUBES) == N_TILES
assert len(VERTEX_COORDS) == N_VERTICES
assert len(EDGE_VERTICES) == N_EDGES

# Axial (q, r) coordinate of every tile, in the engine's tile-index order
# (pointy-top, hexagon of radius 2). tile_resource[i] / tile_number[i] -> here.
TILE_COORDS: tuple[tuple[int, int], ...] = tuple((q, r) for q, r, _ in _TILE_CUBES)

# The two coastal vertices of each port, in port-index order. Copied verbatim
# from catan_engine.layout._generate_mappings; port_allocation[i] -> here.
PORT_VERTEX_COORDS: tuple[tuple[Cube, Cube], ...] = (
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
assert len(PORT_VERTEX_COORDS) == N_PORTS

_RESOURCE_BY_PORT: dict[Port, PortResource] = {
    Port.SHEEP: "sheep",
    Port.WHEAT: "wheat",
    Port.WOOD: "wood",
    Port.BRICK: "brick",
    Port.ORE: "ore",
}

_TERRAIN_BY_TILE: dict[Tile, Terrain] = {
    Tile.SHEEP: Terrain.sheep,
    Tile.WHEAT: Terrain.wheat,
    Tile.WOOD: Terrain.wood,
    Tile.BRICK: Terrain.brick,
    Tile.ORE: Terrain.ore,
    Tile.DESERT: Terrain.desert,
}


def _cube(coord: Cube) -> CubeModel:
    q, r, s = coord
    return CubeModel(q=q, r=r, s=s)


def board_to_model(board: Board, batch_index: int = 0) -> BoardModel:
    """Render one game from a (possibly batched) engine board.

    Converts both the static layout (tiles) and the mutable state (settlements,
    cities, roads, robber) for game ``batch_index`` (default 0).

    Args:
        board: The engine ``(BoardLayout, BoardState)`` pair.
        batch_index: Which game in the batch to convert.

    Returns:
        A :class:`BoardModel` ready to serialise to JSON.
    """
    layout, state = board

    # -- Tiles (static layout) ---------------------------------------------
    resources = layout.tile_resource[batch_index]
    numbers = layout.tile_number[batch_index]
    tiles: list[TileModel] = []
    for i, (q, r) in enumerate(TILE_COORDS):
        tile = Tile(int(resources[i]))
        number = int(numbers[i])
        tiles.append(
            TileModel(
                q=q,
                r=r,
                terrain=_TERRAIN_BY_TILE[tile],
                # The desert carries no number token.
                number=None if tile is Tile.DESERT else number,
            )
        )

    # -- Buildings, roads, robber (mutable state) --------------------------
    # vertex_owner / edge_road store player + 1 (0 = empty); convert to
    # 0-indexed players. vertex_type: 1 = settlement, 2 = city.
    vertex_owner = state.vertex_owner[batch_index]
    vertex_type = state.vertex_type[batch_index]
    edge_road = state.edge_road[batch_index]

    buildings: list[BuildingModel] = []
    for v, coord in enumerate(VERTEX_COORDS):
        owner = int(vertex_owner[v])
        if owner == 0:
            continue
        kind: Literal["settlement", "city"] = (
            "city" if int(vertex_type[v]) == 2 else "settlement"
        )
        buildings.append(BuildingModel(cube=_cube(coord), player=owner - 1, kind=kind))

    roads: list[RoadModel] = []
    for e, (v1, v2) in enumerate(EDGE_VERTICES):
        owner = int(edge_road[e])
        if owner == 0:
            continue
        roads.append(
            RoadModel(a=_cube(VERTEX_COORDS[v1]), b=_cube(VERTEX_COORDS[v2]), player=owner - 1)
        )

    robber_q, robber_r = TILE_COORDS[int(state.robber[batch_index])]
    robber = HexModel(q=robber_q, r=robber_r)

    # -- Ports (static layout) ---------------------------------------------
    # GENERAL is a 3:1 port (resource = None); the rest are 2:1 resource ports.
    port_allocation = layout.port_allocation[batch_index]
    ports: list[PortModel] = []
    for i, (coord_a, coord_b) in enumerate(PORT_VERTEX_COORDS):
        port = Port(int(port_allocation[i]))
        ports.append(
            PortModel(
                a=_cube(coord_a),
                b=_cube(coord_b),
                resource=_RESOURCE_BY_PORT.get(port),
            )
        )

    # -- Players (mutable state) -------------------------------------------
    # player_resources: (players, resources) -> total cards in hand.
    # dev_hand: (players, dev card types) -> unplayed dev cards.
    # victory_points: (players,) building points only.
    player_resources = state.player_resources[batch_index]
    dev_hand = state.dev_hand[batch_index]
    victory_points = state.victory_points[batch_index]
    players: list[PlayerModel] = []
    for p in range(player_resources.shape[0]):
        players.append(
            PlayerModel(
                player=p,
                resource_cards=int(player_resources[p].sum()),
                dev_cards=int(dev_hand[p].sum()),
                victory_points=int(victory_points[p]),
            )
        )

    return BoardModel(
        tiles=tiles,
        buildings=buildings,
        roads=roads,
        ports=ports,
        players=players,
        robber=robber,
    )
