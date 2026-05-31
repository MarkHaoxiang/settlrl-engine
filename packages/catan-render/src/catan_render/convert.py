"""Convert a ``catan_engine`` board into the renderer's wire model.

The engine stores a board as a ``(BoardLayout, BoardState)`` pair of batched JAX
arrays; the renderer needs a small, JSON-friendly :class:`BoardModel`. This
module bridges the two so the FastAPI server can serve a board produced by the
engine.

Geometry note: the engine assigns tile / vertex / edge indices implicitly by the
order it generates them in ``catan_engine.board.layout._generate_mappings``. The
engine exposes authoritative host-side index <-> cube lookups
(``vertex_cube`` / ``edge_cubes`` / ``tile_cube`` and the ``PORT_V`` map), so we
build the renderer's coordinate tables directly from those rather than
re-deriving the geometry. The asserts below pin our tables to the engine's
published counts.
"""

from typing import Literal

from catan_engine.board import Board
from catan_engine.board.layout import (
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    PORT_V,
    edge_cubes,
    tile_cube,
    vertex_cube,
    vertex_index,
)
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.port import Port
from catan_engine.board.tile import Tile

from .models import (
    BoardModel,
    BuildingModel,
    CubeModel,
    DevCardCounts,
    HexModel,
    PlayerModel,
    PortModel,
    ResourceCounts,
    RoadModel,
    Terrain,
    TileModel,
)

PortResource = Literal["sheep", "wheat", "wood", "brick", "ore"]

Cube = tuple[int, int, int]

# -- Resource ordering (single source) -------------------------------------------
# The engine indexes resource arrays (hands, costs, ports, monopoly / trade
# targets) by the ``Tile`` enum, skipping the non-resource desert. Derive the
# ordered resource names from the enum once and reuse everywhere the renderer
# indexes positionally (here and in ``catan_render.actions``); this is also the
# order ``models.ResourceCounts`` / ``PortModel`` declare their fields in.
_RESOURCE_NAMES: tuple[PortResource, ...] = tuple(
    t.name.lower() for t in Tile if t is not Tile.DESERT  # type: ignore[misc]
)

# Dev-card hand ordering, by the ``DevCard`` enum (matches ``DevCardCounts``
# fields). Used to read ``dev_hand`` positionally.
_DEV_CARD_NAMES: tuple[str, ...] = tuple(d.name.lower() for d in DevCard)

# -- Geometry (from catan_engine.board.layout's authoritative lookups) -----------

# Vertex index -> cube (q, r, s) corner coordinate.
VERTEX_COORDS: tuple[Cube, ...] = tuple(vertex_cube(i) for i in range(N_VERTICES))

# Edge index -> the two endpoint vertex indices (resolved back through the same
# cube coords the renderer uses for vertices).
EDGE_VERTICES: tuple[tuple[int, int], ...] = tuple(
    tuple(vertex_index(c) for c in edge_cubes(e)) for e in range(N_EDGES)  # type: ignore[misc]
)

# Tile index -> centre cube coord; ``TILE_COORDS`` is its axial (q, r) projection
# (pointy-top, hexagon of radius 2). tile_resource[i] / tile_number[i] -> here.
_TILE_CUBES: tuple[Cube, ...] = tuple(tile_cube(i) for i in range(N_TILES))
TILE_COORDS: tuple[tuple[int, int], ...] = tuple((q, r) for q, r, _ in _TILE_CUBES)

# Port index -> the cube coords of its two coastal vertices, from ``PORT_V``.
PORT_VERTEX_COORDS: tuple[tuple[Cube, Cube], ...] = tuple(
    (vertex_cube(int(a)), vertex_cube(int(b))) for a, b in PORT_V.tolist()
)

assert len(VERTEX_COORDS) == N_VERTICES
assert len(EDGE_VERTICES) == N_EDGES
assert len(_TILE_CUBES) == N_TILES
assert len(PORT_VERTEX_COORDS) == N_PORTS

# 2:1 resource ports map to their resource name; the 3:1 GENERAL port has none.
_RESOURCE_BY_PORT: dict[Port, PortResource] = {
    Port[name.upper()]: name for name in _RESOURCE_NAMES
}

_TERRAIN_BY_TILE: dict[Tile, Terrain] = {t: Terrain[t.name.lower()] for t in Tile}


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
        # Indexed positionally in enum order (see _RESOURCE_NAMES / _DEV_CARD_NAMES).
        res = player_resources[p]
        dev = dev_hand[p]
        players.append(
            PlayerModel(
                player=p,
                resource_cards=int(res.sum()),
                dev_cards=int(dev.sum()),
                victory_points=int(victory_points[p]),
                resources=ResourceCounts(
                    **{name: int(res[i]) for i, name in enumerate(_RESOURCE_NAMES)}
                ),
                dev_card_types=DevCardCounts(
                    **{name: int(dev[i]) for i, name in enumerate(_DEV_CARD_NAMES)}
                ),
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
