"""Convert a ``settlrl_game.reference`` game into the renderer's wire model.

Bridges a reference ``Game`` (its ``Layout`` plus the live occupancy/hands) to
the JSON-friendly :class:`BoardModel`. All geometry comes from
``settlrl_game.reference.board``'s cube lookups; the reference's enum order
(``Resource`` / ``DevCard``) is the order the wire models declare their fields,
so positional reads line up.
"""

from __future__ import annotations

from typing import Literal

import settlrl_game.reference as ref
from settlrl_game.reference import board as rb

from settlrl_game.models import (
    BankModel,
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

# Resource / dev-card field order, from the reference enums (matches the wire
# ResourceCounts / DevCardCounts field order). Reused wherever the renderer
# indexes positionally (here and in api.flat).
_RESOURCE_NAMES: tuple[PortResource, ...] = tuple(r.name.lower() for r in ref.RESOURCES)  # type: ignore[misc]
_DEV_CARD_NAMES: tuple[str, ...] = tuple(d.name.lower() for d in ref.DevCard)

# -- Geometry (from settlrl_game.reference.board's cube lookups) ------------------

VERTEX_COORDS: tuple[Cube, ...] = tuple(rb.vertex_cube(v) for v in range(rb.N_VERTICES))
EDGE_VERTICES: tuple[tuple[int, int], ...] = tuple(
    rb.edge_vertices(e) for e in range(rb.N_EDGES)
)
# Tile index -> centre axial (q, r) projection (pointy-top hexagon of radius 2).
TILE_COORDS: tuple[tuple[int, int], ...] = tuple(
    (q, r) for q, r, _ in (rb.tile_cube(t) for t in range(rb.N_TILES))
)

_TERRAIN_BY_RESOURCE: dict[ref.Resource | None, Terrain] = {
    None: Terrain.desert,
    **{r: Terrain[r.name.lower()] for r in ref.RESOURCES},
}


def _cube(coord: Cube) -> CubeModel:
    q, r, s = coord
    return CubeModel(q=q, r=r, s=s)


def _port_resource(port: ref.Port) -> PortResource | None:
    """A 2:1 port's resource name, or ``None`` for the generic 3:1 port."""
    if port.type is ref.PortType.GENERIC:
        return None
    resource: ref.Resource = port.type.value
    return _RESOURCE_NAMES[int(resource)]


def board_to_model(game: ref.Game) -> BoardModel:
    """Render a reference ``game``: its static layout plus the mutable
    occupancy / robber / hands."""
    layout = game.layout

    tiles = [
        TileModel(
            q=q,
            r=r,
            terrain=_TERRAIN_BY_RESOURCE[layout.tile_resource[t]],
            number=layout.tile_number[t] or None,  # the desert carries no token
        )
        for t, (q, r) in enumerate(TILE_COORDS)
    ]

    buildings = [
        BuildingModel(
            cube=_cube(VERTEX_COORDS[v]),
            player=player,
            kind="city" if kind is ref.Building.CITY else "settlement",
        )
        for v, (player, kind) in sorted(game.buildings.items())
    ]
    roads = [
        RoadModel(
            a=_cube(VERTEX_COORDS[EDGE_VERTICES[e][0]]),
            b=_cube(VERTEX_COORDS[EDGE_VERTICES[e][1]]),
            player=player,
        )
        for e, player in sorted(game.roads.items())
    ]

    robber_q, robber_r = TILE_COORDS[game.robber]
    ports = [
        PortModel(
            a=_cube(VERTEX_COORDS[port.vertices[0]]),
            b=_cube(VERTEX_COORDS[port.vertices[1]]),
            resource=_port_resource(port),
        )
        for port in layout.ports
    ]

    players = [
        PlayerModel(
            player=p,
            resource_cards=sum(pl.resources.values()),
            dev_cards=sum(pl.dev_cards.values()),
            victory_points=game.building_vp(p),
            knights_played=pl.knights_played,
            longest_road=game.longest_road_owner == p,
            largest_army=game.largest_army_owner == p,
            resources=ResourceCounts(
                **{
                    n: pl.resources[r]
                    for r, n in zip(ref.RESOURCES, _RESOURCE_NAMES, strict=True)
                }
            ),
            dev_card_types=DevCardCounts(
                **{
                    n: pl.dev_cards[d]
                    for d, n in zip(ref.DevCard, _DEV_CARD_NAMES, strict=True)
                }
            ),
        )
        for p, pl in enumerate(game.players)
    ]

    bank = BankModel(
        resources=ResourceCounts(
            **{
                n: game.bank(r)
                for r, n in zip(ref.RESOURCES, _RESOURCE_NAMES, strict=True)
            }
        ),
        dev_cards=sum(game.dev_deck.values()),
    )

    return BoardModel(
        tiles=tiles,
        buildings=buildings,
        roads=roads,
        ports=ports,
        players=players,
        robber=HexModel(q=robber_q, r=robber_r),
        bank=bank,
    )
