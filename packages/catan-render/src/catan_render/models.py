from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class Terrain(StrEnum):
    wheat = "wheat"
    sheep = "sheep"
    wood = "wood"
    ore = "ore"
    brick = "brick"
    desert = "desert"


class HexModel(BaseModel):
    """An axial (pointy-top) tile coordinate."""

    q: int
    r: int


class TileModel(BaseModel):
    q: int
    r: int
    terrain: Terrain
    number: int | None = None


class CubeModel(BaseModel):
    """A board vertex, in the engine's cube coordinates (q + r + s = ±1)."""

    q: int
    r: int
    s: int


class BuildingModel(BaseModel):
    """A settlement or city sitting on a vertex, owned by a 0-indexed player."""

    cube: CubeModel
    player: int
    kind: Literal["settlement", "city"]


class RoadModel(BaseModel):
    """A road along an edge between two vertices, owned by a 0-indexed player."""

    a: CubeModel
    b: CubeModel
    player: int


class PortModel(BaseModel):
    """A harbour spanning two coastal vertices.

    ``resource`` is the resource of a 2:1 port, or ``None`` for a 3:1 general
    port.
    """

    a: CubeModel
    b: CubeModel
    resource: Literal["sheep", "wheat", "wood", "brick", "ore"] | None


class PlayerModel(BaseModel):
    """Summary stats for one (0-indexed) player, shown in the corner panels."""

    player: int
    resource_cards: int  # total resource cards in hand
    dev_cards: int  # total unplayed development cards in hand
    victory_points: int  # building victory points (settlement=1, city=2)


class BoardModel(BaseModel):
    tiles: list[TileModel]
    buildings: list[BuildingModel] = []
    roads: list[RoadModel] = []
    ports: list[PortModel] = []
    players: list[PlayerModel] = []
    # Tile the robber currently occupies (axial coordinate), if any.
    robber: HexModel | None = None
