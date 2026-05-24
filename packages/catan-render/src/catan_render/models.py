from enum import StrEnum
from pydantic import BaseModel


class Terrain(StrEnum):
    wheat = "wheat"
    sheep = "sheep"
    wood = "wood"
    ore = "ore"
    brick = "brick"
    desert = "desert"


class TileModel(BaseModel):
    q: int
    r: int
    terrain: Terrain
    number: int | None = None


class BoardModel(BaseModel):
    tiles: list[TileModel]
