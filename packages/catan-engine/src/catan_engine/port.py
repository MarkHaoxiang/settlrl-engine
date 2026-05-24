from enum import IntEnum

from catan_engine.tile import Tile


class Port(IntEnum):
    SHEEP = Tile.SHEEP.value
    WHEAT = Tile.WHEAT.value
    WOOD = Tile.WOOD.value
    BRICK = Tile.BRICK.value
    ORE = Tile.ORE.value
    GENERAL = 5  # 3:1 port

    def __str__(self) -> str:
        return ("SHP", "WHT", "WOD", "BRK", "ORE", "3:1")[self]
