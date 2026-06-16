from enum import IntEnum


class Tile(IntEnum):
    SHEEP = 0
    WHEAT = 1
    WOOD = 2
    BRICK = 3
    ORE = 4
    DESERT = 5

    def __str__(self) -> str:
        return ("SHP", "WHT", "WOD", "BRK", "ORE", "DST")[self]
