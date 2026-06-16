"""Enumerations and rulebook constants for the reference implementation.

These are defined independently of ``settlrl-engine`` (the whole point of the
oracle); see the page references to the official base-game rulebook. The
mapping between these enums and the engine's own enums lives in the engine's
test conversion layer, not here.
"""

from __future__ import annotations

from enum import Enum, IntEnum


class Resource(IntEnum):
    """The five tradeable resources (the desert produces nothing)."""

    SHEEP = 0  # wool
    WHEAT = 1  # grain
    WOOD = 2  # lumber
    BRICK = 3  # from hills
    ORE = 4  # from mountains


# The canonical resource order, handy for deterministic iteration.
RESOURCES: tuple[Resource, ...] = tuple(Resource)
N_RESOURCES = len(RESOURCES)


class DevCard(IntEnum):
    """The five development-card kinds (rulebook: knight, 3 progress, victory point)."""

    KNIGHT = 0
    ROAD_BUILDING = 1
    YEAR_OF_PLENTY = 2
    MONOPOLY = 3
    VICTORY_POINT = 4


DEV_CARDS: tuple[DevCard, ...] = tuple(DevCard)


class PortType(Enum):
    """A harbour: a 2:1 port for one resource, or the generic 3:1 port."""

    SHEEP = Resource.SHEEP
    WHEAT = Resource.WHEAT
    WOOD = Resource.WOOD
    BRICK = Resource.BRICK
    ORE = Resource.ORE
    GENERIC = None  # 3:1, any resource


class Building(IntEnum):
    """What occupies a built vertex."""

    SETTLEMENT = 1
    CITY = 2


class Phase(Enum):
    """The step of the turn/game that decides which actions are legal."""

    SETUP_SETTLEMENT = "setup_settlement"  # place a free starting settlement
    SETUP_ROAD = "setup_road"  # place the road next to it
    ROLL = "roll"  # must roll dice (may play a Knight first)
    DISCARD = "discard"  # players with >7 cards discard half (after a 7)
    MOVE_ROBBER = "move_robber"  # current player moves robber and steals
    MAIN = "main"  # build / trade / play dev card / end turn
    TRADE_RESPONSE = "trade_response"  # the proposed-to player accepts or rejects
    GAME_OVER = "game_over"  # a player has reached VICTORY_POINTS_TO_WIN


# --- Rulebook constants ----------------------------------------------------

N_PLAYERS = 4

# Build costs as per-resource maps (Building Costs card).
ROAD_COST: dict[Resource, int] = {Resource.WOOD: 1, Resource.BRICK: 1}
SETTLEMENT_COST: dict[Resource, int] = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
    Resource.SHEEP: 1,
    Resource.WHEAT: 1,
}
CITY_COST: dict[Resource, int] = {Resource.ORE: 3, Resource.WHEAT: 2}
DEV_CARD_COST: dict[Resource, int] = {
    Resource.ORE: 1,
    Resource.SHEEP: 1,
    Resource.WHEAT: 1,
}

# Each resource starts with this many cards in the bank.
BANK_INITIAL = 19

# A full, unshuffled development deck (14 knights, 2 each progress, 5 VP) = 25.
DEV_CARD_COUNTS: dict[DevCard, int] = {
    DevCard.KNIGHT: 14,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
    DevCard.VICTORY_POINT: 5,
}

# Per-player piece supply.
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4

VICTORY_POINTS_TO_WIN = 10
LONGEST_ROAD_MIN = 5  # need a 5+ segment road to hold the card
LARGEST_ARMY_MIN = 3  # need 3+ played knights to hold the card
ROBBER_DISCARD_LIMIT = 7  # players with strictly more than this discard half
ROAD_BUILDING_FREE_ROADS = 2
