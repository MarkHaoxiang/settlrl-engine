"""catan-reference: a plain-Python gold-standard implementation of Catan.

Written from the official base-game rulebook, prioritising correctness and
clarity over speed. It exists to differentially test ``catan-engine``: drive
both with the same action stream and compare the resulting states.
"""

from catan_reference.board import (
    N_EDGES,
    N_TILES,
    N_VERTICES,
    Layout,
    Port,
)
from catan_reference.game import (
    Action,
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BuyDevelopmentCard,
    Discard,
    EndTurn,
    Game,
    MaritimeTrade,
    MoveRobber,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    Player,
    Roll,
    SetupRoad,
    SetupSettlement,
)
from catan_reference.types import (
    Building,
    DevCard,
    Phase,
    PortType,
    Resource,
)

__all__ = [
    "Action",
    "Layout",
    "Port",
    "N_TILES",
    "N_VERTICES",
    "N_EDGES",
    "Game",
    "Player",
    "Resource",
    "DevCard",
    "PortType",
    "Building",
    "Phase",
    "SetupSettlement",
    "SetupRoad",
    "Roll",
    "Discard",
    "MoveRobber",
    "BuildRoad",
    "BuildSettlement",
    "BuildCity",
    "BuyDevelopmentCard",
    "PlayKnight",
    "PlayRoadBuilding",
    "PlayYearOfPlenty",
    "PlayMonopoly",
    "MaritimeTrade",
    "EndTurn",
]
