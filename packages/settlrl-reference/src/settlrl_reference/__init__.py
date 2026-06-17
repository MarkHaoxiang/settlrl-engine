"""settlrl-reference: a plain-Python gold-standard implementation of Settlrl.

Written from the official base-game rulebook, prioritising correctness and
clarity over speed. It exists to differentially test ``settlrl-engine``: drive
both with the same action stream and compare the resulting states.
"""

from settlrl_reference.belief import Belief
from settlrl_reference.board import (
    N_EDGES,
    N_TILES,
    N_VERTICES,
    Layout,
    Port,
)
from settlrl_reference.game import (
    AcceptTrade,
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
    Player,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    ProposeTrade,
    RejectTrade,
    Roll,
    SetupRoad,
    SetupSettlement,
)
from settlrl_reference.types import (
    RESOURCES,
    Building,
    DevCard,
    Phase,
    PortType,
    Resource,
)

__all__ = [
    "N_EDGES",
    "N_TILES",
    "N_VERTICES",
    "RESOURCES",
    "AcceptTrade",
    "Action",
    "Belief",
    "BuildCity",
    "BuildRoad",
    "BuildSettlement",
    "Building",
    "BuyDevelopmentCard",
    "DevCard",
    "Discard",
    "EndTurn",
    "Game",
    "Layout",
    "MaritimeTrade",
    "MoveRobber",
    "Phase",
    "PlayKnight",
    "PlayMonopoly",
    "PlayRoadBuilding",
    "PlayYearOfPlenty",
    "Player",
    "Port",
    "PortType",
    "ProposeTrade",
    "RejectTrade",
    "Resource",
    "Roll",
    "SetupRoad",
    "SetupSettlement",
]
