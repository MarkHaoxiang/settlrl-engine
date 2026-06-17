"""settlrl-reference: a plain-Python gold-standard implementation of Settlrl.

Written from the official base-game rulebook, prioritising correctness and
clarity over speed. It exists to differentially test ``settlrl-engine``: drive
both with the same action stream and compare the resulting states.
"""

from settlrl_game.reference.belief import Belief
from settlrl_game.reference.board import (
    N_EDGES,
    N_TILES,
    N_VERTICES,
    Layout,
    Port,
    desert_tile,
    random_layout,
)
from settlrl_game.reference.chance import draw_dev_card, roll_dice, steal
from settlrl_game.reference.game import (
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
from settlrl_game.reference.types import (
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
    "desert_tile",
    "draw_dev_card",
    "random_layout",
    "roll_dice",
    "steal",
]
