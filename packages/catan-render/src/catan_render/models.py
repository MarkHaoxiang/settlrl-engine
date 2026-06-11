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


class ResourceCounts(BaseModel):
    """Resource cards in hand, by type (Tile resource order)."""

    sheep: int
    wheat: int
    wood: int
    brick: int
    ore: int


class DevCardCounts(BaseModel):
    """Unplayed development cards in hand, by type (DevCard order)."""

    knight: int
    road_building: int
    year_of_plenty: int
    monopoly: int
    victory_point: int


class PlayerModel(BaseModel):
    """Summary stats for one (0-indexed) player, shown in the corner panels."""

    player: int
    resource_cards: int  # total resource cards in hand
    dev_cards: int  # total unplayed development cards in hand
    victory_points: int  # building victory points (settlement=1, city=2)
    resources: ResourceCounts  # per-type breakdown of resource_cards
    dev_card_types: DevCardCounts  # per-type breakdown of dev_cards


class BankModel(BaseModel):
    """Cards left in the supply: resource stacks plus the development deck."""

    resources: ResourceCounts
    dev_cards: int


class BoardModel(BaseModel):
    tiles: list[TileModel]
    buildings: list[BuildingModel] = []
    roads: list[RoadModel] = []
    ports: list[PortModel] = []
    players: list[PlayerModel] = []
    # Tile the robber currently occupies (axial coordinate), if any.
    robber: HexModel | None = None
    bank: BankModel | None = None


class EdgeModel(BaseModel):
    """A board edge between two vertices (in cube coordinates)."""

    a: CubeModel
    b: CubeModel


class ActionModel(BaseModel):
    """One legal move for the acting player, decoded from the AEC flat action set.

    ``flat`` is the engine's flat action index (post it back to apply the move).
    ``type`` is the lowercased :class:`ActionType` name. Depending on the type, at
    most one geometry/resource group below is populated; the rest stay ``None``.
    """

    flat: int
    type: str
    label: str
    # Placement target: a vertex (settlement/city), an edge (road), or a tile
    # (robber / knight, with the optional victim player to steal from).
    vertex: CubeModel | None = None
    edge: EdgeModel | None = None
    tile: HexModel | None = None
    victim: int | None = None
    # Resource choices: monopoly (one), year-of-plenty (two), maritime trade,
    # and the domestic trade proposal (give/receive plus the proposed-to partner).
    resource: str | None = None
    resources: list[str] | None = None
    give: str | None = None
    receive: str | None = None
    partner: int | None = None


class GameStatusModel(BaseModel):
    """Turn-flow snapshot for the live game."""

    phase: str
    current_player: int
    acting_player: int
    dice_roll: int
    has_rolled: bool
    your_turn: bool
    terminal: bool
    winner: int | None = None
    # What controls each seat: "human" or a bot kind (catan-agents policy name).
    seats: list[str] = []


class BotMoveModel(BaseModel):
    """One just-played bot move: the seat that acted and the decoded action."""

    player: int
    action: ActionModel


class LogEntryModel(BaseModel):
    """One line of the game's chat / log.

    ``player`` is the seat the line belongs to (``None``: a spectator's chat
    message). Move entries carry the ``action_type`` (the client maps it to an
    icon) and a short text (the action label, or ``"rolled N"`` for rolls).
    """

    id: int
    kind: Literal["move", "chat", "win"]
    player: int | None = None
    action_type: str | None = None
    text: str = ""


class ReplayStateModel(BaseModel):
    """The Replay view's snapshot after ``move`` of ``n_moves`` moves.

    ``log`` holds the moves played up to that point (the win line appears only
    at the final move); ``winner`` / ``seats`` describe the whole record.
    """

    move: int
    n_moves: int
    board: BoardModel
    log: list[LogEntryModel] = []
    winner: int | None = None
    seats: list[str] | None = None


class PlayerBeliefModel(BaseModel):
    """The observer's proven bounds on one player's hand (``lo == hi``: exact)."""

    player: int
    res_lo: ResourceCounts
    res_hi: ResourceCounts


class BeliefModel(BaseModel):
    """Card counting from one human seat's perspective.

    Everything here is derivable from public information (the engine's belief
    tracker), so showing it to that seat never leaks hidden state. The
    observer's own row is omitted — their hand is already on screen.
    """

    observer: int
    players: list[PlayerBeliefModel]


class GameModel(BaseModel):
    """Everything the Play view needs after a move: board + status + legal moves."""

    board: BoardModel
    status: GameStatusModel
    actions: list[ActionModel] = []
    # Set on POST /api/game/bot responses: the move that endpoint just played.
    bot_move: BotMoveModel | None = None
    # The game's chat / log (moves, chat messages, the win), oldest first.
    log: list[LogEntryModel] = []
    # Card counting for the hand-panel seat; None with no human seats.
    belief: BeliefModel | None = None
