"""A complete single game of Catan, implemented from the base-game rulebook.

This is the gold-standard oracle: plain Python, mutable state, ordinary control
flow. Correctness and clarity over speed. Everything operates on one game.

Player convention: players are 0-indexed. ``buildings`` maps a vertex to
``(player, Building)``; ``roads`` maps an edge to a player.

Randomness is *injected* rather than sampled: the dice result, the card drawn
when buying a development card, and the card stolen by the robber are passed in
on the action. This lets a caller (the differential test) feed the engine's
realised outcome to the reference and compare deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from catan_reference import board
from catan_reference.board import Layout
from catan_reference.types import (
    BANK_INITIAL,
    CITY_COST,
    DEV_CARD_COST,
    DEV_CARD_COUNTS,
    LARGEST_ARMY_MIN,
    LONGEST_ROAD_MIN,
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    N_PLAYERS,
    RESOURCES,
    ROAD_BUILDING_FREE_ROADS,
    ROAD_COST,
    ROBBER_DISCARD_LIMIT,
    SETTLEMENT_COST,
    VICTORY_POINTS_TO_WIN,
    Building,
    DevCard,
    Phase,
    PortType,
    Resource,
)

# Snake / boustrophedon setup order over the 2 * N_PLAYERS starting settlements.
SETUP_ORDER: list[int] = list(range(N_PLAYERS)) + list(range(N_PLAYERS - 1, -1, -1))


def _zero_resources() -> dict[Resource, int]:
    return {r: 0 for r in RESOURCES}


def _zero_dev() -> dict[DevCard, int]:
    return {c: 0 for c in DevCard}


# ===========================================================================
# Actions
# ===========================================================================


@dataclass(frozen=True)
class SetupSettlement:
    vertex: int


@dataclass(frozen=True)
class SetupRoad:
    edge: int


@dataclass(frozen=True)
class Roll:
    value: int | None = None  # injected dice sum (2..12)


@dataclass(frozen=True)
class Discard:
    """Give up one card of ``resource`` toward ``player``'s owed discard.

    Discarding is one card per action: the action repeats until the player's
    owed count (set to half the hand when the 7 was rolled) reaches zero. Any
    owing player may act -- the rulebook treats discards as simultaneous, so no
    order is imposed.
    """

    player: int
    resource: Resource


@dataclass(frozen=True)
class MoveRobber:
    tile: int
    victim: int | None = None
    stolen: Resource | None = None  # injected card taken from the victim


@dataclass(frozen=True)
class BuildRoad:
    edge: int


@dataclass(frozen=True)
class BuildSettlement:
    vertex: int


@dataclass(frozen=True)
class BuildCity:
    vertex: int


@dataclass(frozen=True)
class BuyDevelopmentCard:
    card: DevCard | None = None  # injected drawn card


@dataclass(frozen=True)
class PlayKnight:
    tile: int
    victim: int | None = None
    stolen: Resource | None = None  # injected card taken from the victim


@dataclass(frozen=True)
class PlayRoadBuilding:
    pass


@dataclass(frozen=True)
class PlayYearOfPlenty:
    first: Resource
    second: Resource


@dataclass(frozen=True)
class PlayMonopoly:
    resource: Resource


@dataclass(frozen=True)
class MaritimeTrade:
    give: Resource
    receive: Resource


@dataclass(frozen=True)
class EndTurn:
    pass


Action = (
    SetupSettlement
    | SetupRoad
    | Roll
    | Discard
    | MoveRobber
    | BuildRoad
    | BuildSettlement
    | BuildCity
    | BuyDevelopmentCard
    | PlayKnight
    | PlayRoadBuilding
    | PlayYearOfPlenty
    | PlayMonopoly
    | MaritimeTrade
    | EndTurn
)


# ===========================================================================
# State
# ===========================================================================


@dataclass
class Player:
    resources: dict[Resource, int] = field(default_factory=_zero_resources)
    dev_cards: dict[DevCard, int] = field(default_factory=_zero_dev)  # held, unplayed
    dev_bought_this_turn: dict[DevCard, int] = field(default_factory=_zero_dev)
    knights_played: int = 0


@dataclass
class Game:
    layout: Layout
    robber: int  # tile index
    players: list[Player]
    buildings: dict[int, tuple[int, Building]] = field(default_factory=dict)
    roads: dict[int, int] = field(default_factory=dict)  # edge -> player
    dev_deck: dict[DevCard, int] = field(default_factory=lambda: dict(DEV_CARD_COUNTS))

    phase: Phase = Phase.SETUP_SETTLEMENT
    current_player: int = 0
    setup_index: int = 0  # 0..2*N_PLAYERS; settlements placed so far in setup
    dice_roll: int = 0  # last roll, 0 if not rolled this turn
    has_rolled: bool = False
    dev_played_this_turn: bool = False
    free_roads: int = 0  # owed by Road Building
    pending_discard: list[int] = field(default_factory=lambda: [0] * N_PLAYERS)

    longest_road_owner: int | None = None
    largest_army_owner: int | None = None
    longest_road_len: int = 0

    # -- construction ----------------------------------------------------

    @staticmethod
    def new(layout: Layout, robber: int) -> "Game":
        """A fresh game in the setup phase (robber starts on the desert tile)."""
        return Game(
            layout=layout,
            robber=robber,
            players=[Player() for _ in range(N_PLAYERS)],
        )

    # -- derived counts --------------------------------------------------

    def _count_buildings(self, player: int, kind: Building) -> int:
        return sum(1 for (p, b) in self.buildings.values() if p == player and b == kind)

    def settlements(self, player: int) -> int:
        return self._count_buildings(player, Building.SETTLEMENT)

    def cities(self, player: int) -> int:
        return self._count_buildings(player, Building.CITY)

    def roads_built(self, player: int) -> int:
        return sum(1 for p in self.roads.values() if p == player)

    def building_vp(self, player: int) -> int:
        """Victory points from buildings only (settlement 1, city 2)."""
        return self.settlements(player) + 2 * self.cities(player)

    def total_vp(self, player: int) -> int:
        """Buildings + Longest Road + Largest Army + hidden Victory Point cards."""
        vp = self.building_vp(player)
        if self.longest_road_owner == player:
            vp += 2
        if self.largest_army_owner == player:
            vp += 2
        vp += self.players[player].dev_cards[DevCard.VICTORY_POINT]
        return vp

    def bank(self, resource: Resource) -> int:
        held = sum(p.resources[resource] for p in self.players)
        return BANK_INITIAL - held

    # -- affordability ---------------------------------------------------

    def _can_afford(self, player: int, cost: dict[Resource, int]) -> bool:
        hand = self.players[player].resources
        return all(hand[r] >= n for r, n in cost.items())

    def _pay(self, player: int, cost: dict[Resource, int]) -> None:
        hand = self.players[player].resources
        for r, n in cost.items():
            hand[r] -= n

    def _playable(self, player: int, card: DevCard) -> bool:
        """Held and not bought this turn (you can't play a card the turn you buy it)."""
        pl = self.players[player]
        return pl.dev_cards[card] - pl.dev_bought_this_turn[card] > 0

    # -- placement rules -------------------------------------------------

    def distance_rule_ok(self, vertex: int) -> bool:
        """Vertex empty and all directly adjacent vertices empty (rulebook p.7)."""
        if vertex in self.buildings:
            return False
        return all(n not in self.buildings for n in board.VERTEX_NEIGHBORS[vertex])

    def settlement_connected(self, player: int, vertex: int) -> bool:
        """Player owns a road incident to ``vertex`` (required outside setup)."""
        return any(self.roads.get(e) == player for e in board.VERTEX_EDGES[vertex])

    def road_placeable(self, player: int, edge: int) -> bool:
        """Edge empty and reachable from the player's network at a non-blocked end.

        A road extends from your roads/settlements/cities; an opponent's building
        sitting on the connecting intersection blocks routing through it.
        """
        if edge in self.roads:
            return False
        for v in board.edge_vertices(edge):
            owner = self.buildings.get(v)
            if owner is not None and owner[0] == player:
                return True  # own building at this end
            if owner is not None:
                continue  # opponent building blocks routing through here
            if any(
                self.roads.get(e2) == player
                for e2 in board.VERTEX_EDGES[v]
                if e2 != edge
            ):
                return True
        return False

    def port_ratio(self, player: int, give: Resource) -> int:
        """Best maritime ratio for giving ``give``: 4, else 3 (generic), else 2."""
        ratio = 4
        for vertex, (owner, _building) in self.buildings.items():
            if owner != player:
                continue
            port = self.layout.port_at_vertex(vertex)
            if port is None:
                continue
            if port.type is PortType.GENERIC:
                ratio = min(ratio, 3)
            elif port.type.value == give:
                ratio = min(ratio, 2)
        return ratio

    # -- longest road ----------------------------------------------------

    def longest_road_length(self, player: int) -> int:
        """Longest continuous road for ``player`` (no edge reused, no passing
        *through* an opponent's building; the board is tiny so a DFS is fine)."""
        my_edges = [e for e, p in self.roads.items() if p == player]
        if not my_edges:
            return 0

        incident: dict[int, list[int]] = {}
        for e in my_edges:
            for v in board.edge_vertices(e):
                incident.setdefault(v, []).append(e)

        def blocked(vertex: int) -> bool:
            owner = self.buildings.get(vertex)
            return owner is not None and owner[0] != player

        best = 0

        def dfs(vertex: int, used: set[int], length: int) -> None:
            nonlocal best
            best = max(best, length)
            if blocked(vertex):  # may end here, but not continue through
                return
            for e in incident.get(vertex, []):
                if e in used:
                    continue
                a, b = board.edge_vertices(e)
                nxt = b if a == vertex else a
                used.add(e)
                dfs(nxt, used, length + 1)
                used.discard(e)

        for e in my_edges:
            a, b = board.edge_vertices(e)
            dfs(a, {e}, 1)
            dfs(b, {e}, 1)
        return best

    def recompute_longest_road(self) -> None:
        """Reassign the Longest Road card per the rulebook tie rule (p.9).

        The holder keeps the card while still *tied* for the longest road. If the
        holder is beaten and two or more players tie for the new longest, the card
        is set aside (no holder). It is held only when exactly one player has the
        unique longest road of >= 5 segments.
        """
        lengths = [self.longest_road_length(p) for p in range(N_PLAYERS)]
        qualifying = [p for p in range(N_PLAYERS) if lengths[p] >= LONGEST_ROAD_MIN]
        if not qualifying:
            self.longest_road_owner = None
            self.longest_road_len = 0
            return
        top = max(lengths[p] for p in qualifying)
        leaders = [p for p in qualifying if lengths[p] == top]
        holder = self.longest_road_owner
        if holder is not None and holder in leaders:
            new_owner = holder  # holder keeps it while tied for longest
        elif len(leaders) == 1:
            new_owner = leaders[0]  # a single new leader takes it
        else:
            new_owner = None  # holder beaten, but the new longest is tied -> set aside
        self.longest_road_owner = new_owner
        self.longest_road_len = top if new_owner is not None else 0

    def recompute_largest_army(self) -> None:
        """Reassign the Largest Army card: first to 3 knights, taken only by
        *strictly more* (the holder keeps it on a tie).

        Mirrors the Longest Road tie rule: the holder keeps the card while tied
        for the most knights; if beaten and the new lead is itself tied, the card
        is set aside. In real play knights never decrease, so a holder is always
        among the leaders and the set-aside branch is a defensive default for
        out-of-band states.
        """
        knights = [self.players[p].knights_played for p in range(N_PLAYERS)]
        qualifying = [p for p in range(N_PLAYERS) if knights[p] >= LARGEST_ARMY_MIN]
        if not qualifying:
            self.largest_army_owner = None
            return
        top = max(knights[p] for p in qualifying)
        leaders = [p for p in qualifying if knights[p] == top]
        holder = self.largest_army_owner
        if holder is not None and holder in leaders:
            self.largest_army_owner = holder  # keeps it while tied for the most
        elif len(leaders) == 1:
            self.largest_army_owner = leaders[0]  # a strictly larger army takes it
        else:
            self.largest_army_owner = None  # holder beaten, new lead tied -> aside

    # -- production ------------------------------------------------------

    def production(self, roll: int) -> dict[int, dict[Resource, int]]:
        """Resources each player earns from ``roll``, honouring the bank shortage
        rule (rulebook p.4 / Almanac). Does not mutate state."""
        gains: dict[int, dict[Resource, int]] = {
            p: _zero_resources() for p in range(N_PLAYERS)
        }
        for tile in range(board.N_TILES):
            if tile == self.robber:
                continue
            if self.layout.tile_number[tile] != roll:
                continue
            resource = self.layout.tile_resource[tile]
            if resource is None:  # desert
                continue
            for v in board.TILE_VERTICES[tile]:
                owner = self.buildings.get(v)
                if owner is None:
                    continue
                player, kind = owner
                gains[player][resource] += 1 if kind is Building.SETTLEMENT else 2

        # Apply the bank shortage rule per resource.
        granted: dict[int, dict[Resource, int]] = {
            p: _zero_resources() for p in range(N_PLAYERS)
        }
        for r in RESOURCES:
            demand = {p: gains[p][r] for p in range(N_PLAYERS) if gains[p][r] > 0}
            total = sum(demand.values())
            stock = self.bank(r)
            if total <= stock:
                for p, n in demand.items():
                    granted[p][r] = n
            elif len(demand) == 1:
                (p,) = demand
                granted[p][r] = min(demand[p], stock)
            # else: nobody receives this resource this turn
        return granted

    # ===================================================================
    # Legality
    # ===================================================================

    def is_legal(self, action: Action) -> bool:
        match action:
            case SetupSettlement():
                return self._legal_setup_settlement(action)
            case SetupRoad():
                return self._legal_setup_road(action)
            case Roll():
                return self._legal_roll(action)
            case Discard():
                return self._legal_discard(action)
            case MoveRobber():
                return self._legal_move_robber(action)
            case BuildRoad():
                return self._legal_build_road(action)
            case BuildSettlement():
                return self._legal_build_settlement(action)
            case BuildCity():
                return self._legal_build_city(action)
            case BuyDevelopmentCard():
                return self._legal_buy_dev(action)
            case PlayKnight():
                return self._legal_knight(action)
            case PlayRoadBuilding():
                return self._legal_road_building(action)
            case PlayYearOfPlenty():
                return self._legal_year_of_plenty(action)
            case PlayMonopoly():
                return self._legal_monopoly(action)
            case MaritimeTrade():
                return self._legal_maritime(action)
            case EndTurn():
                return self._legal_end_turn(action)

    def _legal_setup_settlement(self, a: SetupSettlement) -> bool:
        return (
            self.phase is Phase.SETUP_SETTLEMENT
            and 0 <= a.vertex < board.N_VERTICES
            and self.distance_rule_ok(a.vertex)
        )

    def _setup_settlement_vertex(self) -> int | None:
        """The current player's setup settlement still awaiting its road."""
        for v, (p, _b) in self.buildings.items():
            if p == self.current_player and not any(
                self.roads.get(e) == p for e in board.VERTEX_EDGES[v]
            ):
                return v
        return None

    def _legal_setup_road(self, a: SetupRoad) -> bool:
        if self.phase is not Phase.SETUP_ROAD or not 0 <= a.edge < board.N_EDGES:
            return False
        if a.edge in self.roads:
            return False
        v = self._setup_settlement_vertex()
        return v is not None and v in board.edge_vertices(a.edge)

    def _legal_roll(self, a: Roll) -> bool:
        return self.phase is Phase.ROLL and not self.has_rolled

    def _legal_discard(self, a: Discard) -> bool:
        if self.phase is not Phase.DISCARD or not 0 <= a.player < N_PLAYERS:
            return False
        if self.pending_discard[a.player] == 0:
            return False
        return self.players[a.player].resources[a.resource] > 0

    def robber_victims(self, tile: int, thief: int) -> list[int]:
        """Players (other than ``thief``) with a building on ``tile`` and cards."""
        present: set[int] = set()
        for v in board.TILE_VERTICES[tile]:
            owner = self.buildings.get(v)
            if owner is not None:
                present.add(owner[0])
        return sorted(
            p
            for p in present
            if p != thief and sum(self.players[p].resources.values()) > 0
        )

    def grant_setup_resources(self, vertex: int, player: int) -> None:
        """Grant one card per adjacent non-desert tile (the 2nd-settlement bonus)."""
        for t in board.VERTEX_TILES[vertex]:
            resource = self.layout.tile_resource[t]
            if resource is not None and self.bank(resource) > 0:
                self.players[player].resources[resource] += 1

    def _legal_robber_move(self, tile: int, victim: int | None, thief: int) -> bool:
        if not 0 <= tile < board.N_TILES or tile == self.robber:
            return False
        victims = self.robber_victims(tile, thief)
        if victims:
            return victim in victims
        return victim is None

    def _legal_move_robber(self, a: MoveRobber) -> bool:
        return self.phase is Phase.MOVE_ROBBER and self._legal_robber_move(
            a.tile, a.victim, self.current_player
        )

    def _legal_build_road(self, a: BuildRoad) -> bool:
        if self.phase is not Phase.MAIN or not self.has_rolled:
            return False
        if not 0 <= a.edge < board.N_EDGES:
            return False
        p = self.current_player
        if self.roads_built(p) >= MAX_ROADS:
            return False
        if not self.road_placeable(p, a.edge):
            return False
        return self.free_roads > 0 or self._can_afford(p, ROAD_COST)

    def _legal_build_settlement(self, a: BuildSettlement) -> bool:
        if self.phase is not Phase.MAIN or not self.has_rolled:
            return False
        if not 0 <= a.vertex < board.N_VERTICES:
            return False
        p = self.current_player
        return (
            self.settlements(p) < MAX_SETTLEMENTS
            and self._can_afford(p, SETTLEMENT_COST)
            and self.distance_rule_ok(a.vertex)
            and self.settlement_connected(p, a.vertex)
        )

    def _legal_build_city(self, a: BuildCity) -> bool:
        if self.phase is not Phase.MAIN or not self.has_rolled:
            return False
        if not 0 <= a.vertex < board.N_VERTICES:
            return False
        p = self.current_player
        owner = self.buildings.get(a.vertex)
        return (
            self.cities(p) < MAX_CITIES
            and owner == (p, Building.SETTLEMENT)
            and self._can_afford(p, CITY_COST)
        )

    def _legal_buy_dev(self, a: BuyDevelopmentCard) -> bool:
        return (
            self.phase is Phase.MAIN
            and self.has_rolled
            and sum(self.dev_deck.values()) > 0
            and self._can_afford(self.current_player, DEV_CARD_COST)
        )

    def _legal_knight(self, a: PlayKnight) -> bool:
        if self.phase not in (Phase.ROLL, Phase.MAIN) or self.dev_played_this_turn:
            return False
        if not self._playable(self.current_player, DevCard.KNIGHT):
            return False
        return self._legal_robber_move(a.tile, a.victim, self.current_player)

    def _legal_road_building(self, a: PlayRoadBuilding) -> bool:
        return (
            self.phase is Phase.MAIN
            and self.has_rolled
            and not self.dev_played_this_turn
            and self._playable(self.current_player, DevCard.ROAD_BUILDING)
        )

    def _legal_year_of_plenty(self, a: PlayYearOfPlenty) -> bool:
        if self.phase is not Phase.MAIN or not self.has_rolled:
            return False
        if self.dev_played_this_turn or not self._playable(
            self.current_player, DevCard.YEAR_OF_PLENTY
        ):
            return False
        # Both cards must be available in the bank.
        if a.first == a.second:
            return self.bank(a.first) >= 2
        return self.bank(a.first) >= 1 and self.bank(a.second) >= 1

    def _legal_monopoly(self, a: PlayMonopoly) -> bool:
        return (
            self.phase is Phase.MAIN
            and self.has_rolled
            and not self.dev_played_this_turn
            and self._playable(self.current_player, DevCard.MONOPOLY)
        )

    def _legal_maritime(self, a: MaritimeTrade) -> bool:
        if self.phase is not Phase.MAIN or not self.has_rolled:
            return False
        if a.give == a.receive:
            return False
        p = self.current_player
        ratio = self.port_ratio(p, a.give)
        return self.players[p].resources[a.give] >= ratio and self.bank(a.receive) >= 1

    def _legal_end_turn(self, a: EndTurn) -> bool:
        return self.phase is Phase.MAIN and self.has_rolled

    # ===================================================================
    # Legal-action enumeration
    # ===================================================================

    def legal_actions(self) -> list[Action]:
        """Every legal action in the current state.

        Stochastic outcome fields (``Roll.value``, the stolen card, the drawn
        development card) are left unset; ``apply`` requires the caller to fill
        them in with the realised outcome.
        """
        if self.phase is Phase.GAME_OVER:
            return []
        out: list[Action] = []
        if self.phase is Phase.SETUP_SETTLEMENT:
            out += [
                SetupSettlement(v)
                for v in range(board.N_VERTICES)
                if self._legal_setup_settlement(SetupSettlement(v))
            ]
            return out
        if self.phase is Phase.SETUP_ROAD:
            out += [
                SetupRoad(e)
                for e in range(board.N_EDGES)
                if self._legal_setup_road(SetupRoad(e))
            ]
            return out
        if self.phase is Phase.DISCARD:
            return self._legal_discards()
        if self.phase is Phase.MOVE_ROBBER:
            out += self._robber_actions(MoveRobber)
            return out
        if self.phase is Phase.ROLL:
            out.append(Roll())
            out += self._robber_actions(PlayKnight, knight=True)
            return out
        # MAIN
        out += [
            BuildRoad(e)
            for e in range(board.N_EDGES)
            if self._legal_build_road(BuildRoad(e))
        ]
        out += [
            BuildSettlement(v)
            for v in range(board.N_VERTICES)
            if self._legal_build_settlement(BuildSettlement(v))
        ]
        out += [
            BuildCity(v)
            for v in range(board.N_VERTICES)
            if self._legal_build_city(BuildCity(v))
        ]
        if self._legal_buy_dev(BuyDevelopmentCard()):
            out.append(BuyDevelopmentCard())
        if self._legal_road_building(PlayRoadBuilding()):
            out.append(PlayRoadBuilding())
        out += [
            a
            for first in RESOURCES
            for second in RESOURCES
            if (a := PlayYearOfPlenty(first, second)) and self._legal_year_of_plenty(a)
        ]
        out += [
            PlayMonopoly(r) for r in RESOURCES if self._legal_monopoly(PlayMonopoly(r))
        ]
        out += [
            MaritimeTrade(g, r)
            for g in RESOURCES
            for r in RESOURCES
            if self._legal_maritime(MaritimeTrade(g, r))
        ]
        out += self._robber_actions(PlayKnight, knight=True)
        out.append(EndTurn())
        return out

    def _legal_discards(self) -> list[Action]:
        """Single-card discards for the first owing player (one per held resource).

        Discards are simultaneous in the rulebook, so ``is_legal`` accepts any
        owing player; for enumeration we serialize in player order (lowest
        index first). The order cannot change any outcome -- each player's
        discard choice is independent -- and it matches the engine's fixed
        order, so the differential driver exercises identical action streams.
        """
        for p in range(N_PLAYERS):
            if self.pending_discard[p] > 0:
                return [
                    Discard(p, r)
                    for r in RESOURCES
                    if self.players[p].resources[r] > 0
                ]
        return []

    def _robber_actions(
        self,
        ctor: type[MoveRobber] | type[PlayKnight],
        knight: bool = False,
    ) -> list[Action]:
        if knight and not (
            self.phase in (Phase.ROLL, Phase.MAIN)
            and not self.dev_played_this_turn
            and self._playable(self.current_player, DevCard.KNIGHT)
        ):
            return []
        out: list[Action] = []
        for tile in range(board.N_TILES):
            if tile == self.robber:
                continue
            victims = self.robber_victims(tile, self.current_player)
            if victims:
                out += [ctor(tile, v) for v in victims]
            else:
                out.append(ctor(tile, None))
        return out

    # ===================================================================
    # Application
    # ===================================================================

    def apply(self, action: Action) -> None:
        if not self.is_legal(action):
            raise ValueError(f"illegal action in phase {self.phase}: {action}")
        match action:
            case SetupSettlement():
                self._apply_setup_settlement(action)
            case SetupRoad():
                self._apply_setup_road(action)
            case Roll():
                self._apply_roll(action)
            case Discard():
                self._apply_discard(action)
            case MoveRobber():
                self._apply_move_robber(action)
            case BuildRoad():
                self._apply_build_road(action)
            case BuildSettlement():
                self._apply_build_settlement(action)
            case BuildCity():
                self._apply_build_city(action)
            case BuyDevelopmentCard():
                self._apply_buy_dev(action)
            case PlayKnight():
                self._apply_knight(action)
            case PlayRoadBuilding():
                self._apply_road_building(action)
            case PlayYearOfPlenty():
                self._apply_year_of_plenty(action)
            case PlayMonopoly():
                self._apply_monopoly(action)
            case MaritimeTrade():
                self._apply_maritime(action)
            case EndTurn():
                self._apply_end_turn(action)

    # -- setup -----------------------------------------------------------

    def _apply_setup_settlement(self, a: SetupSettlement) -> None:
        self.buildings[a.vertex] = (self.current_player, Building.SETTLEMENT)
        # The second settlement (reverse pass) grants its adjacent resources.
        if self.setup_index >= N_PLAYERS:
            self.grant_setup_resources(a.vertex, self.current_player)
        self.phase = Phase.SETUP_ROAD

    def _apply_setup_road(self, a: SetupRoad) -> None:
        self.roads[a.edge] = self.current_player
        self.setup_index += 1
        if self.setup_index < len(SETUP_ORDER):
            self.current_player = SETUP_ORDER[self.setup_index]
            self.phase = Phase.SETUP_SETTLEMENT
        else:
            self.current_player = 0
            self.phase = Phase.ROLL

    # -- roll / production ----------------------------------------------

    def _apply_roll(self, a: Roll) -> None:
        assert a.value is not None, "Roll requires an injected dice value"
        self.dice_roll = a.value
        self.has_rolled = True
        if a.value == 7:
            for p in range(N_PLAYERS):
                hand = sum(self.players[p].resources.values())
                self.pending_discard[p] = (
                    hand // 2 if hand > ROBBER_DISCARD_LIMIT else 0
                )
            if any(self.pending_discard):
                self.phase = Phase.DISCARD
            else:
                self.phase = Phase.MOVE_ROBBER
            return
        for p, gains in self.production(a.value).items():
            for r, n in gains.items():
                self.players[p].resources[r] += n
        self.phase = Phase.MAIN

    def _apply_discard(self, a: Discard) -> None:
        self.players[a.player].resources[a.resource] -= 1
        self.pending_discard[a.player] -= 1
        if not any(self.pending_discard):
            self.phase = Phase.MOVE_ROBBER

    # -- robber ----------------------------------------------------------

    def _do_steal(
        self, thief: int, victim: int | None, stolen: Resource | None
    ) -> None:
        if victim is None:
            return
        assert stolen is not None, "a steal from a victim requires the injected card"
        self.players[victim].resources[stolen] -= 1
        self.players[thief].resources[stolen] += 1

    def _apply_move_robber(self, a: MoveRobber) -> None:
        self.robber = a.tile
        self._do_steal(self.current_player, a.victim, a.stolen)
        self.phase = Phase.MAIN  # always resolves a post-7 robber move

    # -- building --------------------------------------------------------

    def _apply_build_road(self, a: BuildRoad) -> None:
        if self.free_roads > 0:
            self.free_roads -= 1
        else:
            self._pay(self.current_player, ROAD_COST)
        self.roads[a.edge] = self.current_player
        self.recompute_longest_road()
        self._check_win()

    def _apply_build_settlement(self, a: BuildSettlement) -> None:
        self._pay(self.current_player, SETTLEMENT_COST)
        self.buildings[a.vertex] = (self.current_player, Building.SETTLEMENT)
        # A new settlement can break an opponent's road, so recompute.
        self.recompute_longest_road()
        self._check_win()

    def _apply_build_city(self, a: BuildCity) -> None:
        self._pay(self.current_player, CITY_COST)
        self.buildings[a.vertex] = (self.current_player, Building.CITY)
        self._check_win()

    def _apply_buy_dev(self, a: BuyDevelopmentCard) -> None:
        assert a.card is not None, "BuyDevelopmentCard requires the injected drawn card"
        self._pay(self.current_player, DEV_CARD_COST)
        self.dev_deck[a.card] -= 1
        self.players[self.current_player].dev_cards[a.card] += 1
        self.players[self.current_player].dev_bought_this_turn[a.card] += 1
        self._check_win()  # a drawn Victory Point card can win immediately

    # -- development cards ----------------------------------------------

    def _apply_knight(self, a: PlayKnight) -> None:
        p = self.current_player
        self.players[p].dev_cards[DevCard.KNIGHT] -= 1
        self.players[p].knights_played += 1
        self.dev_played_this_turn = True
        self.robber = a.tile
        self.recompute_largest_army()
        self._do_steal(p, a.victim, a.stolen)
        self._check_win()

    def _apply_road_building(self, a: PlayRoadBuilding) -> None:
        p = self.current_player
        self.players[p].dev_cards[DevCard.ROAD_BUILDING] -= 1
        self.dev_played_this_turn = True
        # Grant up to 2 free roads, capped by the remaining road supply.
        self.free_roads += min(
            ROAD_BUILDING_FREE_ROADS, MAX_ROADS - self.roads_built(p)
        )

    def _apply_year_of_plenty(self, a: PlayYearOfPlenty) -> None:
        p = self.current_player
        self.players[p].dev_cards[DevCard.YEAR_OF_PLENTY] -= 1
        self.dev_played_this_turn = True
        self.players[p].resources[a.first] += 1
        self.players[p].resources[a.second] += 1

    def _apply_monopoly(self, a: PlayMonopoly) -> None:
        p = self.current_player
        self.players[p].dev_cards[DevCard.MONOPOLY] -= 1
        self.dev_played_this_turn = True
        for other in range(N_PLAYERS):
            if other == p:
                continue
            taken = self.players[other].resources[a.resource]
            self.players[other].resources[a.resource] = 0
            self.players[p].resources[a.resource] += taken

    def _apply_maritime(self, a: MaritimeTrade) -> None:
        p = self.current_player
        ratio = self.port_ratio(p, a.give)
        self.players[p].resources[a.give] -= ratio
        self.players[p].resources[a.receive] += 1

    def _apply_end_turn(self, a: EndTurn) -> None:
        self.dice_roll = 0
        self.has_rolled = False
        self.dev_played_this_turn = False
        self.free_roads = 0
        self.players[self.current_player].dev_bought_this_turn = _zero_dev()
        self.current_player = (self.current_player + 1) % N_PLAYERS
        self.phase = Phase.ROLL

    def _check_win(self) -> None:
        if self.total_vp(self.current_player) >= VICTORY_POINTS_TO_WIN:
            self.phase = Phase.GAME_OVER
