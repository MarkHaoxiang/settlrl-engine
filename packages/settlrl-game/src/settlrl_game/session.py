"""A single live Settlrl game, backed by the plain-Python ``settlrl_game.reference``.

``GameSession`` wraps a reference ``Game``: each seat is a human (hotseat) or a
bot played by a remote bot service (:mod:`settlrl_app.bots.providers`) — the
game server runs no bot policies in-process. It owns a seeded RNG that samples
the stochastic outcomes the game's actions take (dice, dev draws, steals), so a
game is reproducible from its seed and its flat move trace. It exposes the
board, the acting seat's legal flat actions, a status snapshot, the chat / move
log, card counting, a replayable ``GameRecord`` export, and :meth:`auto_step`
(a random legal move) for advancing a stalled turn or standing in for an
unreachable bot.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Literal, cast

import settlrl_game.reference as ref
from settlrl_game.actions import N_FLAT, decode_actions, legal_flats, to_action
from settlrl_game.convert import _RESOURCE_NAMES
from settlrl_game.models import (
    BeliefModel,
    GameStatusModel,
    LogEntryModel,
    PlayerBeliefModel,
    ResourceCounts,
    TradeOfferModel,
)
from settlrl_game.record import GameRecord, Move

# A seat assignment: "human" (hotseat) or a remote bot kind.
SeatLike = str

# Seat kind for a human-controlled seat; every other kind names a remote bot.
HUMAN = "human"

# Oldest log entries are dropped past this many (long random games can take
# thousands of moves; the client only ever shows the tail).
_LOG_CAP = 500


@dataclass(frozen=True)
class GameSetup:
    """What determines a fresh game; with the move trace it reconstructs one
    exactly. Round-trips through a plain dict for persistence."""

    seed: int
    n_players: int
    number_placement: Literal["random", "spiral"]
    seats: list[SeatLike]
    victory_points_to_win: int = 10

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_players": self.n_players,
            "number_placement": self.number_placement,
            "seats": self.seats,
            "victory_points_to_win": self.victory_points_to_win,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> GameSetup:
        """Read a setup back from its dict form (extra keys are ignored; a header
        without ``victory_points_to_win`` predates it and defaults to 10)."""
        return cls(
            seed=cast(int, d["seed"]),
            n_players=cast(int, d["n_players"]),
            number_placement=cast(Literal["random", "spiral"], d["number_placement"]),
            seats=list(cast("Sequence[SeatLike]", d["seats"])),
            victory_points_to_win=cast(int, d.get("victory_points_to_win", 10)),
        )


class IllegalActionError(ValueError):
    """Raised when an action that is not currently legal is applied."""


class GameSession:
    """A live game over a reference ``Game``.

    ``n_players`` (2..4) is how many seats the game has. ``seats`` assigns a
    controller to every seat: ``"human"`` or a remote bot kind (default: all
    human). No seat has to be human -- an all-bot game is driven by the remote
    providers, with :meth:`auto_step` as the liveness fallback.
    """

    def __init__(
        self,
        seed: int = 0,
        n_players: int = 4,
        seats: Sequence[SeatLike] | None = None,
        external_kinds: frozenset[str] = frozenset(),
        victory_points_to_win: int = 10,
    ) -> None:
        self.n_players = n_players
        self.reset(
            seed,
            seats=seats,
            external_kinds=external_kinds,
            victory_points_to_win=victory_points_to_win,
        )

    @classmethod
    def from_setup(
        cls, setup: GameSetup, external_kinds: frozenset[str] = frozenset()
    ) -> GameSession:
        """A fresh game at its opening position (replay its moves to advance)."""
        session = cls(
            seed=setup.seed,
            n_players=setup.n_players,
            seats=setup.seats,
            external_kinds=external_kinds,
            victory_points_to_win=setup.victory_points_to_win,
        )
        if setup.number_placement != "random":  # the ctor defaulted to "random"
            session.reset(
                setup.seed,
                number_placement=setup.number_placement,
                seats=setup.seats,
                external_kinds=external_kinds,
                victory_points_to_win=setup.victory_points_to_win,
            )
        return session

    def reset(
        self,
        seed: int = 0,
        n_players: int | None = None,
        number_placement: Literal["random", "spiral"] = "random",
        seats: Sequence[SeatLike] | None = None,
        external_kinds: frozenset[str] | None = None,
        victory_points_to_win: int = 10,
    ) -> None:
        """Start a fresh game.

        ``n_players`` changes the seat count (None keeps it); ``seats`` assigns
        every seat (None means all human) and must have ``n_players`` entries,
        each ``"human"`` or a remote bot kind. Every non-human kind must be in
        ``external_kinds`` — the kinds registered providers serve; the server
        runs no bots itself. ``external_kinds`` None keeps the current set.
        ``number_placement`` lays the number tokens: ``"random"`` shuffles them,
        ``"spiral"`` follows the rulebook spiral (terrain / ports are unchanged).
        ``victory_points_to_win`` is the total VP that ends the game (default 10).
        """
        if n_players is not None:
            self.n_players = n_players
        if external_kinds is not None:
            self._external_kinds = external_kinds
        if seats is None:
            seats = [HUMAN] * self.n_players
        if len(seats) != self.n_players:
            raise ValueError(f"expected {self.n_players} seats, got {len(seats)}")
        for kind in seats:
            if kind != HUMAN and kind not in self._external_kinds:
                raise ValueError(f"unknown seat kind: {kind!r}")
        self.seats: list[str] = list(seats)
        self.seed = seed
        self.number_placement: Literal["random", "spiral"] = number_placement
        self.victory_points_to_win = victory_points_to_win
        # One seeded RNG drives the board and every later stochastic outcome, so
        # the game is reproducible from (seed, flat moves).
        self._rng = Random(seed)
        layout = ref.random_layout(self._rng, self.number_placement)
        self.game = ref.Game.new(
            layout,
            ref.desert_tile(layout),
            n_players=self.n_players,
            victory_points_to_win=victory_points_to_win,
        )
        self._belief = ref.Belief.new(self.n_players)
        self._log: list[LogEntryModel] = []
        self._log_id = 0
        self._win_logged = False
        self._moves: list[Move] = []

    def set_seat_kind(self, seat: int, kind: SeatLike) -> None:
        """Relabel one seat's controller (lobby seat control / matchmaking
        bot-fill). A seat's kind is metadata — the engine only knows
        ``n_players`` — so this just swaps the label; valid only before any move
        is played. A non-human kind is added to the accepted external kinds."""
        if self._moves:
            raise IllegalActionError("cannot change seats after the game has started")
        if not 0 <= seat < self.n_players:
            raise ValueError(f"no seat {seat}")
        if kind != HUMAN:
            self._external_kinds = self._external_kinds | {kind}
        self.seats[seat] = kind

    # -- views ------------------------------------------------------------

    def acting_seat(self) -> int:
        """Whose move it is: an owing player during discard, the partner during
        a trade response, otherwise the current player."""
        g = self.game
        if g.phase is ref.Phase.DISCARD:
            return next(
                (p for p in range(g.n_players) if g.pending_discard[p] > 0),
                g.current_player,
            )
        if g.phase is ref.Phase.TRADE_RESPONSE and g.trade_partner is not None:
            return g.trade_partner
        return g.current_player

    @property
    def moves_played(self) -> int:
        """How many moves have been applied (0 = a game no one has started)."""
        return len(self._moves)

    @property
    def moves(self) -> list[Move]:
        """The full applied-move trace, in order (uncapped, unlike the log)."""
        return self._moves

    def moves_flat(self) -> list[int]:
        """The applied moves as flat action indices (the bot-service wire form)."""
        return [m.flat for m in self._moves]

    @property
    def belief_state(self) -> ref.Belief:
        """The raw card-counting tracker (for the bot-service engine bridge)."""
        return self._belief

    @property
    def setup(self) -> GameSetup:
        """This game's reconstructable setup."""
        return GameSetup(
            self.seed,
            self.n_players,
            self.number_placement,
            list(self.seats),
            self.victory_points_to_win,
        )

    def terminal(self) -> bool:
        return self.game.phase is ref.Phase.GAME_OVER

    def legal_flat(self) -> list[int]:
        """Flat indices of the actions legal for the acting player right now."""
        return legal_flats(self.game)

    def belief(self, observer: int | None = None) -> BeliefModel | None:
        """Card counting from ``observer``'s perspective (default: the acting
        human, falling back to the first human; None with no human seats). The
        observer's own row is omitted; everything served is publicly
        derivable."""
        if observer is None:
            acting = self.acting_seat()
            observer = (
                acting
                if self.seats[acting] == HUMAN
                else next((i for i, s in enumerate(self.seats) if s == HUMAN), None)
            )
        if observer is None:
            return None
        lo = self._belief.res_lo[observer]
        hi = self._belief.res_hi[observer]
        return BeliefModel(
            observer=observer,
            players=[
                PlayerBeliefModel(
                    player=p,
                    res_lo=ResourceCounts(
                        **{n: lo[p][i] for i, n in enumerate(_RESOURCE_NAMES)}
                    ),
                    res_hi=ResourceCounts(
                        **{n: hi[p][i] for i, n in enumerate(_RESOURCE_NAMES)}
                    ),
                )
                for p in range(self.n_players)
                if p != observer
            ],
        )

    # -- moves ------------------------------------------------------------

    def _resolve(
        self, action: ref.Action
    ) -> tuple[ref.Action, int | None, int | None, int | None]:
        """Sample a stochastic action's outcome from the RNG; returns the filled
        action and the (dice, drawn, stolen) values to record."""
        if isinstance(action, ref.Roll):
            dice = ref.roll_dice(self._rng)
            return ref.Roll(dice), dice, None, None
        if isinstance(action, ref.BuyDevelopmentCard):
            card = ref.draw_dev_card(self.game, self._rng)
            return ref.BuyDevelopmentCard(card), None, int(card), None
        if (
            isinstance(action, ref.MoveRobber | ref.PlayKnight)
            and action.victim is not None
        ):
            stolen = ref.steal(self.game, action.victim, self._rng)
            return (
                type(action)(action.tile, action.victim, stolen),
                None,
                None,
                int(stolen),
            )
        return action, None, None, None

    def _apply_resolved(
        self,
        seat: int,
        flat: int,
        resolved: ref.Action,
        dice: int | None,
        drawn: int | None,
        stolen: int | None,
    ) -> None:
        """Apply an already-resolved action, advance the belief, and log it."""
        before = copy.deepcopy(self.game)
        self.game.apply(resolved)
        self._belief.update(before, self.game, resolved)
        self._moves.append(Move(seat, flat, dice, drawn, stolen))
        self._log_move(seat, flat, dice)

    def _play(self, seat: int, flat: int, action: ref.Action) -> None:
        """Resolve outcomes (sampling the RNG), apply, advance belief, log."""
        resolved, dice, drawn, stolen = self._resolve(action)
        self._apply_resolved(seat, flat, resolved, dice, drawn, stolen)

    def _recorded_outcome(self, action: ref.Action, move: Move) -> ref.Action | None:
        """``action`` with ``move``'s stored stochastic outcome injected, or None
        when a needed outcome is missing (a legacy journal: re-sample instead)."""
        if isinstance(action, ref.Roll):
            return None if move.dice is None else ref.Roll(move.dice)
        if isinstance(action, ref.BuyDevelopmentCard):
            if move.drawn is None:
                return None
            return ref.BuyDevelopmentCard(ref.DevCard(move.drawn))
        if (
            isinstance(action, ref.MoveRobber | ref.PlayKnight)
            and action.victim is not None
        ):
            if move.stolen is None:
                return None
            return type(action)(action.tile, action.victim, ref.Resource(move.stolen))
        return action  # non-stochastic: no outcome needed

    def apply(self, flat: int) -> None:
        """Apply the acting seat's chosen flat action."""
        if not 0 <= flat < N_FLAT:
            raise IllegalActionError(f"action {flat} is out of range")
        action = to_action(flat, self.game)
        if not self.game.is_legal(action):
            raise IllegalActionError(f"action {flat} is not legal right now")
        self._play(self.acting_seat(), flat, action)

    def apply_recorded(self, move: Move) -> None:
        """Re-apply a journalled move, reusing its stored stochastic outcome so a
        rebuilt game does not re-sample the seed — which would diverge for any
        game that used a random fallback (:meth:`auto_step`). Falls back to
        sampling for a legacy move that carries no outcome."""
        if not 0 <= move.flat < N_FLAT:
            raise IllegalActionError(f"action {move.flat} is out of range")
        action = to_action(move.flat, self.game)
        if not self.game.is_legal(action):
            raise IllegalActionError(f"action {move.flat} is not legal right now")
        seat = self.acting_seat()
        resolved = self._recorded_outcome(action, move)
        if resolved is None:
            self._play(seat, move.flat, action)
        else:
            self._apply_resolved(
                seat, move.flat, resolved, move.dice, move.drawn, move.stolen
            )

    def auto_step(self) -> int | None:
        """Play a uniformly-random legal move for the acting seat, whatever its
        kind — used to advance a stalled or abandoned turn, or to stand in for a
        bot whose remote provider is unreachable. Returns the flat played, or
        None when nothing is playable."""
        if self.terminal():
            return None
        legal = self.legal_flat()
        if not legal:
            return None
        flat = self._rng.choice(legal)
        self._play(self.acting_seat(), flat, to_action(flat, self.game))
        return flat

    # -- chat / log ---------------------------------------------------------

    def log(self) -> list[LogEntryModel]:
        """The game's chat / log (moves, chat messages, the win), oldest first."""
        return self._log

    def add_chat(self, player: int | None, text: str) -> None:
        """Append a chat line (``player`` is its seat; ``None``: a spectator)."""
        self._push_log("chat", player=player, text=text)

    def record(self) -> GameRecord:
        """The game so far as a replayable :class:`GameRecord` (seats noted in
        ``meta``; ``winner`` is None while still running)."""
        return GameRecord(
            seed=self.seed,
            n_players=self.n_players,
            number_placement=self.number_placement,
            victory_points_to_win=self.victory_points_to_win,
            moves=tuple(self._moves),
            winner=self.winner(),
            meta={"seats": self.seats},
        )

    def _push_log(
        self,
        kind: Literal["move", "chat", "win"],
        *,
        player: int | None = None,
        action_type: str | None = None,
        text: str = "",
    ) -> None:
        self._log.append(
            LogEntryModel(
                id=self._log_id,
                kind=kind,
                player=player,
                action_type=action_type,
                text=text,
            )
        )
        self._log_id += 1
        if len(self._log) > _LOG_CAP:
            del self._log[: len(self._log) - _LOG_CAP]

    def _log_move(self, seat: int, flat: int, dice: int | None) -> None:
        """Log a just-played move (and the win, once the game ends)."""
        action = decode_actions([flat])[0]
        text = f"rolled {dice}" if dice is not None else action.label
        self._push_log("move", player=seat, action_type=action.type, text=text)
        winner = self.winner()
        if winner is not None and not self._win_logged:
            self._win_logged = True
            self._push_log("win", player=winner, text="wins")

    # -- status -----------------------------------------------------------

    def winner(self) -> int | None:
        """Winning seat once the game is over (None while it's still running).

        A win only happens on the winner's own turn, so the terminal state's
        current player is the winner.
        """
        return self.game.current_player if self.terminal() else None

    def status(self) -> GameStatusModel:
        """A snapshot of turn flow for the wire model."""
        g = self.game
        terminal = self.terminal()
        acting = self.acting_seat()
        trade = None
        if g.trade_partner is not None and g.trade_give and g.trade_receive:
            # Render games play only the 1:1 propose rows: one card each side.
            trade = TradeOfferModel(
                proposer=g.current_player,
                partner=g.trade_partner,
                give=_RESOURCE_NAMES[g.trade_give.index(1)],
                receive=_RESOURCE_NAMES[g.trade_receive.index(1)],
            )
        return GameStatusModel(
            phase=g.phase.value,
            current_player=g.current_player,
            acting_player=acting,
            dice_roll=g.dice_roll,
            has_rolled=g.has_rolled,
            your_turn=(not terminal) and self.seats[acting] == HUMAN,
            terminal=terminal,
            winner=self.winner(),
            seats=self.seats,
            victory_points_to_win=self.victory_points_to_win,
            trade=trade,
        )
