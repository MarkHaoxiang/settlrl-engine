"""A single live Catan game, driven through the engine's AEC wrapper.

``GameSession`` wraps ``CatanAECEnv``: each seat is a human (hotseat) or a
``catan-agents`` bot; bots advance one move at a time (:meth:`bot_step`) so
the frontend can pace and animate them. The session exposes the board, the
acting seat's legal flat actions, a status snapshot, the chat / move log, and
a replayable ``GameRecord`` export.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax
import numpy as np
from catan_engine.board import Board
from catan_engine.board.state import VICTORY_POINTS_TO_WIN, GamePhase
from catan_engine.env.aec import CatanAECEnv
from catan_engine.record import GameRecord, Move

from .actions import decode_actions
from .bots import POLICIES, bot_act
from .models import GameStatusModel, LogEntryModel

# Seat kind for a human-controlled seat; every other kind is a POLICIES name.
HUMAN = "human"

# Guard against a pathological non-terminating bot loop (a full game is well
# under this many engine steps).
_MAX_BOT_STEPS = 50_000

# Oldest log entries are dropped past this many (long random games can take
# thousands of moves; the client only ever shows the tail).
_LOG_CAP = 500


class IllegalActionError(ValueError):
    """Raised when an action that is not currently legal is applied."""


class GameSession:
    """A live game behind the single-game AEC env.

    ``n_players`` (2..4) is how many seats the game has. ``seats`` assigns a
    controller to every seat: ``"human"`` or a ``catan-agents`` policy name
    (default: a human on seat 0 and ``"random"`` bots elsewhere). No seat has
    to be human -- an all-bot game is driven entirely by ``bot_step``.
    """

    def __init__(
        self,
        seed: int = 0,
        n_players: int = 4,
        seats: Sequence[str] | None = None,
    ) -> None:
        self.n_players = n_players
        self.reset(seed, seats=seats)

    def reset(
        self,
        seed: int = 0,
        n_players: int | None = None,
        number_placement: Literal["random", "spiral"] = "random",
        seats: Sequence[str] | None = None,
    ) -> None:
        """Start a fresh game.

        ``n_players`` changes the seat count (None keeps it); ``seats`` assigns
        every seat (None means a human on seat 0 and ``"random"`` bots
        elsewhere) and must have ``n_players`` entries, each ``"human"`` or a
        policy name.
        """
        if n_players is not None:
            self.n_players = n_players
        if seats is None:
            seats = [HUMAN] + ["random"] * (self.n_players - 1)
        if len(seats) != self.n_players:
            raise ValueError(f"expected {self.n_players} seats, got {len(seats)}")
        unknown = sorted(set(seats) - {HUMAN} - set(POLICIES))
        if unknown:
            raise ValueError(f"unknown seat kind(s): {', '.join(unknown)}")
        unsupported = sorted(
            kind
            for kind in set(seats) - {HUMAN}
            if self.n_players not in POLICIES[kind].n_players
        )
        if unsupported:
            raise ValueError(
                f"seat kind(s) not available in a {self.n_players}-player game: "
                f"{', '.join(unsupported)}"
            )
        self.seats: list[str] = list(seats)
        self.seed = seed
        self.number_placement: Literal["random", "spiral"] = number_placement
        self.env = CatanAECEnv(
            seed=seed,
            n_players=self.n_players,
            number_placement=number_placement,
            # Belief seats read the env's honest per-observer card counting.
            track_beliefs=any(
                kind != HUMAN and POLICIES[kind].observes == "belief"
                for kind in self.seats
            ),
        )
        # Dedicated key so bot choices are reproducible per seed and independent
        # of the engine's own randomness.
        self._key = jax.random.key(seed)
        self._log: list[LogEntryModel] = []
        self._log_id = 0
        self._win_logged = False
        # Full move trace for GameRecord export (unlike the capped chat log).
        self._moves: list[Move] = []

    # -- engine views -----------------------------------------------------

    @property
    def board(self) -> Board:
        """The underlying batched ``(BoardLayout, BoardState)`` (one game)."""
        return self.env._env.board

    def acting_seat(self) -> int:
        return int(self.env._env.agent_selection[0])

    def terminal(self) -> bool:
        return all(self.env.terminations.values())

    def legal_flat(self) -> np.ndarray:
        """Flat indices of the actions legal for the acting player right now."""
        mask = np.asarray(self.env.observe(self.env.agent_selection)["action_mask"])
        return np.flatnonzero(mask)

    # -- moves ------------------------------------------------------------

    def apply(self, flat: int) -> None:
        """Apply the acting human's chosen flat action."""
        if flat not in self.legal_flat():
            raise IllegalActionError(f"action {flat} is not legal right now")
        seat = self.acting_seat()
        self.env.step(int(flat))
        self._log_move(seat, int(flat))

    def bot_step(self) -> int | None:
        """Play one bot move if a bot seat is acting; return the flat played.

        Returns None when no bot move is due: a human seat is acting, the game
        is over, or the acting bot has no legal move.
        """
        if self.terminal():
            return None
        seat = self.acting_seat()
        if self.seats[seat] == HUMAN or self.legal_flat().size == 0:
            return None
        self._key, k = jax.random.split(self._key)
        flat = bot_act(self.seats[seat], k, self.env._env, seat)
        self.env.step(flat)
        self._log_move(seat, flat)
        return flat

    def _run_bots(self) -> None:
        """Play bot moves until a human seat is acting (or the game ends)."""
        for _ in range(_MAX_BOT_STEPS):
            if self.bot_step() is None:
                break

    # -- chat / log ---------------------------------------------------------

    def log(self) -> list[LogEntryModel]:
        """The game's chat / log (moves, chat messages, the win), oldest first."""
        return self._log

    def add_chat(self, player: int | None, text: str) -> None:
        """Append a chat line (``player`` is its seat; ``None``: a spectator)."""
        self._push_log("chat", player=player, text=text)

    def record(self) -> GameRecord:
        """The game so far as a replayable ``catan_engine.record.GameRecord``
        (seats noted in ``meta``; ``winner`` is None while still running)."""
        return GameRecord(
            seed=self.seed,
            n_players=self.n_players,
            number_placement=self.number_placement,
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

    def _log_move(self, seat: int, flat: int) -> None:
        """Log a just-played move (and the win, once the game ends)."""
        action = decode_actions([flat])[0]
        dice = (
            int(self.env._env._state.dice_roll[0])
            if action.type == "roll_dice"
            else None
        )
        self._moves.append(Move(player=seat, flat=flat, dice=dice))
        text = f"rolled {dice}" if dice is not None else action.label
        self._push_log("move", player=seat, action_type=action.type, text=text)
        winner = self.winner()
        if winner is not None and not self._win_logged:
            self._win_logged = True
            self._push_log("win", player=winner, text="wins")

    # -- status -----------------------------------------------------------

    def winner(self) -> int | None:
        """Winning seat once the game is over (None while it's still running)."""
        if not self.terminal():
            return None
        vps = np.asarray(self.env._env._vps[0])
        return (
            int(np.argmax(vps)) if bool((vps >= VICTORY_POINTS_TO_WIN).any()) else None
        )

    def status(self) -> GameStatusModel:
        """A snapshot of turn flow for the wire model."""
        state = self.env._env._state
        terminal = self.terminal()
        acting = self.acting_seat()
        return GameStatusModel(
            phase=GamePhase(int(state.phase[0])).name.lower(),
            current_player=int(state.current_player[0]),
            acting_player=acting,
            dice_roll=int(state.dice_roll[0]),
            has_rolled=bool(state.has_rolled[0]),
            your_turn=(not terminal) and self.seats[acting] == HUMAN,
            terminal=terminal,
            winner=self.winner(),
            seats=self.seats,
        )
