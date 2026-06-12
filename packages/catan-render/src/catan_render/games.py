"""The live-game registry: many concurrent games, each with claimed seats.

A ``GameHandle`` owns one :class:`GameSession`, its lock (FastAPI runs sync
endpoints in a threadpool, and a session is not thread-safe), and the seat
claims: joining a human seat issues an opaque token, and every privileged
request proves seat ownership by presenting tokens. The registry caps the
number of live games by evicting finished or long-idle ones; a running game
that has been touched recently is never evicted, so a full registry of active
games refuses creation instead.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Sequence
from typing import Literal, cast

from .models import BotMoveModel
from .session import HUMAN, GameSession, SeatLike
from .store import GameJournal, GameStore

# Games are addressed by unguessable ids; tokens prove seat ownership. The id
# is the only thing gating a game's public snapshot and its finished-game
# record export, so it needs real entropy (an attacker must not enumerate ids).
_ID_BYTES = 8
_TOKEN_BYTES = 16

# An unfinished game untouched for this long counts as abandoned and may be
# evicted to make room.
_IDLE_TTL_S = 3600.0

# A game that no one has played a move in is reclaimed far sooner: a real game
# gets moving within minutes, so a slot held by an unstarted game past this is
# almost always a create-flood leftover, not a game someone is about to join.
_UNSTARTED_TTL_S = 600.0


class RegistryFullError(Exception):
    """Every slot holds a recently-active running game; creation must wait."""


class GameHandle:
    """One live game plus its concurrency and seat-ownership state."""

    def __init__(self, game_id: str, session: GameSession) -> None:
        self.id = game_id
        self.session = session
        # A Condition doubles as the per-game lock: mutators hold it and call
        # bump(); push subscribers and the bot driver wait() on it.
        self.lock = threading.Condition()
        # seat -> token for claimed human seats.
        self.claims: dict[int, str] = {}
        self.touched = time.monotonic()
        # Monotonic change counter: every state change bumps it, and waiters
        # re-serialise their view when it moves.
        self.version = 0
        # The last bot move played (cleared when a human acts), so pushed
        # snapshots can animate it.
        self.bot_move: BotMoveModel | None = None
        # Set on eviction: waiters and the bot driver exit.
        self.closed = False
        # Crash-recovery journal (None when persistence is off).
        self.journal: GameJournal | None = None

    def touch(self) -> None:
        self.touched = time.monotonic()

    def bump(self) -> None:
        """Mark a state change and wake waiters (caller holds the lock)."""
        self.version += 1
        if self.journal is not None:
            self.journal.sync_moves(self.session.moves)
        self.lock.notify_all()

    def human_seats(self) -> list[int]:
        return [i for i, kind in enumerate(self.session.seats) if kind == HUMAN]

    def claim(self, seat: int | None = None) -> tuple[int, str]:
        """Claim ``seat`` (or the first unclaimed human seat) and mint its token.

        Raises ``LookupError`` when no seat is free and ``ValueError`` when the
        requested seat is not a human seat or is already claimed.
        """
        free = [s for s in self.human_seats() if s not in self.claims]
        if seat is None:
            if not free:
                raise LookupError("no unclaimed human seat")
            seat = free[0]
        elif seat not in self.human_seats():
            raise ValueError(f"seat {seat} is not a human seat")
        elif seat in self.claims:
            raise ValueError(f"seat {seat} is already claimed")
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        self.claims[seat] = token
        if self.journal is not None:
            self.journal.claim(seat, token)
        return seat, token

    def owned_seats(self, tokens: list[str]) -> set[int]:
        """The seats proven by ``tokens`` (unknown tokens are ignored)."""
        presented = set(tokens)
        return {s for s, t in self.claims.items() if t in presented}


class GameRegistry:
    """Id-addressed live games. All methods are thread-safe.

    ``max_games`` caps memory: past it, the least-recently-touched finished or
    abandoned game is evicted; ``create`` raises :class:`RegistryFullError`
    when nothing is evictable. A ``store`` persists games so they survive a
    restart (see :func:`restore_registry`).
    """

    def __init__(self, max_games: int = 32, store: GameStore | None = None) -> None:
        self._games: dict[str, GameHandle] = {}
        self._max = max_games
        self._store = store
        self._lock = threading.Lock()

    def create(self, session: GameSession) -> GameHandle:
        with self._lock:
            self._evict()
            game_id = secrets.token_urlsafe(_ID_BYTES)
            while game_id in self._games:
                game_id = secrets.token_urlsafe(_ID_BYTES)
            handle = GameHandle(game_id, session)
            if self._store is not None:
                handle.journal = self._store.create(_journal_header(game_id, session))
            self._games[game_id] = handle
            return handle

    def get(self, game_id: str) -> GameHandle | None:
        with self._lock:
            handle = self._games.get(game_id)
            if handle is not None:
                handle.touch()
            return handle

    def all_handles(self) -> list[GameHandle]:
        """A snapshot of the live handles (e.g. to start drivers after a
        restart)."""
        with self._lock:
            return list(self._games.values())

    def _insert(self, handle: GameHandle) -> None:
        """Place an already-built handle (used by restore; no eviction)."""
        with self._lock:
            self._games[handle.id] = handle

    def _evict(self) -> None:
        while len(self._games) >= self._max:
            # Finished games go first, then the least recently touched — but a
            # running game someone touched recently is never evicted from
            # under its players. Unstarted games (no move played) age out on a
            # much shorter clock so a burst of empty games can't pin every slot.
            now = time.monotonic()
            evictable = [
                h
                for h in self._games.values()
                if h.session.terminal()
                or (h.session.moves_played == 0 and h.touched < now - _UNSTARTED_TTL_S)
                or h.touched < now - _IDLE_TTL_S
            ]
            if not evictable:
                raise RegistryFullError("all games are active; try again later")
            victim = min(evictable, key=lambda h: (not h.session.terminal(), h.touched))
            del self._games[victim.id]
            with victim.lock:
                victim.closed = True
                victim.bump()
            if victim.journal is not None:
                victim.journal.close()
            if self._store is not None:
                self._store.remove(victim.id)


def _reconstructable_seats(session: GameSession) -> list[SeatLike]:
    """``session``'s seats as a ``reset``-compatible list (round-trips through
    the journal header)."""
    seats: list[SeatLike] = []
    for kind, params in zip(session.seats, session.seat_params, strict=True):
        if kind == HUMAN or not params:
            seats.append(kind)
        else:
            seats.append({"kind": kind, "params": dict(params)})
    return seats


def _journal_header(game_id: str, session: GameSession) -> dict[str, object]:
    return {
        "id": game_id,
        "seed": session.seed,
        "n_players": session.n_players,
        "number_placement": session.number_placement,
        "seats": _reconstructable_seats(session),
    }


def restore_registry(store: GameStore, max_games: int = 32) -> GameRegistry:
    """Rebuild a registry from a store: replay each game's journal back into a
    live handle, so a restart resumes games in progress (callers restart bot
    drivers for the returned handles). Drivers are not started here."""
    registry = GameRegistry(max_games=max_games, store=store)
    for header, events in store.load():
        handle = _rebuild_handle(store, header, events)
        if handle is not None:
            registry._insert(handle)
    return registry


def _rebuild_handle(
    store: GameStore, header: dict[str, object], events: list[dict[str, object]]
) -> GameHandle | None:
    """Reconstruct one game from its header and event log, or None if it can't
    be replayed (a corrupt file is dropped rather than failing the boot)."""
    try:
        seed = int(str(header["seed"]))
        n_players = int(str(header["n_players"]))
        placement = cast(Literal["random", "spiral"], header["number_placement"])
        seats = cast(Sequence[SeatLike], header["seats"])
        session = GameSession(seed=seed, n_players=n_players)
        session.reset(seed, number_placement=placement, seats=seats)
        claims: dict[int, str] = {}
        n_moves = 0
        for ev in events:
            kind = ev.get("t")
            if kind == "move":
                session.apply(int(str(ev["flat"])))
                n_moves += 1
            elif kind == "chat":
                session.add_chat(cast("int | None", ev.get("player")), str(ev["text"]))
            elif kind == "claim":
                claims[int(str(ev["seat"]))] = str(ev["token"])
    except (KeyError, ValueError, TypeError):
        return None
    handle = GameHandle(str(header["id"]), session)
    handle.claims = claims
    handle.version = len(events)
    handle.journal = store.reopen(str(header["id"]), moves_written=n_moves)
    return handle
