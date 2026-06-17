"""The live-game registry: many concurrent games, each with claimed seats.

A ``GameHandle`` owns one :class:`GameSession`, an :class:`asyncio.Lock` (the
engine is blocking and not thread-safe, so each game's session is mutated by
one task at a time — the lock is held across the ``to_thread`` offload), and the
seat claims: joining a human seat issues an opaque token, and every privileged
request proves seat ownership by presenting tokens. The registry caps the
number of live games by evicting finished or long-idle ones; a running game
that has been touched recently is never evicted, so a full registry of active
games refuses creation instead.

State changes are broadcast through a swap-on-bump :class:`asyncio.Event`
(:meth:`GameHandle.bump`): waiters (the SSE stream, the bot driver) capture the
current event and await it; a bump installs a fresh event and sets the old one,
so it wakes everyone parked on it without a lost wakeup. Decoupling the waker
from the mutation lock lets :meth:`bump` (and eviction) run from synchronous
code — the registry, mutated only from the event loop, never needs to await.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import cast

from .models import BotMoveModel
from .session import HUMAN, GameSession, GameSetup
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

# A queued creator who stops polling (closed tab) drops out of the line this
# long after their last poll, so a ghost can't hold up the queue. Keep it a few
# poll intervals so a live client never loses its place.
_TICKET_BYTES = 9
_TICKET_TTL_S = 12.0


class RegistryFullError(Exception):
    """Every slot holds a recently-active running game; creation must wait."""


@dataclass
class _Ticket:
    """One creator waiting for a free slot; ``last_seen`` is their last poll."""

    id: str
    last_seen: float


@dataclass(frozen=True)
class QueuePosition:
    """A creator's place in line when the server is at its concurrency cap."""

    ticket: str
    position: int  # 1-based
    total: int


class GameHandle:
    """One live game plus its concurrency and seat-ownership state."""

    def __init__(self, game_id: str, session: GameSession) -> None:
        self.id = game_id
        self.session = session
        # Held (async with) while mutating/reading the session, across the
        # engine to_thread offload, so one task touches a session at a time.
        self.lock = asyncio.Lock()
        # Swapped and set on every bump; waiters capture it before checking the
        # version, then await it (see the module docstring).
        self._changed = asyncio.Event()
        # seat -> token for claimed human seats.
        self.claims: dict[int, str] = {}
        # seat -> owning user id, for seats claimed by a signed-in account (so
        # the seat follows the user across devices, without the seat token).
        self.claim_users: dict[int, str] = {}
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

    def _wake(self) -> None:
        """Wake everyone parked on the current change event and arm the next."""
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def bump(self) -> None:
        """Mark a state change, journal new moves, and wake waiters.

        Called by mutators while holding :attr:`lock`; the waker itself takes no
        lock, so it is safe to call from synchronous code (e.g. eviction)."""
        self.version += 1
        if self.journal is not None:
            self.journal.sync_moves(self.session.moves)
        self._wake()

    def human_seats(self) -> list[int]:
        return [i for i, kind in enumerate(self.session.seats) if kind == HUMAN]

    def claim(
        self, seat: int | None = None, user_id: str | None = None
    ) -> tuple[int, str]:
        """Claim ``seat`` (or the first unclaimed human seat) and mint its token.
        When ``user_id`` is given (a signed-in claimer) the seat is also tied to
        that account, so they own it on any device without the token.

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
        if user_id is not None:
            self.claim_users[seat] = user_id
        if self.journal is not None:
            self.journal.claim(seat, token, user_id)
        return seat, token

    def owned_seats(self, tokens: list[str], user_id: str | None = None) -> set[int]:
        """The seats the requester owns: any proven by ``tokens`` plus any tied
        to their ``user_id`` (unknown tokens are ignored)."""
        presented = set(tokens)
        owned = {s for s, t in self.claims.items() if t in presented}
        if user_id is not None:
            owned |= {s for s, uid in self.claim_users.items() if uid == user_id}
        return owned

    def seats_for_user(self, user_id: str) -> list[int]:
        """The seats this account owns in this game (for the user's game list)."""
        return sorted(s for s, uid in self.claim_users.items() if uid == user_id)


class GameRegistry:
    """Id-addressed live games.

    Every method runs on the event loop and does no awaiting, so the ``_games``
    dict and the queue are mutated atomically without an explicit lock.

    ``max_games`` caps memory: past it, the least-recently-touched finished or
    abandoned game is evicted; ``create`` raises :class:`RegistryFullError`
    when nothing is evictable. ``max_active`` caps how many games run at once:
    past it, :meth:`admit` puts new creators in a FIFO queue instead (keep it
    below ``max_games`` so finished games can always be evicted to seat the next
    in line). A ``store`` persists games so they survive a restart (see
    :func:`restore_registry`).
    """

    def __init__(
        self,
        max_games: int = 32,
        max_active: int = 16,
        store: GameStore | None = None,
    ) -> None:
        self._games: dict[str, GameHandle] = {}
        self._max = max_games
        self._max_active = max_active
        self._store = store
        self._queue: list[_Ticket] = []

    def create(self, session: GameSession) -> GameHandle:
        return self._create_locked(session)

    def admit(
        self, session: GameSession, ticket_id: str | None
    ) -> GameHandle | QueuePosition:
        """Seat a new game, or return the caller's place in line when at the
        concurrency cap. ``ticket_id`` is the caller's prior place (None on the
        first try); callers re-present it each poll until they get a handle.
        FIFO: a freed slot seats the head of the queue, and a fresh request
        never jumps a non-empty line.
        """
        now = time.monotonic()
        self._prune_tickets(now)
        slot_free = self._active_count() < self._max_active
        ticket = next((t for t in self._queue if t.id == ticket_id), None)
        if ticket is not None:
            ticket.last_seen = now
            if slot_free and self._queue[0] is ticket:
                self._queue.remove(ticket)
                return self._create_locked(session)
            return self._position(ticket)
        if slot_free and not self._queue:
            return self._create_locked(session)
        ticket = _Ticket(secrets.token_urlsafe(_TICKET_BYTES), now)
        self._queue.append(ticket)
        return self._position(ticket)

    def _create_locked(self, session: GameSession) -> GameHandle:
        self._evict()
        game_id = secrets.token_urlsafe(_ID_BYTES)
        while game_id in self._games:
            game_id = secrets.token_urlsafe(_ID_BYTES)
        handle = GameHandle(game_id, session)
        if self._store is not None:
            handle.journal = self._store.create(game_id, session.setup.to_dict())
        self._games[game_id] = handle
        return handle

    def _active_count(self) -> int:
        return sum(1 for h in self._games.values() if not h.session.terminal())

    def _prune_tickets(self, now: float) -> None:
        self._queue = [t for t in self._queue if t.last_seen >= now - _TICKET_TTL_S]

    def _position(self, ticket: _Ticket) -> QueuePosition:
        return QueuePosition(ticket.id, self._queue.index(ticket) + 1, len(self._queue))

    def get(self, game_id: str) -> GameHandle | None:
        handle = self._games.get(game_id)
        if handle is not None:
            handle.touch()
        return handle

    def all_handles(self) -> list[GameHandle]:
        """A snapshot of the live handles (e.g. to start drivers after a
        restart)."""
        return list(self._games.values())

    def _insert(self, handle: GameHandle) -> None:
        """Place an already-built handle (used by restore; no eviction)."""
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
            victim.closed = True
            victim._wake()  # the game is gone; just wake its SSE/driver to exit
            if victim.journal is not None:
                victim.journal.close()
            if self._store is not None:
                self._store.remove(victim.id)


def restore_registry(
    store: GameStore, max_games: int = 32, max_active: int = 16
) -> GameRegistry:
    """Rebuild a registry from a store: replay each game's journal back into a
    live handle, so a restart resumes games in progress (callers restart bot
    drivers for the returned handles). Drivers are not started here."""
    registry = GameRegistry(max_games=max_games, max_active=max_active, store=store)
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
    claims: dict[int, str] = {}
    claim_users: dict[int, str] = {}
    try:
        session = GameSession.from_setup(GameSetup.from_dict(header))
        for event in events:
            _replay_event(session, claims, claim_users, event)
    except (KeyError, ValueError, TypeError):
        return None
    game_id = str(header["id"])
    handle = GameHandle(game_id, session)
    handle.claims = claims
    handle.claim_users = claim_users
    handle.version = len(events)
    moves = sum(1 for e in events if e.get("t") == "move")
    handle.journal = store.reopen(game_id, moves_written=moves)
    return handle


def _replay_event(
    session: GameSession,
    claims: dict[int, str],
    claim_users: dict[int, str],
    event: dict[str, object],
) -> None:
    """Apply one journalled event back onto the rebuilding game."""
    match event.get("t"):
        case "move":
            session.apply(cast(int, event["flat"]))
        case "chat":
            session.add_chat(
                cast("int | None", event.get("player")), str(event["text"])
            )
        case "claim":
            seat = cast(int, event["seat"])
            claims[seat] = str(event["token"])
            user_id = event.get("user_id")
            if user_id is not None:
                claim_users[seat] = cast(str, user_id)
