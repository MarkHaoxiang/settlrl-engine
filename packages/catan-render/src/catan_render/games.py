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

from .session import HUMAN, GameSession

# Games are addressed by short ids; tokens prove seat ownership.
_ID_BYTES = 4
_TOKEN_BYTES = 16

# An unfinished game untouched for this long counts as abandoned and may be
# evicted to make room.
_IDLE_TTL_S = 3600.0


class RegistryFullError(Exception):
    """Every slot holds a recently-active running game; creation must wait."""


class GameHandle:
    """One live game plus its concurrency and seat-ownership state."""

    def __init__(self, game_id: str, session: GameSession) -> None:
        self.id = game_id
        self.session = session
        self.lock = threading.Lock()
        # seat -> token for claimed human seats.
        self.claims: dict[int, str] = {}
        self.touched = time.monotonic()

    def touch(self) -> None:
        self.touched = time.monotonic()

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
        return seat, token

    def owned_seats(self, tokens: list[str]) -> set[int]:
        """The seats proven by ``tokens`` (unknown tokens are ignored)."""
        presented = set(tokens)
        return {s for s, t in self.claims.items() if t in presented}


class GameRegistry:
    """Id-addressed live games. All methods are thread-safe.

    ``max_games`` caps memory: past it, the least-recently-touched finished or
    abandoned game is evicted; ``create`` raises :class:`RegistryFullError`
    when nothing is evictable.
    """

    def __init__(self, max_games: int = 32) -> None:
        self._games: dict[str, GameHandle] = {}
        self._max = max_games
        self._lock = threading.Lock()

    def create(self, session: GameSession) -> GameHandle:
        with self._lock:
            self._evict()
            game_id = secrets.token_urlsafe(_ID_BYTES)
            while game_id in self._games:
                game_id = secrets.token_urlsafe(_ID_BYTES)
            handle = GameHandle(game_id, session)
            self._games[game_id] = handle
            return handle

    def get(self, game_id: str) -> GameHandle | None:
        with self._lock:
            handle = self._games.get(game_id)
            if handle is not None:
                handle.touch()
            return handle

    def _evict(self) -> None:
        while len(self._games) >= self._max:
            # Finished games go first, then the least recently touched — but a
            # running game someone touched recently is never evicted from
            # under its players.
            cutoff = time.monotonic() - _IDLE_TTL_S
            evictable = [
                h
                for h in self._games.values()
                if h.session.terminal() or h.touched < cutoff
            ]
            if not evictable:
                raise RegistryFullError("all games are active; try again later")
            victim = min(evictable, key=lambda h: (not h.session.terminal(), h.touched))
            del self._games[victim.id]
