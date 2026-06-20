"""Elo Quick Match: pair waiting players of similar rating into one game.

A player polls :meth:`Matchmaker.matchmake` (re-presenting its ticket each time,
like the create-queue in :mod:`settlrl_app.game.games`) until it comes back with a
seat. Players are pooled per ``n_players``; a match forms when enough near-Elo
humans are waiting, or — so no one is stuck — once the oldest waiter has waited
long enough, with the remaining seats filled by the registered bots whose bucket
rating sits closest to the humans' mean. The Elo window widens with that wait, so
a lone or oddly-rated player eventually matches.

The whole call is serialised by one lock: forming a match awaits (rating lookups,
the engine reset offload, the registry create), and overlapping polls must not
both seat the same waiters. Notification is poll-based — no SSE surface.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

import anyio.to_thread
from settlrl_game.session import HUMAN, GameSession

from settlrl_app.api.deps import needs_driver
from settlrl_app.bots.providers import ProviderRegistry
from settlrl_app.game.games import (
    GameHandle,
    GameRegistry,
    RegistryFullError,
    win_threshold,
)
from settlrl_app.ratings import INITIAL_MU, INITIAL_SIGMA, display_rating
from settlrl_app.storage.store import GameStore

_TICKET_BYTES = 9

# Pairing knobs (tunable). The Elo window starts at ``_BASE_WINDOW`` and widens by
# ``_WIDEN_PER_10S`` for every 10s the oldest waiter has waited; once that wait
# reaches ``_NEVER_STUCK_S`` a match forms with bots filling the empty seats. A
# ticket idle past ``_TICKET_TTL_S`` (stopped polling) drops out of the pool.
_BASE_WINDOW = 150.0
_WIDEN_PER_10S = 75.0
_NEVER_STUCK_S = 20.0
_TICKET_TTL_S = 12.0


def elo_window(wait_secs: float, base: float, widen_per_10s: float) -> float:
    """The half-width of the rating window for a waiter that has waited
    ``wait_secs``: it widens linearly so an oddly-rated player still matches."""
    return base + widen_per_10s * (wait_secs / 10.0)


@dataclass
class _Entry:
    """One player waiting in a bucket. ``result`` is set once matched: the game
    and the seat (with its token) handed back on the ticket's next poll."""

    ticket: str
    user_id: str | None
    rating: float
    joined_at: float
    last_seen: float
    result: tuple[str, int, str] | None = None


@dataclass
class Match:
    """What a poll returns: a seat in a created game, or a place still in line."""

    ticket: str
    waiting: int
    result: tuple[str, int, str] | None = None


class Matchmaker:
    """Per-``n_players`` Elo matchmaking over the live registry."""

    def __init__(
        self,
        registry: GameRegistry,
        bots: ProviderRegistry,
        spawn_driver: Callable[[GameHandle], None],
        *,
        store: GameStore | None = None,
        turn_timeout: float = 0.0,
        base_window: float = _BASE_WINDOW,
        widen_per_10s: float = _WIDEN_PER_10S,
        never_stuck_s: float = _NEVER_STUCK_S,
        ticket_ttl_s: float = _TICKET_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._bots = bots
        self._spawn_driver = spawn_driver
        self._store = store
        self._turn_timeout = turn_timeout
        self._base_window = base_window
        self._widen_per_10s = widen_per_10s
        self._never_stuck_s = never_stuck_s
        self._ticket_ttl_s = ticket_ttl_s
        self._clock = clock
        self._pools: dict[int, list[_Entry]] = {}
        self._lock = asyncio.Lock()

    def waiting(self, n_players: int) -> int:
        """Unmatched waiters in a bucket."""
        return sum(1 for e in self._pools.get(n_players, []) if e.result is None)

    async def matchmake(
        self, n_players: int, ticket: str | None, user_id: str | None
    ) -> Match:
        """Enqueue or refresh the caller's ticket and try to form a match. Returns
        the caller's seat once matched, else its still-waiting place in line."""
        async with self._lock:
            now = self._clock()
            pool = self._pools.setdefault(n_players, [])
            self._prune(pool, now)
            entry = next((e for e in pool if e.ticket == ticket), None)
            if entry is None:
                entry = _Entry(
                    ticket=secrets.token_urlsafe(_TICKET_BYTES),
                    user_id=user_id,
                    rating=await self._rating("account", user_id, n_players),
                    joined_at=now,
                    last_seen=now,
                )
                pool.append(entry)
            else:
                entry.last_seen = now
            if entry.result is None:
                await self._try_form(pool, n_players, now)
            return Match(entry.ticket, self.waiting(n_players), entry.result)

    def _prune(self, pool: list[_Entry], now: float) -> None:
        pool[:] = [e for e in pool if e.last_seen >= now - self._ticket_ttl_s]

    async def _rating(self, kind: str, subject_id: str | None, n_players: int) -> float:
        if subject_id is None or self._store is None:
            return display_rating(INITIAL_MU, INITIAL_SIGMA)
        return await self._store.rating_for(kind, subject_id, n_players)

    async def _try_form(self, pool: list[_Entry], n_players: int, now: float) -> None:
        waiting = sorted(
            (e for e in pool if e.result is None), key=lambda e: e.joined_at
        )
        if not waiting:
            return
        oldest = waiting[0]
        window = elo_window(
            now - oldest.joined_at, self._base_window, self._widen_per_10s
        )
        # Gather near-Elo waiters, but never the same account twice (a signed-in
        # player queued from two tabs): one human must not fill two seats.
        group: list[_Entry] = []
        seen_users: set[str] = set()
        for e in waiting:
            if abs(e.rating - oldest.rating) > window:
                continue
            if e.user_id is not None and e.user_id in seen_users:
                continue
            group.append(e)
            if e.user_id is not None:
                seen_users.add(e.user_id)
            if len(group) >= n_players:
                break
        full = len(group) >= n_players
        timed_out = now - oldest.joined_at >= self._never_stuck_s
        if not full and not timed_out:
            return
        humans = group
        mean = sum(e.rating for e in humans) / len(humans)
        fill = await self._bot_fill(n_players - len(humans), n_players, mean)
        if fill is None:  # not enough humans and no bot can fill the table
            return
        await self._create_match(n_players, humans, fill)

    async def _bot_fill(
        self, count: int, n_players: int, mean_rating: float
    ) -> list[str] | None:
        """``count`` bot kinds for the empty seats, each the registered kind whose
        bucket rating is closest to ``mean_rating`` (cycling if more seats than
        kinds). ``[]`` when none are needed; ``None`` when seats need filling but
        no registered bot supports this player count."""
        if count <= 0:
            return []
        kinds = []
        for k, spec in self._bots.catalog().items():
            counts = spec.get("counts")
            if isinstance(counts, list) and n_players in counts:
                kinds.append(k)
        if not kinds:
            return None
        scored = [
            (abs(await self._rating("bot", k, n_players) - mean_rating), k)
            for k in kinds
        ]
        scored.sort()
        order = [k for _, k in scored]
        return [order[i % len(order)] for i in range(count)]

    async def _create_match(
        self, n_players: int, humans: list[_Entry], bot_kinds: list[str]
    ) -> None:
        seats = [HUMAN] * len(humans) + bot_kinds
        seed = secrets.randbelow(2**31)
        external = self._bots.remote_kinds() | set(bot_kinds)
        session = GameSession(seed=seed, n_players=n_players)
        await anyio.to_thread.run_sync(
            lambda: session.reset(
                seed,
                seats=seats,
                external_kinds=external,
                victory_points_to_win=win_threshold(n_players),
            )
        )
        try:
            handle = self._registry.create(session)
        except RegistryFullError:
            return  # no slot right now; the waiters poll again
        for seat, entry in enumerate(humans):
            _, token = handle.claim(seat, entry.user_id)
            entry.result = (handle.id, seat, token)
        if needs_driver(handle, self._turn_timeout):
            self._spawn_driver(handle)
        handle.bump()
