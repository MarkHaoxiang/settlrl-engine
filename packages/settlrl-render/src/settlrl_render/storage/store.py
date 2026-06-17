"""Persistence so live games survive a restart, on the shared async DB.

A game is recorded as a header row -- the immutable ``(seed, n_players,
number_placement, seats)`` it is fully determined by, per
``settlrl_engine.record`` -- plus an ordered event log (a move, seat claim, or
chat line per row), the :class:`~settlrl_render.storage.db.GameRow` / ``GameEvent``
tables of the one :class:`~settlrl_render.storage.db.Database`.

Writes are **write-behind**: callers (the registry, a mutator's ``bump``) enqueue
synchronously and never await, and a single background task drains the queue
against the DB in order. So the engine-driving code stays synchronous and the
event loop never blocks on a commit. :meth:`GameStore.start` launches the writer;
:meth:`GameStore.aclose` enqueues a sentinel and waits, draining everything
queued before it -- so a clean shutdown loses nothing (only a hard crash can
lose the last unwritten events).

Reconstructing a live ``GameSession`` from a loaded game (replaying its moves
through the engine) lives in ``games``, which owns the engine and bot drivers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from settlrl_render.storage.db import Database, GameEvent, GameRow

if TYPE_CHECKING:
    from settlrl_render.game.record import Move


@dataclass(frozen=True)
class _WriteHeader:
    game_id: str
    header: dict[str, object]


@dataclass(frozen=True)
class _Append:
    game_id: str
    payload: dict[str, object]


@dataclass(frozen=True)
class _Remove:
    game_id: str


_Op = _WriteHeader | _Append | _Remove


class GameJournal:
    """The append point for one game's events.

    Mutators append under the game's lock; ``sync_moves`` enqueues only the moves
    not yet recorded, so a single hook on every state change captures human and
    bot moves alike."""

    def __init__(self, store: GameStore, game_id: str, moves_written: int) -> None:
        self._store = store
        self._game_id = game_id
        self._moves_written = moves_written

    def sync_moves(self, moves: Sequence[Move]) -> None:
        for move in moves[self._moves_written :]:
            self._store._append(
                self._game_id,
                {
                    "t": "move",
                    "player": move.player,
                    "flat": move.flat,
                    "dice": move.dice,
                },
            )
        self._moves_written = len(moves)

    def claim(self, seat: int, token: str, user_id: str | None = None) -> None:
        self._store._append(
            self._game_id,
            {"t": "claim", "seat": seat, "token": token, "user_id": user_id},
        )

    def chat(self, player: int | None, text: str) -> None:
        self._store._append(
            self._game_id, {"t": "chat", "player": player, "text": text}
        )

    def close(self) -> None:
        pass


class GameStore:
    """Game journals on the shared :class:`Database`, written by one background
    task. Enqueue methods are synchronous and non-blocking."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._queue: asyncio.Queue[_Op | None] = asyncio.Queue()
        self._writer: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Launch the writer task (call once a loop is running)."""
        self._writer = asyncio.create_task(self._drain())

    async def aclose(self) -> None:
        """Drain everything queued so far, then stop the writer."""
        if self._writer is not None:
            await self._queue.put(None)
            await self._writer
            self._writer = None

    def create(self, game_id: str, setup: Mapping[str, object]) -> GameJournal:
        """Record a new game's header and return its journal."""
        self._queue.put_nowait(_WriteHeader(game_id, dict(setup)))
        return GameJournal(self, game_id, moves_written=0)

    def reopen(self, game_id: str, moves_written: int) -> GameJournal:
        """A journal for a game already loaded from the store (after a restart)."""
        return GameJournal(self, game_id, moves_written)

    def remove(self, game_id: str) -> None:
        """Drop a game and its events (on eviction)."""
        self._queue.put_nowait(_Remove(game_id))

    def _append(self, game_id: str, payload: dict[str, object]) -> None:
        self._queue.put_nowait(_Append(game_id, payload))

    async def _drain(self) -> None:
        while True:
            op = await self._queue.get()
            if op is None:
                return
            async with self._db.sessionmaker() as session:
                if isinstance(op, _WriteHeader):
                    await session.merge(GameRow(id=op.game_id, header=op.header))
                elif isinstance(op, _Append):
                    session.add(GameEvent(game_id=op.game_id, payload=op.payload))
                else:
                    await session.execute(
                        delete(GameEvent).where(GameEvent.game_id == op.game_id)
                    )
                    await session.execute(
                        delete(GameRow).where(GameRow.id == op.game_id)
                    )
                await session.commit()

    async def load(
        self,
    ) -> list[tuple[dict[str, object], list[dict[str, object]]]]:
        """``(header, events)`` for every stored game, events in order. The
        header carries the game id back under ``"id"`` (it is the row's key)."""
        async with self._db.sessionmaker() as session:
            rows = (await session.execute(select(GameRow))).scalars().all()
            loaded: list[tuple[dict[str, object], list[dict[str, object]]]] = []
            for row in rows:
                events = list(
                    (
                        await session.execute(
                            select(GameEvent.payload)
                            .where(GameEvent.game_id == row.id)
                            .order_by(GameEvent.seq)
                        )
                    )
                    .scalars()
                    .all()
                )
                loaded.append(({"id": row.id, **row.header}, events))
            return loaded
