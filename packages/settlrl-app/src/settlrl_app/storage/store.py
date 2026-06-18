"""Persistence so live games survive a restart, on the shared async DB.

A game is recorded as a header row -- the immutable ``(seed, n_players,
number_placement, seats)`` it is fully determined by -- plus an ordered event
log (a move, seat claim, or chat line per row), the
:class:`~settlrl_app.storage.db.GameRow` / ``GameEvent`` tables of the one
:class:`~settlrl_app.storage.db.Database`.

Writes are **write-behind**: callers (the registry, a mutator's ``bump``) enqueue
synchronously and never await, and a single background task drains the queue
against the DB in order. So the game-driving code stays synchronous and the
event loop never blocks on a commit. :meth:`GameStore.start` launches the writer;
:meth:`GameStore.aclose` enqueues a sentinel and waits, draining everything
queued before it -- so a clean shutdown loses nothing (only a hard crash can
lose the last unwritten events).

Reconstructing a live ``GameSession`` from a loaded game (replaying its moves)
lives in ``games``, which owns the game and bot drivers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from settlrl_game.record import GameRecord, Move
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from settlrl_app.storage.db import Database, GameEvent, GameRow

# Finished games kept as replayable history; past this the oldest are pruned.
_HISTORY_CAP = 200


@dataclass(frozen=True)
class _WriteHeader:
    game_id: str
    header: dict[str, object]


@dataclass(frozen=True)
class _Append:
    game_id: str
    payload: dict[str, object]


@dataclass(frozen=True)
class _Finish:
    game_id: str
    finished_at: float
    winner: int | None
    owners: dict[str, list[int]]


@dataclass(frozen=True)
class _Remove:
    game_id: str


_Op = _WriteHeader | _Append | _Finish | _Remove


@dataclass(frozen=True)
class FinishedGame:
    """One past game in a user's history (its replayable record lives in the
    journal under ``id``)."""

    id: str
    finished_at: float
    winner: int | None
    header: dict[str, object]
    owners: dict[str, list[int]]


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
        # The full resolved outcome is stored (not just the flat), so a game
        # rebuilds faithfully from the journal without re-sampling the seed --
        # which would diverge for any game that used the random fallback.
        for move in moves[self._moves_written :]:
            self._store._append(
                self._game_id,
                {
                    "t": "move",
                    "player": move.player,
                    "flat": move.flat,
                    "dice": move.dice,
                    "drawn": move.drawn,
                    "stolen": move.stolen,
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

    def finish(
        self, finished_at: float, winner: int | None, owners: dict[str, list[int]]
    ) -> None:
        """Mark the game finished, so it is kept as history rather than removed."""
        self._store._queue.put_nowait(
            _Finish(self._game_id, finished_at, winner, owners)
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
                elif isinstance(op, _Finish):
                    await session.execute(
                        update(GameRow)
                        .where(GameRow.id == op.game_id)
                        .values(
                            finished_at=op.finished_at,
                            winner=op.winner,
                            owners=op.owners,
                        )
                    )
                    await self._prune_history(session)
                else:
                    await session.execute(
                        delete(GameEvent).where(GameEvent.game_id == op.game_id)
                    )
                    await session.execute(
                        delete(GameRow).where(GameRow.id == op.game_id)
                    )
                await session.commit()

    async def _prune_history(self, session: AsyncSession) -> None:
        """Drop finished games (and their events) beyond the newest cap."""
        stale = (
            (
                await session.execute(
                    select(GameRow.id)
                    .where(GameRow.finished_at.is_not(None))
                    .order_by(GameRow.finished_at.desc())
                    .offset(_HISTORY_CAP)
                )
            )
            .scalars()
            .all()
        )
        if stale:
            await session.execute(delete(GameEvent).where(GameEvent.game_id.in_(stale)))
            await session.execute(delete(GameRow).where(GameRow.id.in_(stale)))

    async def history(self) -> list[FinishedGame]:
        """Finished games kept as history, newest first."""
        async with self._db.sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        select(GameRow)
                        .where(GameRow.finished_at.is_not(None))
                        .order_by(GameRow.finished_at.desc())
                    )
                )
                .scalars()
                .all()
            )
            return [
                FinishedGame(
                    id=row.id,
                    finished_at=row.finished_at or 0.0,
                    winner=row.winner,
                    header=row.header,
                    owners=row.owners or {},
                )
                for row in rows
            ]

    async def load(
        self,
    ) -> list[tuple[dict[str, object], list[dict[str, object]]]]:
        """``(header, events)`` for every **live** stored game, events in order
        (finished games are kept as history, not restored). The header carries the
        game id back under ``"id"`` (it is the row's key)."""
        async with self._db.sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        select(GameRow).where(GameRow.finished_at.is_(None))
                    )
                )
                .scalars()
                .all()
            )
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

    async def finished_record(self, game_id: str) -> GameRecord | None:
        """The replayable :class:`GameRecord` for a **finished** stored game
        (rebuilt from its journalled moves and their stored outcomes — so an
        evicted past game still downloads / replays). None if it is not a stored,
        finished game."""
        async with self._db.sessionmaker() as session:
            row = await session.get(GameRow, game_id)
            if row is None or row.finished_at is None:
                return None
            events = (
                (
                    await session.execute(
                        select(GameEvent.payload)
                        .where(GameEvent.game_id == game_id)
                        .order_by(GameEvent.seq)
                    )
                )
                .scalars()
                .all()
            )
        placement: Literal["random", "spiral"] = (
            "spiral" if row.header.get("number_placement") == "spiral" else "random"
        )
        moves = tuple(
            Move(
                player=e["player"],
                flat=e["flat"],
                dice=e.get("dice"),
                drawn=e.get("drawn"),
                stolen=e.get("stolen"),
            )
            for e in events
            if e.get("t") == "move"
        )
        return GameRecord(
            seed=int(row.header["seed"]),
            n_players=int(row.header["n_players"]),
            number_placement=placement,
            moves=moves,
            winner=row.winner,
            meta={"seats": row.header.get("seats", [])},
        )
