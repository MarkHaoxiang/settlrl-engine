"""Persistence so live games survive a restart, backed by SQLite.

A game is recorded as a header row -- the immutable ``(seed, n_players,
number_placement, seats)`` it is fully determined by, per
``catan_engine.record`` -- plus an ordered event log (a move, seat claim, or
chat line per row). SQLite gives the durability for free: each write is its own
committed transaction, so a crash can lose at most the last uncommitted event,
never corrupt the rest. One connection, serialised by a lock (writes are tiny
and infrequent), shared by every game's journal.

The store is pure I/O; reconstructing a live ``GameSession`` from a loaded game
(replaying its moves through the engine) lives in ``games`` and ``server``,
which already own the engine and the bot drivers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    from catan_engine.record import Move


class GameJournal:
    """The append point for one game's events.

    Mutators append under the game's lock; ``sync_moves`` writes only the moves
    not yet recorded, so a single hook on every state change captures human and
    bot moves alike. The store owns the connection, so ``close`` is a no-op.
    """

    def __init__(self, store: GameStore, game_id: str, moves_written: int) -> None:
        self._store = store
        self._game_id = game_id
        self._moves_written = moves_written

    def sync_moves(self, moves: Sequence["Move"]) -> None:
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

    def claim(self, seat: int, token: str) -> None:
        self._store._append(self._game_id, {"t": "claim", "seat": seat, "token": token})

    def chat(self, player: int | None, text: str) -> None:
        self._store._append(
            self._game_id, {"t": "chat", "player": player, "text": text}
        )

    def close(self) -> None:
        pass


class GameStore:
    """A SQLite database of games and their event logs. Thread-safe."""

    def __init__(self, root: "os.PathLike[str] | str") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.root / "games.db", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS games(id TEXT PRIMARY KEY, header TEXT)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events("
            "seq INTEGER PRIMARY KEY, game_id TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._db.commit()

    def create(self, game_id: str, setup: Mapping[str, object]) -> GameJournal:
        """Record a new game's header and return its journal."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO games(id, header) VALUES(?, ?)",
                (game_id, json.dumps({"id": game_id, **setup})),
            )
            self._db.commit()
        return GameJournal(self, game_id, moves_written=0)

    def reopen(self, game_id: str, moves_written: int) -> GameJournal:
        """A journal for a game already loaded from the store (after a restart)."""
        return GameJournal(self, game_id, moves_written)

    def remove(self, game_id: str) -> None:
        """Drop a game and its events (on eviction)."""
        with self._lock:
            self._db.execute("DELETE FROM games WHERE id = ?", (game_id,))
            self._db.execute("DELETE FROM events WHERE game_id = ?", (game_id,))
            self._db.commit()

    def _append(self, game_id: str, event: dict[str, object]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO events(game_id, payload) VALUES(?, ?)",
                (game_id, json.dumps(event)),
            )
            self._db.commit()

    def load(self) -> Iterator[tuple[dict[str, object], list[dict[str, object]]]]:
        """``(header, events)`` for every stored game, events in order."""
        with self._lock:
            games = self._db.execute("SELECT id, header FROM games").fetchall()
            loaded = [
                (
                    json.loads(header),
                    [
                        json.loads(payload)
                        for (payload,) in self._db.execute(
                            "SELECT payload FROM events WHERE game_id = ? ORDER BY seq",
                            (game_id,),
                        )
                    ],
                )
                for game_id, header in games
            ]
        yield from loaded
