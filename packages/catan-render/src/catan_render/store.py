"""On-disk persistence so live games survive a restart.

Each game is one append-only JSONL file under the store root: the first line
is a header (the immutable ``(seed, n_players, number_placement, seats)`` a
game is fully determined by, per ``catan_engine.record``), and every later
line is an event -- a move, a seat claim, or a chat line -- in the order it
happened. Appending is crash-safe: a torn final line is simply dropped on
load, and the moves before it still replay into the same position.

The store is pure I/O; reconstructing a live ``GameSession`` from a loaded
file (replaying its moves through the engine) lives in ``games`` and ``server``,
which already own the engine and the bot drivers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catan_engine.record import Move


class GameJournal:
    """An open append-only file for one game's events.

    Mutators append under the game's lock, so a single journal's writes are
    serialised; different games write different files. ``sync_moves`` appends
    only the moves not yet written, so a single hook on every state change
    captures human and bot moves alike.
    """

    def __init__(self, file: "os.PathLike[str] | str", moves_written: int) -> None:
        # Held open for the game's lifetime (append-only); closed on eviction.
        self._fh = open(file, "a", encoding="utf-8")  # noqa: SIM115
        self._moves_written = moves_written

    def _append(self, event: dict[str, object]) -> None:
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()

    def header(self, game_id: str, setup: Mapping[str, object]) -> None:
        self._append({"t": "header", "id": game_id, **setup})

    def sync_moves(self, moves: Sequence["Move"]) -> None:
        for move in moves[self._moves_written :]:
            self._append(
                {
                    "t": "move",
                    "player": move.player,
                    "flat": move.flat,
                    "dice": move.dice,
                }
            )
        self._moves_written = len(moves)

    def claim(self, seat: int, token: str) -> None:
        self._append({"t": "claim", "seat": seat, "token": token})

    def chat(self, player: int | None, text: str) -> None:
        self._append({"t": "chat", "player": player, "text": text})

    def close(self) -> None:
        self._fh.close()


class GameStore:
    """A directory of per-game journals. Game ids must be filesystem-safe
    (the registry mints url-safe base64 ids, which are)."""

    def __init__(self, root: "os.PathLike[str] | str") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, game_id: str) -> Path:
        return self.root / f"{game_id}.jsonl"

    def create(self, game_id: str, setup: Mapping[str, object]) -> GameJournal:
        """Start a journal for a new game, writing its header line."""
        journal = GameJournal(self._path(game_id), moves_written=0)
        journal.header(game_id, setup)
        return journal

    def reopen(self, game_id: str, moves_written: int) -> GameJournal:
        """Re-open a loaded game's journal to keep appending after a restart."""
        return GameJournal(self._path(game_id), moves_written=moves_written)

    def remove(self, game_id: str) -> None:
        """Delete a game's file (on eviction)."""
        self._path(game_id).unlink(missing_ok=True)

    def load(self) -> Iterator[tuple[dict[str, object], list[dict[str, object]]]]:
        """Yield ``(header, events)`` for every stored game.

        A file whose header is unreadable is skipped; within a file, parsing
        stops at the first malformed line (a crash mid-append), keeping the
        events before it.
        """
        for path in sorted(self.root.glob("*.jsonl")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            try:
                header = json.loads(lines[0])
            except json.JSONDecodeError:
                continue
            if header.get("t") != "header":
                continue
            events: list[dict[str, object]] = []
            for line in lines[1:]:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # torn final write; the rest of the game is intact
            yield header, events
