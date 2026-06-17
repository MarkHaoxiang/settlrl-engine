"""A loaded game record, replayed into per-move snapshots for the Replay view.

``ReplaySession`` takes a :class:`GameRecord` (e.g. the JSON from
``GET /api/game/record``), replays it on a reference game once, and keeps the
wire-format board after every move plus a move log -- so scrubbing to any point
of the game is a lookup, not a re-simulation. Replaying validates the record
(:class:`ReplayError` on tampering / drift).
"""

from __future__ import annotations

from settlrl_render.api.actions import decode_actions
from settlrl_render.api.convert import board_to_model
from settlrl_render.api.models import BoardModel, LogEntryModel, ReplayStateModel
from settlrl_render.game.record import GameRecord, initial_game, replay

# Refuse pathologically long records (snapshots are kept in memory).
_MAX_MOVES = 20_000

# The state's log shows at most this many lines (matches the live game's cap).
_LOG_TAIL = 500


class ReplaySession:
    """One replayed game: a board snapshot per move and the move log."""

    def __init__(self, record: GameRecord) -> None:
        if len(record.moves) > _MAX_MOVES:
            raise ValueError(f"record has more than {_MAX_MOVES} moves")
        self.record = record
        self._boards: list[BoardModel] = [board_to_model(initial_game(record))]
        self._log: list[LogEntryModel] = []
        for i, (move, game) in enumerate(
            zip(record.moves, replay(record), strict=True)
        ):
            self._boards.append(board_to_model(game))
            action = decode_actions([move.flat])[0]
            text = f"rolled {move.dice}" if move.dice is not None else action.label
            self._log.append(
                LogEntryModel(
                    id=i,
                    kind="move",
                    player=move.player,
                    action_type=action.type,
                    text=text,
                )
            )
        if record.winner is not None:
            self._log.append(
                LogEntryModel(
                    id=len(record.moves),
                    kind="win",
                    player=record.winner,
                    text="wins",
                )
            )

    @property
    def n_moves(self) -> int:
        return len(self.record.moves)

    def state(self, move: int) -> ReplayStateModel:
        """The snapshot after ``move`` moves (0 = the opening board).

        Raises ``IndexError`` when ``move`` is outside ``0..n_moves``.
        """
        if not 0 <= move <= self.n_moves:
            raise IndexError(f"move {move} outside 0..{self.n_moves}")
        # The win line only shows once the scrubber reaches the end.
        upto = move if move < self.n_moves else len(self._log)
        log = self._log[:upto]
        seats = self.record.meta.get("seats")
        return ReplayStateModel(
            move=move,
            n_moves=self.n_moves,
            board=self._boards[move],
            log=log[-_LOG_TAIL:],
            winner=self.record.winner,
            seats=seats if isinstance(seats, list) else None,
        )
