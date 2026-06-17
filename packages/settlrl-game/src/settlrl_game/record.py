"""Replayable reference-game records.

A game is determined by its seed (the board, via ``random_layout``) plus the
flat move trace. Each move also stores its resolved stochastic outcome (the
dice total, the drawn development card, the stolen resource) so a record
replays without re-running any RNG and a separate process (the bot service) can
reconstruct the exact position. JSON round-trips for persistence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from random import Random
from typing import Literal

import settlrl_reference as ref

from settlrl_render.api.actions import to_action


class ReplayError(ValueError):
    """A record that does not replay cleanly (tampered, or drifted semantics)."""


@dataclass(frozen=True)
class Move:
    """One applied move: the seat, the flat action, and its resolved outcome.

    ``dice`` is set for a roll, ``drawn`` (a ``DevCard`` index) for a dev-card
    buy, ``stolen`` (a ``Resource`` index) for a robber steal; the rest are None.
    """

    player: int
    flat: int
    dice: int | None = None
    drawn: int | None = None
    stolen: int | None = None


@dataclass(frozen=True)
class GameRecord:
    seed: int
    n_players: int
    number_placement: str
    moves: tuple[Move, ...] = ()
    winner: int | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "n_players": self.n_players,
                "number_placement": self.number_placement,
                "moves": [asdict(m) for m in self.moves],
                "winner": self.winner,
                "meta": self.meta,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> GameRecord:
        d = json.loads(text)
        return cls(
            seed=d["seed"],
            n_players=d["n_players"],
            number_placement=d.get("number_placement", "random"),
            moves=tuple(Move(**m) for m in d.get("moves", [])),
            winner=d.get("winner"),
            meta=d.get("meta", {}),
        )


def initial_game(record: GameRecord) -> ref.Game:
    """The opening position: the seed's board, before any move."""
    placement: Literal["random", "spiral"] = (
        "spiral" if record.number_placement == "spiral" else "random"
    )
    layout = ref.random_layout(Random(record.seed), placement)
    return ref.Game.new(layout, ref.desert_tile(layout), n_players=record.n_players)


def _with_outcome(action: ref.Action, move: Move) -> ref.Action:
    """Refill an action's stochastic field from the move's stored outcome."""
    if isinstance(action, ref.Roll):
        return ref.Roll(move.dice)
    if isinstance(action, ref.BuyDevelopmentCard):
        assert move.drawn is not None
        return ref.BuyDevelopmentCard(ref.DevCard(move.drawn))
    if (
        isinstance(action, ref.MoveRobber | ref.PlayKnight)
        and action.victim is not None
    ):
        assert move.stolen is not None
        return type(action)(action.tile, action.victim, ref.Resource(move.stolen))
    return action


def replay(record: GameRecord) -> Iterator[ref.Game]:
    """Replay the record, yielding the game after each move.

    The same mutated ``Game`` is yielded each step, so snapshot it eagerly.
    Raises :class:`ReplayError` if a move is illegal in the reconstructed state.
    """
    game = initial_game(record)
    for move in record.moves:
        action = _with_outcome(to_action(move.flat, game), move)
        try:
            game.apply(action)
        except ValueError as exc:
            raise ReplayError(str(exc)) from exc
        yield game
