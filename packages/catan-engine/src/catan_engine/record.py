"""Serialisable records of complete games: generate, save as JSON, and replay.

A game is fully determined by its configuration (``seed``, ``n_players``,
``number_placement`` -- all engine randomness derives from the seed) plus the
sequence of flat actions played, so a :class:`GameRecord` stores exactly that.
The JSON form additionally annotates every move with the action type and its
decoded, human-readable parameters; the annotations are derived from the flat
index on save and ignored on load (``flat`` is authoritative).

Schema (version 1)::

    {
      "version": 1,
      "seed": 7,
      "n_players": 4,
      "number_placement": "random",
      "winner": 2,                      // null while unfinished
      "meta": {...},                    // free-form caller metadata, if any
      "moves": [
        {"player": 0, "flat": 93, "type": "setup_settlement", "vertex": 21},
        {"player": 0, "flat": 0, "type": "roll_dice", "dice": 8},
        {"player": 0, "flat": 412, "type": "maritime_trade",
         "give": "wood", "receive": "ore"},
        ...
      ]
    }

Board positions are the engine's canonical vertex/edge/tile indices (map them
to coordinates with the host-side lookups in ``board.layout``); resources are
named. ``dice`` records the outcome of each ``roll_dice`` move so the file
reads as a full transcript and :func:`replay` can verify determinism.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.board.state import VICTORY_POINTS_TO_WIN
from catan_engine.board.tile import Tile
from catan_engine.env import ActionType, BatchedCatanEnv, N_FLAT, flat_to_action
from catan_engine.board import Board

__all__ = ["GameRecord", "Move", "ReplayError", "record_game", "replay"]

_VERSION = 1

# Host-side copy of the flat action table: row -> (type, idx, target).
_row_type, _row_params = flat_to_action(jnp.arange(N_FLAT))
_ATYPE = np.asarray(_row_type)
_IDX = np.asarray(_row_params.idx)
_TARGET = np.asarray(_row_params.target)

_RESOURCE_NAMES = tuple(t.name.lower() for t in Tile if t is not Tile.DESERT)

_VERTEX_TYPES = {
    ActionType.SETUP_SETTLEMENT,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
}
_EDGE_TYPES = {ActionType.SETUP_ROAD, ActionType.BUILD_ROAD}
_ROBBER_TYPES = {ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT}
_RESOURCE_TYPES = {ActionType.DISCARD, ActionType.PLAY_MONOPOLY}


class ReplayError(ValueError):
    """A record is inconsistent with the game its configuration produces."""


@dataclass(frozen=True)
class Move:
    """One played move: the acting player, the flat action, and -- for
    ``roll_dice`` -- the rolled total."""

    player: int
    flat: int
    dice: int | None = None


@dataclass(frozen=True)
class GameRecord:
    """A serialisable game: configuration, the moves played, and the winner.

    ``winner`` is None while the game is unfinished; ``meta`` is free-form
    caller metadata carried through serialisation untouched.
    """

    seed: int
    n_players: int = 4
    number_placement: Literal["random", "spiral"] = "random"
    moves: tuple[Move, ...] = ()
    winner: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self, *, indent: int | None = 2) -> str:
        """The record as JSON (see the module docstring for the schema)."""
        doc: dict[str, Any] = {
            "version": _VERSION,
            "seed": self.seed,
            "n_players": self.n_players,
            "number_placement": self.number_placement,
            "winner": self.winner,
        }
        if self.meta:
            doc["meta"] = self.meta
        doc["moves"] = [
            {"player": m.player, "flat": m.flat, **_describe(m)} for m in self.moves
        ]
        return json.dumps(doc, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> GameRecord:
        """Parse a record; per-move annotations are ignored (``flat`` rules)."""
        doc = json.loads(text)
        if doc.get("version") != _VERSION:
            raise ValueError(f"unsupported record version: {doc.get('version')!r}")
        winner = doc["winner"]
        return cls(
            seed=int(doc["seed"]),
            n_players=int(doc["n_players"]),
            number_placement=doc["number_placement"],
            moves=tuple(
                Move(
                    player=int(m["player"]),
                    flat=int(m["flat"]),
                    dice=None if m.get("dice") is None else int(m["dice"]),
                )
                for m in doc["moves"]
            ),
            winner=None if winner is None else int(winner),
            meta=doc.get("meta", {}),
        )


def _describe(move: Move) -> dict[str, Any]:
    """The readable annotation for a move (action type + decoded parameters)."""
    at = ActionType(int(_ATYPE[move.flat]))
    idx = int(_IDX[move.flat])
    target = int(_TARGET[move.flat])
    out: dict[str, Any] = {"type": at.name.lower()}
    if at in _VERTEX_TYPES:
        out["vertex"] = idx
    elif at in _EDGE_TYPES:
        out["edge"] = idx
    elif at in _ROBBER_TYPES:
        out["tile"] = idx
        if target >= 0:
            out["victim"] = target
    elif at in _RESOURCE_TYPES:
        out["resource"] = _RESOURCE_NAMES[idx]
    elif at is ActionType.PLAY_YEAR_OF_PLENTY:
        out["resources"] = [_RESOURCE_NAMES[idx], _RESOURCE_NAMES[target]]
    elif at is ActionType.MARITIME_TRADE:
        out["give"] = _RESOURCE_NAMES[idx]
        out["receive"] = _RESOURCE_NAMES[target]
    elif at is ActionType.ROLL_DICE and move.dice is not None:
        out["dice"] = move.dice
    return out


def _make_env(record: GameRecord) -> BatchedCatanEnv:
    return BatchedCatanEnv(
        batch_size=1,
        seed=record.seed,
        auto_reset=False,
        n_players=record.n_players,
        number_placement=record.number_placement,
    )


def _step_flat(env: BatchedCatanEnv, flat: int) -> None:
    at, params = flat_to_action(jnp.asarray([flat], dtype=jnp.int32))
    env.step(at, params)


def _terminal(env: BatchedCatanEnv) -> bool:
    return bool(np.asarray(env.terminations[0]).all())


def _winner(env: BatchedCatanEnv) -> int | None:
    vps = np.asarray(env._vps[0])
    if _terminal(env) and bool((vps >= VICTORY_POINTS_TO_WIN).any()):
        return int(np.argmax(vps))
    return None


# An action chooser for record_game: given a PRNG key and the live single-game
# env (read flat_mask / observe / board), the flat action to play.
Act = Callable[[jax.Array, BatchedCatanEnv], int]


def _uniform_random(key: jax.Array, env: BatchedCatanEnv) -> int:
    legal = np.flatnonzero(np.asarray(env.flat_mask()[0]))
    return int(legal[jax.random.randint(key, (), 0, legal.size)])


def record_game(
    seed: int = 0,
    *,
    n_players: int = 4,
    number_placement: Literal["random", "spiral"] = "random",
    act: Act | None = None,
    meta: dict[str, Any] | None = None,
    max_moves: int = 100_000,
) -> GameRecord:
    """Play one game to completion and return its record.

    ``act`` chooses each move (default: uniformly random over the legal
    actions); a returned illegal action raises ``ValueError``, and a game not
    finishing within ``max_moves`` raises ``RuntimeError``.
    """
    record = GameRecord(
        seed=seed, n_players=n_players, number_placement=number_placement
    )
    env = _make_env(record)
    choose: Act = act if act is not None else _uniform_random
    key = jax.random.key(seed)
    moves: list[Move] = []
    for _ in range(max_moves):
        if _terminal(env):
            break
        player = int(env.agent_selection[0])
        key, k = jax.random.split(key)
        flat = choose(k, env)
        if not bool(env.flat_mask()[0, flat]):
            raise ValueError(f"act chose illegal action {flat} for player {player}")
        _step_flat(env, flat)
        dice = (
            int(env.board[1].dice_roll[0])
            if ActionType(int(_ATYPE[flat])) is ActionType.ROLL_DICE
            else None
        )
        moves.append(Move(player=player, flat=flat, dice=dice))
    else:
        raise RuntimeError(f"game did not finish within {max_moves} moves")
    return GameRecord(
        seed=seed,
        n_players=n_players,
        number_placement=number_placement,
        moves=tuple(moves),
        winner=_winner(env),
        meta=meta or {},
    )


def replay(record: GameRecord) -> Iterator[Board]:
    """Re-play a record move by move, yielding the board after each move.

    Validates as it goes -- the acting player, each move's legality, recorded
    dice outcomes, and (on exhaustion) the recorded winner -- raising
    :class:`ReplayError` on any mismatch with the deterministic game the
    record's configuration produces.
    """
    env = _make_env(record)
    for n, move in enumerate(record.moves):
        acting = int(env.agent_selection[0])
        if acting != move.player:
            raise ReplayError(
                f"move {n}: recorded for player {move.player}, "
                f"but player {acting} is acting"
            )
        if not bool(env.flat_mask()[0, move.flat]):
            raise ReplayError(f"move {n}: action {move.flat} is not legal")
        _step_flat(env, move.flat)
        if move.dice is not None:
            rolled = int(env.board[1].dice_roll[0])
            if rolled != move.dice:
                raise ReplayError(
                    f"move {n}: recorded roll {move.dice}, engine rolled {rolled}"
                )
        yield env.board
    if record.winner is not None and _winner(env) != record.winner:
        raise ReplayError(
            f"recorded winner {record.winner}, engine produced {_winner(env)}"
        )
