"""Serialisable records of complete games: generate, save as JSON, and replay.

A game is fully determined by ``(seed, n_players, number_placement)`` plus
the action trace (all engine randomness derives from the seed). Each move is
stored as the action-type *name* plus its ``(idx, target)`` parameters --
stable identifiers that survive growth of the flat action table, unlike flat
indices or ``ActionType`` integer values, which renumber whenever actions are
added (the version-1 schema stored flat indices and every pre-trade record
went stale). The remaining per-move fields are readable derivations, ignored
on load. Schema (version 3; version 2 stored ProposeTrade's params in a
retired 1:1 encoding)::

    {
      "version": 3,
      "seed": 7,
      "n_players": 4,
      "number_placement": "random",
      "winner": 2,                      // null while unfinished
      "meta": {...},                    // free-form caller metadata, if any
      "moves": [
        {"player": 0, "type": "setup_settlement", "idx": 21, "target": 0,
         "vertex": 21},
        {"player": 0, "type": "roll_dice", "idx": 0, "target": 0, "dice": 8},
        {"player": 0, "type": "maritime_trade", "idx": 2, "target": 4,
         "give": "wood", "receive": "ore"},
        ...
      ]
    }

Version-1 files are migrated on load from their annotations (which carry the
same stable identifiers); their recorded flat indices are ignored. Positions
are the engine's vertex/edge/tile indices (cube lookups live in
``board.layout``); ``dice`` records each roll so :func:`replay` can verify
determinism.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from settlrl_engine.board import Board
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import KeyScalar
from settlrl_engine.board.tile import Tile
from settlrl_engine.env import (
    N_FLAT,
    ActionType,
    BatchedSettlrlEnv,
    flat_to_action,
    random_flat,
)
from settlrl_engine.mechanics.trade import (
    _COUNT_BITS,
    _COUNT_MASK,
    _PARTNER_BITS,
    pack_trade_single,
)

__all__ = [
    "GameRecord",
    "Move",
    "ReplayError",
    "initial_board",
    "record_game",
    "replay",
]

_VERSION = 3

# Host-side copy of the flat action table: row -> (type, idx, target), and the
# reverse map serialisation uses to recover a move's flat row from its stable
# (type, idx, target) identifier.
_row_type, _row_params = flat_to_action(jnp.arange(N_FLAT))
_ATYPE = np.asarray(_row_type)
_IDX = np.asarray(_row_params.idx)
_TARGET = np.asarray(_row_params.target)
_ROW: dict[tuple[int, int, int], int] = {
    (int(a), int(i), int(t)): row
    for row, (a, i, t) in enumerate(zip(_ATYPE, _IDX, _TARGET, strict=True))
}

_RESOURCE_NAMES = tuple(t.name.lower() for t in Tile if t is not Tile.DESERT)
_RESOURCE_INDEX = {name: i for i, name in enumerate(_RESOURCE_NAMES)}

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
            {
                "player": m.player,
                "type": ActionType(int(_ATYPE[m.flat])).name.lower(),
                "idx": int(_IDX[m.flat]),
                "target": int(_TARGET[m.flat]),
                **_describe(m),
            }
            for m in self.moves
        ]
        return json.dumps(doc, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> GameRecord:
        """Parse a record; the stable ``(type, idx, target)`` triple rules.

        Version-1 files (which stored flat indices) are migrated through
        their annotations; everything not needed to identify the move is
        ignored.
        """
        doc = json.loads(text)
        version = doc.get("version")
        if version not in (1, 2, _VERSION):
            raise ValueError(f"unsupported record version: {version!r}")
        winner = doc["winner"]
        return cls(
            seed=int(doc["seed"]),
            n_players=int(doc["n_players"]),
            number_placement=doc["number_placement"],
            moves=tuple(
                Move(
                    player=int(m["player"]),
                    flat=_move_flat(m, version),
                    dice=None if m.get("dice") is None else int(m["dice"]),
                )
                for m in doc["moves"]
            ),
            winner=None if winner is None else int(winner),
            meta=doc.get("meta", {}),
        )


def _move_flat(m: dict[str, Any], version: int) -> int:
    """The flat row of a serialized move, from its stable identifiers."""
    try:
        at = ActionType[str(m["type"]).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown action type {m['type']!r}") from exc
    # v1 stored only flat indices; v2 stored ProposeTrade in a retired 1:1
    # encoding. Both migrate through their (identical) annotations.
    direct = version >= 3 or (version == 2 and at is not ActionType.PROPOSE_TRADE)
    idx, target = (int(m["idx"]), int(m["target"])) if direct else _legacy_params(at, m)
    try:
        return _ROW[(int(at), idx, target)]
    except KeyError as exc:
        raise ValueError(
            f"move {m['type']!r} (idx={idx}, target={target}) is not a move "
            "of the current action table"
        ) from exc


def _legacy_params(at: ActionType, m: dict[str, Any]) -> tuple[int, int]:
    """``(idx, target)`` recovered from a v1/v2 move's annotations (v1 flat
    indices are stale — the table has grown — and v2 propose params used the
    retired 1:1 encoding)."""
    if at in _VERTEX_TYPES:
        return int(m["vertex"]), 0
    if at in _EDGE_TYPES:
        return int(m["edge"]), 0
    if at in _ROBBER_TYPES:
        return int(m["tile"]), int(m.get("victim", -1))
    if at in _RESOURCE_TYPES:
        return _RESOURCE_INDEX[m["resource"]], 0
    if at is ActionType.PLAY_YEAR_OF_PLENTY:
        first, second = m["resources"]
        return _RESOURCE_INDEX[first], _RESOURCE_INDEX[second]
    if at is ActionType.MARITIME_TRADE:
        return _RESOURCE_INDEX[m["give"]], _RESOURCE_INDEX[m["receive"]]
    if at is ActionType.PROPOSE_TRADE:
        return pack_trade_single(
            _RESOURCE_INDEX[m["give"]], _RESOURCE_INDEX[m["receive"]], int(m["partner"])
        )
    return 0, 0  # parameterless (roll / buy dev / road building / trade response)


def _packed_resource(packed: int) -> str:
    """The single resource name of a 1:1 packed count field (table rows only)."""
    counts = [(packed >> (_COUNT_BITS * r)) & _COUNT_MASK for r in range(N_RESOURCES)]
    (r,) = (i for i, c in enumerate(counts) if c)
    return _RESOURCE_NAMES[r]


def _describe(move: Move) -> dict[str, Any]:
    """The readable annotation for a move (decoded parameters by name)."""
    at = ActionType(int(_ATYPE[move.flat]))
    idx = int(_IDX[move.flat])
    target = int(_TARGET[move.flat])
    out: dict[str, Any] = {}
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
    elif at is ActionType.PROPOSE_TRADE:
        # Table propose rows are the 1:1 subset, so each side is one name.
        out["give"] = _packed_resource(idx)
        out["receive"] = _packed_resource(target >> _PARTNER_BITS)
        out["partner"] = target & ((1 << _PARTNER_BITS) - 1)
    elif at is ActionType.ROLL_DICE and move.dice is not None:
        out["dice"] = move.dice
    return out


def _make_env(record: GameRecord) -> BatchedSettlrlEnv:
    return BatchedSettlrlEnv(
        batch_size=1,
        seed=record.seed,
        auto_reset=False,
        n_players=record.n_players,
        number_placement=record.number_placement,
    )


def initial_board(record: GameRecord) -> Board:
    """The board a record's game opens on, before any move is played."""
    return _make_env(record).board


def _step_flat(env: BatchedSettlrlEnv, flat: int) -> None:
    at, params = flat_to_action(jnp.asarray([flat], dtype=jnp.int32))
    env.step(at, params)


def _terminal(env: BatchedSettlrlEnv) -> bool:
    return bool(np.asarray(env.terminations[0]).all())


def _winner(env: BatchedSettlrlEnv) -> int | None:
    # The winner is the terminal state's current player (a win happens only on
    # the winner's own turn), not the VP argmax: an off-turn player may also
    # sit at 10+ without having won.
    if _terminal(env):
        return int(np.asarray(env.board[1].current_player[0]))
    return None


# An action chooser for record_game: given a PRNG key and the live single-game
# env (read flat_mask / observe / board), the flat action to play.
Act = Callable[[KeyScalar, BatchedSettlrlEnv], int]


def _uniform_random(key: KeyScalar, env: BatchedSettlrlEnv) -> int:
    """One type-first :func:`random_flat` draw over the acting player's moves."""
    return int(random_flat(key, env.flat_mask()[0]))


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

    ``act`` chooses each move (default: random type-first legal play,
    :func:`settlrl_engine.mechanics.flat.random_flat`); a returned illegal
    action raises ``ValueError``, and a game not finishing within
    ``max_moves`` raises ``RuntimeError``.
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
