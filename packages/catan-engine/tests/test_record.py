"""Game records: generation, JSON roundtrip, and validated deterministic replay.

A randomly-played game is recorded once per player count (module-scoped -- a
full game is thousands of steps) and shared across the tests; tampering tests
corrupt early moves so replay fails fast.
"""

import dataclasses
import json

import jax
import numpy as np
import pytest

from catan_engine.env import BatchedCatanEnv
from catan_engine.record import GameRecord, Move, ReplayError, record_game, replay


@pytest.fixture(scope="module")
def record4() -> GameRecord:
    return record_game(seed=0, meta={"note": "test game"})


@pytest.fixture(scope="module")
def record2() -> GameRecord:
    return record_game(seed=1, n_players=2, number_placement="spiral")


def test_generated_game_completes(record4: GameRecord) -> None:
    assert record4.winner is not None
    assert len(record4.moves) > 10
    # The transcript opens with the setup snake and records dice outcomes.
    assert {m.player for m in record4.moves} == {0, 1, 2, 3}
    rolls = [m.dice for m in record4.moves if m.dice is not None]
    assert rolls and all(2 <= r <= 12 for r in rolls)


def test_json_roundtrip(record4: GameRecord) -> None:
    assert GameRecord.from_json(record4.to_json()) == record4


def test_json_is_readable(record4: GameRecord) -> None:
    doc = json.loads(record4.to_json())
    assert doc["version"] == 1
    assert doc["meta"] == {"note": "test game"}
    moves = doc["moves"]
    first = moves[0]
    # Setup opens with a settlement: annotated with its type and vertex.
    assert first["type"] == "setup_settlement" and isinstance(first["vertex"], int)
    types = {m["type"] for m in moves}
    assert {"setup_road", "roll_dice", "end_turn"} <= types
    roll = next(m for m in moves if m["type"] == "roll_dice")
    assert 2 <= roll["dice"] <= 12


def test_replay_reproduces_the_game(record2: GameRecord) -> None:
    boards = list(replay(record2))
    assert len(boards) == len(record2.moves)
    # The final state is bit-identical to a second replay (determinism).
    final, again = boards[-1], list(replay(record2))[-1]
    assert all(
        bool((a == b).all())
        for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(again))
    )


def test_replay_rejects_wrong_player(record4: GameRecord) -> None:
    bad = dataclasses.replace(
        record4,
        moves=(dataclasses.replace(record4.moves[0], player=3),)
        + record4.moves[1:],
    )
    with pytest.raises(ReplayError, match="player"):
        next(replay(bad))


def test_replay_rejects_illegal_move(record4: GameRecord) -> None:
    # A second action on the same vertex as move 0 cannot be legal at move 1.
    bad = dataclasses.replace(
        record4,
        moves=(record4.moves[0], record4.moves[0]) + record4.moves[2:],
    )
    steps = replay(bad)
    next(steps)
    with pytest.raises(ReplayError, match="not legal"):
        next(steps)


def test_replay_rejects_wrong_dice(record4: GameRecord) -> None:
    idx, roll = next(
        (i, m) for i, m in enumerate(record4.moves) if m.dice is not None
    )
    assert roll.dice is not None
    wrong = dataclasses.replace(roll, dice=roll.dice % 12 + 2)
    bad = dataclasses.replace(
        record4, moves=record4.moves[:idx] + (wrong,) + record4.moves[idx + 1 :]
    )
    with pytest.raises(ReplayError, match="rolled"):
        for _ in replay(bad):
            pass


def test_replay_rejects_wrong_winner(record2: GameRecord) -> None:
    assert record2.winner is not None
    bad = dataclasses.replace(record2, winner=(record2.winner + 1) % 2)
    with pytest.raises(ReplayError, match="winner"):
        for _ in replay(bad):
            pass


def test_record_game_rejects_illegal_act() -> None:
    # An act that picks the first currently-*illegal* flat action.
    def bad_act(key: jax.Array, env: BatchedCatanEnv) -> int:
        mask = np.asarray(env.flat_mask()[0]).astype(bool)
        return int(np.flatnonzero(~mask)[0])

    with pytest.raises(ValueError, match="illegal"):
        record_game(seed=0, act=bad_act)


def test_partial_record_replays_without_winner_check(record2: GameRecord) -> None:
    partial = dataclasses.replace(record2, moves=record2.moves[:8], winner=None)
    assert len(list(replay(partial))) == 8


def test_from_json_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="version"):
        GameRecord.from_json('{"version": 99}')


def test_move_defaults() -> None:
    assert Move(player=0, flat=1).dice is None
