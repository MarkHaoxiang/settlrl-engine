"""The flat action space and its structured (MoveModel) wire form.

The flat index is the engine's opaque action id; the bot wire speaks the
structured :class:`MoveModel` in board coordinates. These pin the translation
between them (a stable contract across the wire) and the legal-move enumeration.
"""

import pytest
from settlrl_game.actions import (
    N_FLAT,
    flat_for_move,
    legal_flats,
    legal_moves,
    move_for_flat,
)
from settlrl_game.botproto import MoveModel
from settlrl_game.session import GameSession


def test_flat_move_round_trips_for_every_action() -> None:
    for flat in range(N_FLAT):
        assert flat_for_move(move_for_flat(flat)) == flat


def test_legal_moves_match_legal_flats() -> None:
    game = GameSession(seed=7, n_players=3, seats=["human"] * 3).game
    by_move = {flat_for_move(m) for m in legal_moves(game)}
    assert by_move == set(legal_flats(game))


def test_move_carries_board_coordinates() -> None:
    # A settlement names a vertex in cube coordinates; a road an edge of two.
    settle = next(
        m
        for m in (move_for_flat(f) for f in range(N_FLAT))
        if m.type == "setup_settlement"
    )
    assert settle.vertex is not None and settle.edge is None
    road = next(
        m for m in (move_for_flat(f) for f in range(N_FLAT)) if m.type == "setup_road"
    )
    assert road.edge is not None and road.vertex is None


def test_flat_for_move_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        flat_for_move(MoveModel(type="teleport"))
    with pytest.raises(ValueError):
        flat_for_move(MoveModel(type="setup_settlement"))  # missing vertex
    with pytest.raises(ValueError):
        flat_for_move(MoveModel(type="discard", resource="gold"))  # not a resource
