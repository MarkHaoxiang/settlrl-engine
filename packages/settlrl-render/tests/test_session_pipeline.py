"""End-to-end sync through a live :class:`GameSession`.

These exercise the full pipeline the server uses — drive the AEC env, decode the
legal moves, apply one, and convert the resulting board — asserting the pieces
stay consistent with each other and with the engine. The decisive check is
:func:`test_applied_placement_lands_at_decoded_coordinate`: it proves the
renderer's decoded coordinate is *where the engine actually places*, tying the
decode layer to real engine behaviour rather than to a parallel reconstruction.
"""

import pytest
from settlrl_engine.board.state import GamePhase
from settlrl_engine.env import N_FLAT
from settlrl_render.api.actions import decode_actions
from settlrl_render.api.convert import board_to_model
from settlrl_render.api.models import BoardModel, CubeModel
from settlrl_render.game.session import GameSession, IllegalActionError


def _cube(c: CubeModel) -> tuple[int, int, int]:
    return (c.q, c.r, c.s)


def test_session_board_converts() -> None:
    sess = GameSession(seed=0)
    model = board_to_model(sess.board)
    assert isinstance(model, BoardModel)
    assert len(model.tiles) == 19
    assert len(model.ports) == 9
    assert len(model.players) == 4


def test_status_phase_is_a_real_phase() -> None:
    sess = GameSession(seed=0)
    valid = {p.name.lower() for p in GamePhase}
    assert sess.status().phase in valid


def test_legal_actions_decode_and_roundtrip() -> None:
    # Every legal move decodes, and its flat id is genuinely legal.
    sess = GameSession(seed=0)
    legal = [int(f) for f in sess.legal_flat()]
    assert legal, "human should have legal moves at game start (setup phase)"
    legal_set = set(legal)
    for m in decode_actions(legal):
        assert m.flat in legal_set


def test_illegal_action_rejected() -> None:
    sess = GameSession(seed=0)
    legal = {int(f) for f in sess.legal_flat()}
    illegal = next(f for f in range(N_FLAT) if f not in legal)
    with pytest.raises(IllegalActionError):
        sess.apply(illegal)


def test_applied_placement_lands_at_decoded_coordinate() -> None:
    # In setup the human's first move is a settlement. Decode the legal
    # settlements, apply one, and confirm board_to_model shows a building owned
    # by seat 0 at exactly the decoded vertex — the renderer and engine agree on
    # what that action index means on the board.
    sess = GameSession(seed=0)
    assert sess.status().phase == "setup_settlement"

    actions = decode_actions([int(f) for f in sess.legal_flat()])
    settlement = next(a for a in actions if a.type == "setup_settlement")
    assert settlement.vertex is not None
    target = _cube(settlement.vertex)

    sess.apply(settlement.flat)

    model = board_to_model(sess.board)
    placed = [b for b in model.buildings if b.player == 0]
    assert any(_cube(b.cube) == target for b in placed), (
        f"expected a seat-0 building at {target}, got {[_cube(b.cube) for b in placed]}"
    )
