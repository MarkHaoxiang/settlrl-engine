"""Tests for the standalone bot service (the standardized bot API).

The service is a pure function of a game record: given a setup and the flat
moves so far, it replays and returns the acting bot's move. These drive it
through its own TestClient.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from settlrl_render.bots.bot_service import create_bot_app
from settlrl_render.game.session import GameSession, GameSetup


@pytest.fixture()
def client() -> Iterator[TestClient]:
    yield TestClient(create_bot_app())


def _all_bot_setup() -> dict[str, object]:
    # The game server stores bot seats verbatim, so build the setup dict directly
    # (a GameSession would need the kinds in external_kinds, which the service has).
    return GameSetup(
        seed=3, n_players=2, number_placement="random", seats=["random", "random"]
    ).to_dict()


def _mirror() -> GameSession:
    """A legality oracle for the same position; seat kinds don't affect what's
    legal, so a plain all-human session mirrors the bot game's legal moves."""
    return GameSession(seed=3, n_players=2, seats=["human", "human"])


def test_catalog_lists_built_in_kinds(client: TestClient) -> None:
    catalog = client.get("/catalog").json()
    assert "random" in catalog
    assert 2 in catalog["random"]["counts"]


def test_act_returns_a_legal_move_for_the_acting_seat(client: TestClient) -> None:
    setup = _all_bot_setup()
    # From the opening of an all-bot game, seat 0 (a bot) is acting.
    resp = client.post(
        "/act", json={"game_id": "g1", "setup": setup, "moves": [], "seat": 0}
    )
    assert resp.status_code == 200, resp.text
    flat = resp.json()["flat"]
    # The returned move is legal in a freshly replayed copy of the same position.
    assert flat in {int(f) for f in _mirror().legal_flat()}


def test_act_advances_move_by_move(client: TestClient) -> None:
    """Feeding back each chosen move (growing the trace) keeps producing legal
    moves — exercising the service's incremental replay cache."""
    setup = _all_bot_setup()
    mirror = _mirror()
    moves: list[int] = []
    for _ in range(8):
        seat = mirror.acting_seat()
        resp = client.post(
            "/act", json={"game_id": "g2", "setup": setup, "moves": moves, "seat": seat}
        )
        assert resp.status_code == 200, resp.text
        flat = resp.json()["flat"]
        assert flat in {int(f) for f in mirror.legal_flat()}
        mirror.apply(flat)
        moves.append(flat)


def test_act_409_when_seat_not_acting(client: TestClient) -> None:
    setup = _all_bot_setup()
    resp = client.post(
        "/act", json={"game_id": "g3", "setup": setup, "moves": [], "seat": 1}
    )
    assert resp.status_code == 409


def test_act_422_on_bad_setup(client: TestClient) -> None:
    resp = client.post(
        "/act", json={"game_id": "g4", "setup": {"seed": 1}, "moves": [], "seat": 0}
    )
    assert resp.status_code == 422
