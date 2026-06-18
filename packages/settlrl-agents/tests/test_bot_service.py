"""Tests for the one-bot service (the bot wire protocol).

A service hosts a single bot and tracks each game in flight: ``/act`` applies the
moves it has not seen yet (structured, in board coordinates) and returns the
acting seat's move. These drive it through a ``TestClient``.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from settlrl_agents.service.app import create_app
from settlrl_agents.service.bots import make_bot
from settlrl_game.actions import flat_for_move, move_for_flat
from settlrl_game.botproto import MoveModel
from settlrl_game.session import GameSession, GameSetup


def _move(resp: Any) -> MoveModel:
    """The MoveModel from an /act response."""
    return MoveModel(**resp.json()["move"])


@pytest.fixture()
def client() -> Iterator[TestClient]:
    yield TestClient(create_app(make_bot("random")))


def _all_bot_setup() -> dict[str, Any]:
    return GameSetup(
        seed=3, n_players=2, number_placement="random", seats=["random", "random"]
    ).to_dict()


def _mirror() -> GameSession:
    """A legality oracle for the same position; seat kinds don't affect what's
    legal, so a plain all-human session mirrors the bot game's legal moves."""
    return GameSession(seed=3, n_players=2, seats=["human", "human"])


def _structured(flats: list[int]) -> list[dict[str, Any]]:
    return [move_for_flat(f).model_dump() for f in flats]


def test_info_reports_the_bot(client: TestClient) -> None:
    info = client.get("/info").json()
    assert info["name"] == "random"
    assert 2 in info["counts"]


def test_act_returns_a_legal_move_for_the_acting_seat(client: TestClient) -> None:
    body = {
        "game_id": "g1",
        "seat": 0,
        "setup": _all_bot_setup(),
        "base": 0,
        "moves": [],
    }
    resp = client.post("/act", json=body)
    assert resp.status_code == 200, resp.text
    flat = flat_for_move(_move(resp))
    assert flat in {int(f) for f in _mirror().legal_flat()}


def test_act_advances_incrementally(client: TestClient) -> None:
    """Feeding only each new move (a growing ``base``) keeps producing legal
    moves — exercising the service's incremental game tracking."""
    setup = _all_bot_setup()
    mirror = _mirror()
    service_count = 0  # moves the service already holds
    for _ in range(8):
        seat = mirror.acting_seat()
        history = mirror.moves_flat()
        body = {
            "game_id": "g2",
            "seat": seat,
            "setup": setup,
            "base": service_count,
            "moves": _structured(history[service_count:]),
        }
        resp = client.post("/act", json=body)
        assert resp.status_code == 200, resp.text
        service_count = len(history)  # the service applied the tail we sent
        flat = flat_for_move(_move(resp))
        assert flat in {int(f) for f in mirror.legal_flat()}
        mirror.apply(flat)


def test_act_resync_when_base_is_ahead(client: TestClient) -> None:
    body = {
        "game_id": "g3",
        "seat": 0,
        "setup": _all_bot_setup(),
        "base": 999,
        "moves": [],
    }
    resp = client.post("/act", json=body)
    assert resp.status_code == 409
    assert resp.json()["detail"] == {"resync": True, "have": 0}


def test_act_409_when_seat_not_acting(client: TestClient) -> None:
    body = {
        "game_id": "g4",
        "seat": 1,
        "setup": _all_bot_setup(),
        "base": 0,
        "moves": [],
    }
    assert client.post("/act", json=body).status_code == 409


def test_act_422_on_bad_setup(client: TestClient) -> None:
    body = {"game_id": "g5", "seat": 0, "setup": {"seed": 1}, "base": 0, "moves": []}
    assert client.post("/act", json=body).status_code == 422
