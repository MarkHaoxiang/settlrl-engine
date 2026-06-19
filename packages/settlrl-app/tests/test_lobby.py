"""The public lobby and owner seat control.

Listed games surface in ``GET /api/lobby`` with their open seats; a participant
can retarget an unclaimed seat (open <-> bot) before play begins.
"""

from collections.abc import Iterator

import pytest
from _helpers import bot_registry
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app(GameRegistry(), providers=bot_registry())) as c:
        yield c


def _create(client: TestClient, **body: object) -> dict[str, object]:
    return client.post(  # type: ignore[no-any-return]
        "/api/games",
        json={"seed": 0, "n_players": 2, "seats": ["human", "human"], **body},
    ).json()


def _hdr(doc: dict[str, object]) -> dict[str, str]:
    tokens: dict[str, str] = dict(doc["tokens"])  # type: ignore[call-overload]
    return {"X-Seat-Tokens": ",".join(tokens.values())}


def test_only_listed_games_with_open_seats_appear(client: TestClient) -> None:
    _create(client, claim="first")  # unlisted (default)
    assert client.get("/api/lobby").json() == []

    doc = _create(client, seed=1, claim="first", listed=True)
    lobby = client.get("/api/lobby").json()
    assert [g["id"] for g in lobby] == [doc["id"]]
    assert lobby[0]["open_seats"] == 1 and lobby[0]["n_players"] == 2
    assert lobby[0]["claimed"] == [0]


def test_seat_control_turns_an_open_seat_into_a_bot(client: TestClient) -> None:
    doc = _create(client, claim="first", listed=True)
    game = doc["id"]
    hdr = _hdr(doc)

    r = client.post(
        f"/api/games/{game}/seats", json={"seat": 1, "kind": "random"}, headers=hdr
    )
    assert r.status_code == 200
    assert r.json()["status"]["seats"] == ["human", "random"]
    # No open human seats left -> it drops out of the lobby and can start.
    assert client.get("/api/lobby").json() == []


def test_seat_control_guards(client: TestClient) -> None:
    doc = _create(client, claim="first", listed=True)
    game = doc["id"]
    hdr = _hdr(doc)

    # An outsider (no seat) can't change seats.
    assert (
        client.post(
            f"/api/games/{game}/seats", json={"seat": 1, "kind": "random"}
        ).status_code
        == 403
    )
    # A claimed seat can't be retargeted.
    assert (
        client.post(
            f"/api/games/{game}/seats", json={"seat": 0, "kind": "random"}, headers=hdr
        ).status_code
        == 409
    )
    # An unknown bot kind is rejected.
    assert (
        client.post(
            f"/api/games/{game}/seats", json={"seat": 1, "kind": "clever"}, headers=hdr
        ).status_code
        == 422
    )
