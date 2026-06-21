"""One place at a time: a player may hold a seat in only one lobby or live game.

A second host / join / quick-match is refused with a 409 carrying the existing
place's id and kind (so the client can offer to resume it). Accounts are keyed by
id, guests by their X-Client-Id browser; a guest with no id sent is unrestricted.
"""

from collections.abc import Iterator

import pytest
from _helpers import bot_registry, start_game
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app
from settlrl_game.session import GameSession


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app(GameRegistry(), providers=bot_registry())) as c:
        yield c


def _token(client: TestClient, email: str) -> str:
    client.post("/api/auth/register", json={"email": email, "password": "password1"})
    return str(
        client.post(
            "/api/auth/login", data={"username": email, "password": "password1"}
        ).json()["access_token"]
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _open_lobby(client: TestClient, headers: dict[str, str]) -> str:
    return str(
        client.post(
            "/api/lobbies", json={"mode": "online", "n_players": 4}, headers=headers
        ).json()["id"]
    )


def test_live_game_for_user_excludes_unowned() -> None:
    reg = GameRegistry()
    session = GameSession(seed=0, n_players=2)
    session.reset(0, seats=["human", "human"])
    handle = reg.create(session)
    handle.claim(0, user_id="u1")
    assert reg.live_game_for_user("u1") is handle
    assert reg.live_game_for_user("u2") is None


def test_hosting_blocked_while_in_a_live_game(client: TestClient) -> None:
    host = _bearer(_token(client, "a@example.com"))
    game_id, _ = start_game(client, ["human", "random"], headers=host)
    res = client.post("/api/lobbies", json={"mode": "online"}, headers=host)
    assert res.status_code == 409
    assert (
        res.json()["detail"]["id"] == game_id and res.json()["detail"]["kind"] == "game"
    )


def test_joining_blocked_while_in_another_game(client: TestClient) -> None:
    a = _bearer(_token(client, "a@example.com"))
    b = _bearer(_token(client, "b@example.com"))
    game_id, _ = start_game(client, ["human", "random"], headers=a)
    other = _open_lobby(client, b)
    res = client.post(f"/api/lobbies/{other}/join", json={"seat": 1}, headers=a)
    assert res.status_code == 409 and res.json()["detail"]["id"] == game_id


def test_hosting_blocked_while_in_a_lobby(client: TestClient) -> None:
    acct = _bearer(_token(client, "a@example.com"))
    lobby = _open_lobby(client, acct)
    res = client.post("/api/lobbies", json={"mode": "online"}, headers=acct)
    assert res.status_code == 409
    assert (
        res.json()["detail"]["id"] == lobby and res.json()["detail"]["kind"] == "lobby"
    )


def test_guest_without_a_client_id_is_unrestricted(client: TestClient) -> None:
    # No X-Client-Id (an old client) has no server identity, so the limit can't
    # apply — both hosts succeed.
    assert client.post("/api/lobbies", json={"mode": "online"}).status_code == 200
    assert client.post("/api/lobbies", json={"mode": "online"}).status_code == 200


def test_guest_browser_is_held_to_one(client: TestClient) -> None:
    hdr = {"X-Client-Id": "browser-1"}
    lobby = client.post("/api/lobbies", json={"mode": "online"}, headers=hdr).json()[
        "id"
    ]
    # A second host from the same browser is refused, pointing at the first.
    res = client.post("/api/lobbies", json={"mode": "online"}, headers=hdr)
    assert res.status_code == 409 and res.json()["detail"]["id"] == lobby
    # As is joining someone else's table.
    other = _open_lobby(client, _bearer(_token(client, "host@example.com")))
    assert (
        client.post(
            f"/api/lobbies/{other}/join", json={"seat": 1}, headers=hdr
        ).status_code
        == 409
    )
    # A different browser is unaffected.
    assert (
        client.post(
            "/api/lobbies",
            json={"mode": "online"},
            headers={"X-Client-Id": "browser-2"},
        ).status_code
        == 200
    )
