"""One live game per account, and leaving / closing an unstarted lobby.

A signed-in player may hold a seat in only one non-terminal game at a time: a
second create / join / quick-match is refused (with the existing game's id, so
the client can offer to resume). A lobby that hasn't started can be left — the
host closes it for everyone, any other participant just frees their seat. Guests
carry no server identity, so the limit is not enforced for them here.
"""

from collections.abc import Iterator

import pytest
from _helpers import bot_registry
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


def _vs_bots(client: TestClient, headers: dict[str, str], seed: int = 0) -> str:
    """A 1-human-vs-bots game the account holds (seat 0) — ready at once."""
    return str(
        client.post(
            "/api/games",
            json={"seed": seed, "seats": ["human", "random", "random", "random"]},
            headers=headers,
        ).json()["id"]
    )


def _open_lobby(client: TestClient, headers: dict[str, str]) -> str:
    """A 4-human lobby the host holds (seat 0), seats 1-3 open — so it stays
    unstarted (and reconfigurable / closable) even after one more player joins."""
    return str(
        client.post(
            "/api/games",
            json={
                "n_players": 4,
                "seats": ["human", "human", "human", "human"],
                "claim": "first",
            },
            headers=headers,
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


def test_create_blocked_while_in_a_live_game(client: TestClient) -> None:
    host = _bearer(_token(client, "a@example.com"))
    game = _vs_bots(client, host)
    res = client.post(
        "/api/games",
        json={"seed": 1, "seats": ["human", "random", "random", "random"]},
        headers=host,
    )
    assert res.status_code == 409
    assert res.json()["detail"]["game_id"] == game


def test_join_blocked_while_in_another_game(client: TestClient) -> None:
    a = _bearer(_token(client, "a@example.com"))
    b = _bearer(_token(client, "b@example.com"))
    mine = _vs_bots(client, a)
    other = _open_lobby(client, b)  # b hosts, seat 1 open
    res = client.post(f"/api/games/{other}/join", json={"seat": 1}, headers=a)
    assert res.status_code == 409
    assert res.json()["detail"]["game_id"] == mine


def test_guest_without_a_client_id_is_unrestricted(client: TestClient) -> None:
    # No X-Client-Id (an old client) has no server identity, so the limit can't
    # apply — both creates succeed.
    assert client.post("/api/games", json={"seed": 0}).status_code == 200
    assert client.post("/api/games", json={"seed": 1}).status_code == 200


def test_guest_browser_is_held_to_one_game(client: TestClient) -> None:
    hdr = {"X-Client-Id": "browser-1"}
    game = client.post("/api/games", json={"seed": 0}, headers=hdr).json()["id"]
    # A second create from the same browser is refused, pointing at the first.
    res = client.post("/api/games", json={"seed": 1}, headers=hdr)
    assert res.status_code == 409 and res.json()["detail"]["game_id"] == game
    # As is joining someone else's open lobby.
    other = _open_lobby(client, _bearer(_token(client, "host@example.com")))
    join = client.post(f"/api/games/{other}/join", json={"seat": 1}, headers=hdr)
    assert join.status_code == 409
    # A different browser is unaffected.
    assert (
        client.post(
            "/api/games", json={"seed": 2}, headers={"X-Client-Id": "browser-2"}
        ).status_code
        == 200
    )


def test_host_close_removes_the_lobby_and_bounces_others(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    other = _bearer(_token(client, "other@example.com"))
    game = _open_lobby(client, host)
    client.post(f"/api/games/{game}/join", json={"seat": 1}, headers=other)

    res = client.post(f"/api/games/{game}/leave", headers=host)
    assert res.status_code == 200 and res.json()["closed"] is True
    # The game is gone for everyone.
    assert client.get(f"/api/games/{game}", headers=other).status_code == 404
    # And the host can host a fresh one now.
    assert client.post("/api/games", json={"seed": 9}, headers=host).status_code == 200


def test_participant_leave_frees_the_seat_and_the_limit(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    other = _bearer(_token(client, "other@example.com"))
    game = _open_lobby(client, host)
    client.post(f"/api/games/{game}/join", json={"seat": 1}, headers=other)

    res = client.post(f"/api/games/{game}/leave", headers=other)
    assert res.status_code == 200 and res.json()["closed"] is False
    # The lobby survives for the host with seat 1 open again.
    snap = client.get(f"/api/games/{game}", headers=host).json()
    assert snap["seats_claimed"] == [0]
    # The leaver is free to host their own game.
    assert client.post("/api/games", json={"seed": 7}, headers=other).status_code == 200


def test_leave_requires_a_seat(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    outsider = _bearer(_token(client, "x@example.com"))
    game = _open_lobby(client, host)
    assert client.post(f"/api/games/{game}/leave", headers=outsider).status_code == 403
    assert client.post(f"/api/games/{game}/leave").status_code == 403


def test_leave_rejected_once_the_game_has_started(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    game = _vs_bots(client, host)  # ready at once (vs bots)
    assert client.post(f"/api/games/{game}/leave", headers=host).status_code == 409
