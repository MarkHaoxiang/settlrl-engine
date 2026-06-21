"""Reconfiguring a lobby's game before it starts.

The host edits the map / player count / VP / lobby flags from the lobby room;
the board is rebuilt in place so the game id, the surviving seat claims, and the
chat all carry over. These drive that through ``POST /api/games/{id}/configure``.
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


@pytest.fixture()
def client_no_random() -> Iterator[TestClient]:
    """A server whose only bot service is ``greedy`` — no ``random`` kind exists,
    the production shape where the count-growth seat fill must not assume one."""
    reg = bot_registry(["greedy"])
    with TestClient(create_app(GameRegistry(), providers=reg)) as c:
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


def _host_open_game(client: TestClient, headers: dict[str, str]) -> str:
    """A 2-human game the host holds (seat 0), seat 1 left open — the lobby-room
    shape, which never auto-starts so it stays reconfigurable."""
    return str(
        client.post(
            "/api/games",
            json={
                "seed": 1,
                "n_players": 2,
                "seats": ["human", "human"],
                "claim": "first",
            },
            headers=headers,
        ).json()["id"]
    )


def test_configure_reseeds_and_sets_vp_keeping_claim_and_chat(
    client: TestClient,
) -> None:
    host = _bearer(_token(client, "host@example.com"))
    game = _host_open_game(client, host)
    client.post(
        f"/api/games/{game}/chat", json={"text": "hi all", "player": 0}, headers=host
    )

    before = client.get(f"/api/games/{game}", headers=host).json()
    res = client.post(
        f"/api/games/{game}/configure",
        json={"seed": 999, "victory_points_to_win": 12},
        headers=host,
    )
    assert res.status_code == 200
    snap = res.json()
    # The board was rebuilt (a different seed lays out different tiles) but the
    # host still owns seat 0, the VP target changed, and the chat survived.
    assert snap["your_seats"] == [0]
    assert snap["status"]["victory_points_to_win"] == 12
    assert snap["board"] != before["board"]
    assert any(e["kind"] == "chat" and e["text"] == "hi all" for e in snap["log"])
    # Seat 1 is still open, so the game is still waiting (reconfigurable).
    assert snap["seats_claimed"] == [0]


def test_configure_to_two_players_drops_the_now_gone_claims(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    # A 4-human hotseat the host holds entirely (seats 0-3 all claimed).
    game = client.post(
        "/api/games",
        json={"n_players": 4, "seats": ["human"] * 4, "claim": "all"},
        headers=host,
    ).json()["id"]
    assert client.get(f"/api/games/{game}", headers=host).json()["your_seats"] == [
        0,
        1,
        2,
        3,
    ]

    snap = client.post(
        f"/api/games/{game}/configure", json={"n_players": 2}, headers=host
    ).json()
    assert snap["status"]["seats"] == ["human", "human"]
    assert snap["your_seats"] == [0, 1]  # seats 2 and 3 (and their claims) are gone


def test_configure_grows_count_opening_human_seats(
    client_no_random: TestClient,
) -> None:
    # Growing the count without naming seats must open the new ones for humans —
    # not assume a "random" bot service that isn't registered here (it would 422).
    host = _bearer(_token(client_no_random, "host@example.com"))
    game = _host_open_game(client_no_random, host)

    res = client_no_random.post(
        f"/api/games/{game}/configure", json={"n_players": 4}, headers=host
    )
    assert res.status_code == 200, res.json()
    assert res.json()["status"]["seats"] == ["human", "human", "human", "human"]


def test_configure_updates_lobby_flags(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    game = _host_open_game(client, host)
    assert client.get("/api/lobby").json() == []  # not listed yet

    client.post(
        f"/api/games/{game}/configure",
        json={"listed": True, "searchable": True},
        headers=host,
    )
    (row,) = client.get("/api/lobby").json()
    assert row["id"] == game and row["searchable"] is True


def test_configure_is_host_only(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    game = _host_open_game(client, host)
    # A second account joins seat 1 but does not own seat 0, so it cannot configure.
    other = _bearer(_token(client, "other@example.com"))
    client.post(f"/api/games/{game}/join", json={"seat": 1}, headers=other)
    assert (
        client.post(
            f"/api/games/{game}/configure", json={"seed": 5}, headers=other
        ).status_code
        == 403
    )
    # Anonymous, holding no seat, is likewise refused.
    assert (
        client.post(f"/api/games/{game}/configure", json={"seed": 5}).status_code == 403
    )


def test_configure_rejected_once_a_move_is_played(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    # One human + bots: ready from the start, so the host can play immediately.
    game = client.post(
        "/api/games",
        json={
            "seed": 0,
            "seats": ["human", "random", "random", "random"],
            "claim": "first",
        },
        headers=host,
    ).json()["id"]
    snap = client.get(f"/api/games/{game}", headers=host).json()
    client.post(
        f"/api/games/{game}/action",
        json={"flat": snap["actions"][0]["flat"]},
        headers=host,
    )
    assert (
        client.post(
            f"/api/games/{game}/configure", json={"seed": 7}, headers=host
        ).status_code
        == 409
    )


def test_create_honours_a_custom_win_target(client: TestClient) -> None:
    host = _bearer(_token(client, "host@example.com"))
    game = client.post(
        "/api/games",
        json={"n_players": 2, "seats": ["human", "human"], "victory_points_to_win": 8},
        headers=host,
    ).json()["id"]
    snap = client.get(f"/api/games/{game}", headers=host).json()
    assert snap["status"]["victory_points_to_win"] == 8
