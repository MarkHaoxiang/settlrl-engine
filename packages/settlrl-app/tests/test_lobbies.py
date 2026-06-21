"""Pre-game lobbies and their materialisation into a game.

A lobby holds only configuration and claims — no engine — until the host starts
it, which is allowed only once every seat is decided. Starting copies the
claimed seats into a fresh game, so each lobby seat token keeps working there.
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


def _token(client: TestClient, email: str) -> str:
    client.post("/api/auth/register", json={"email": email, "password": "password1"})
    return str(
        client.post(
            "/api/auth/login", data={"username": email, "password": "password1"}
        ).json()["access_token"]
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seat(created: dict[str, object]) -> dict[str, str]:
    held: dict[str, str] = dict(created["tokens"])  # type: ignore[call-overload]
    return {"X-Seat-Tokens": ",".join(held.values())}


def _host(client: TestClient, **body: object) -> dict[str, object]:
    return client.post("/api/lobbies", json=body).json()  # type: ignore[no-any-return]


def test_online_host_holds_seat_zero_and_the_rest_stay_open(client: TestClient) -> None:
    room = _host(client, mode="online", n_players=4)
    snap = client.get(f"/api/lobbies/{room['id']}", headers=_seat(room)).json()
    assert snap["mode"] == "online"
    assert snap["your_seats"] == [0] and snap["seats_claimed"] == [0]
    assert snap["ready"] is False  # seats 1-3 open, so it can't start


def test_hotseat_host_holds_every_seat_and_is_ready(client: TestClient) -> None:
    room = _host(client, mode="hotseat", n_players=2)
    snap = client.get(f"/api/lobbies/{room['id']}", headers=_seat(room)).json()
    assert snap["seats_claimed"] == [0, 1] and snap["ready"] is True


def test_start_is_rejected_while_a_human_seat_is_open(client: TestClient) -> None:
    room = _host(client, mode="online", n_players=2)
    res = client.post(f"/api/lobbies/{room['id']}/start", json={}, headers=_seat(room))
    assert res.status_code == 409  # a half-empty table can't begin


def test_bot_filling_then_starting_materialises_a_game(client: TestClient) -> None:
    room = _host(client, mode="online", n_players=2)
    rid, seat = room["id"], _seat(room)
    # Fill the open seat with a bot — now every seat is decided.
    client.post(
        f"/api/lobbies/{rid}/seats", json={"seat": 1, "kind": "random"}, headers=seat
    )
    started = client.post(f"/api/lobbies/{rid}/start", json={}, headers=seat)
    assert started.status_code == 200
    game_id = started.json()["game_id"]
    # The host's lobby seat token still owns seat 0 in the materialised game.
    game = client.get(f"/api/games/{game_id}", headers=seat).json()
    assert game["your_seats"] == [0]
    assert game["status"]["seats"] == ["human", "random"]
    # The lobby's SSE snapshot now points everyone at the game.
    assert (
        client.get(f"/api/lobbies/{rid}", headers=seat).json()["started_game_id"]
        == game_id
    )


def test_a_joiner_takes_an_open_seat(client: TestClient) -> None:
    room = _host(client, mode="online", n_players=2)
    joined = client.post(f"/api/lobbies/{room['id']}/join", json={}).json()
    assert list(joined["tokens"]) == ["1"]  # the first open human seat
    snap = client.get(f"/api/lobbies/{room['id']}", headers=_seat(room)).json()
    assert snap["seats_claimed"] == [0, 1] and snap["ready"] is True


def test_host_closes_the_lobby_for_everyone(client: TestClient) -> None:
    room = _host(client, mode="online", n_players=4)
    client.post(f"/api/lobbies/{room['id']}/join", json={"seat": 1})
    res = client.post(f"/api/lobbies/{room['id']}/leave", headers=_seat(room))
    assert res.status_code == 204
    assert client.get(f"/api/lobbies/{room['id']}").status_code == 404


def test_a_participant_leaves_freeing_their_seat(client: TestClient) -> None:
    room = _host(client, mode="online", n_players=4)
    joiner = client.post(f"/api/lobbies/{room['id']}/join", json={"seat": 1}).json()
    res = client.post(f"/api/lobbies/{room['id']}/leave", headers=_seat(joiner))
    assert res.status_code == 204
    snap = client.get(f"/api/lobbies/{room['id']}", headers=_seat(room)).json()
    assert snap["seats_claimed"] == [0]  # seat 1 is open again, the lobby lives


def test_only_the_host_starts_or_configures(client: TestClient) -> None:
    room = _host(client, mode="online", n_players=4)
    joiner = client.post(f"/api/lobbies/{room['id']}/join", json={"seat": 1}).json()
    assert (
        client.post(
            f"/api/lobbies/{room['id']}/configure",
            json={"seed": 5},
            headers=_seat(joiner),
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/lobbies/{room['id']}/start", json={}, headers=_seat(joiner)
        ).status_code
        == 403
    )


def test_hosting_a_second_lobby_is_refused(client: TestClient) -> None:
    acct = _bearer(_token(client, "a@example.com"))
    client.post("/api/lobbies", json={"mode": "online", "n_players": 2}, headers=acct)
    assert (
        client.post("/api/lobbies", json={"mode": "online"}, headers=acct).status_code
        == 409
    )


def test_listed_lobbies_appear_in_the_public_list(client: TestClient) -> None:
    acct = _bearer(_token(client, "h@example.com"))
    room = client.post(
        "/api/lobbies",
        json={"mode": "online", "n_players": 4, "listed": True},
        headers=acct,
    ).json()
    (row,) = client.get("/api/lobbies").json()
    assert row["id"] == room["id"] and row["open_seats"] == 3
