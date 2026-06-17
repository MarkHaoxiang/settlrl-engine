"""Seats follow the account that claimed them.

A signed-in player owns their seats by user id, not just by the per-device seat
token — so they are recognised on any device (and can list their games) without
carrying the token around. These drive that through the routes.
"""

import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from settlrl_render.games import GameRegistry
from settlrl_render.server import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    yield TestClient(create_app(GameRegistry(), warm=False))


def _token(client: TestClient, email: str, password: str = "password1") -> str:
    client.post("/api/auth/register", json={"email": email, "password": password})
    return str(
        client.post(
            "/api/auth/login", data={"username": email, "password": password}
        ).json()["access_token"]
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_authenticated_creator_is_recognised_without_the_seat_token(
    client: TestClient,
) -> None:
    token = _token(client, "a@example.com")
    game = client.post(
        "/api/games",
        json={"seed": 0, "seats": ["human", "random", "random", "random"]},
        headers=_bearer(token),
    ).json()["id"]

    # The account owns seat 0 with only the bearer token — no seat token sent.
    snap = client.get(f"/api/games/{game}", headers=_bearer(token)).json()
    assert snap["your_seats"] == [0]
    assert snap["status"]["your_turn"] and snap["actions"]

    # And can act on that seat with just the account token.
    flat = snap["actions"][0]["flat"]
    assert (
        client.post(
            f"/api/games/{game}/action", json={"flat": flat}, headers=_bearer(token)
        ).status_code
        == 200
    )


def test_other_users_and_anonymous_own_nothing(client: TestClient) -> None:
    owner = _token(client, "owner@example.com")
    other = _token(client, "other@example.com")
    game = client.post(
        "/api/games",
        json={"seed": 0, "seats": ["human", "random", "random", "random"]},
        headers=_bearer(owner),
    ).json()["id"]

    assert (
        client.get(f"/api/games/{game}", headers=_bearer(other)).json()["your_seats"]
        == []
    )
    assert client.get(f"/api/games/{game}").json()["your_seats"] == []


def test_my_games_lists_only_the_users_games(client: TestClient) -> None:
    a = _token(client, "a@example.com")
    b = _token(client, "b@example.com")
    game_a = client.post(
        "/api/games",
        json={"seed": 0, "seats": ["human", "random", "random", "random"]},
        headers=_bearer(a),
    ).json()["id"]
    client.post(
        "/api/games",
        json={"seed": 1, "seats": ["human", "random", "random", "random"]},
        headers=_bearer(b),
    )

    mine = client.get("/api/me/games", headers=_bearer(a)).json()
    assert [g["id"] for g in mine] == [game_a]
    assert mine[0]["seats"] == [0]
    assert client.get("/api/me/games").status_code == 401  # requires sign-in


def test_join_ties_the_seat_to_the_account(client: TestClient) -> None:
    a = _token(client, "a@example.com")
    b = _token(client, "b@example.com")
    # Two human seats, none claimed at create.
    game = client.post(
        "/api/games",
        json={
            "seed": 0,
            "seats": ["human", "human", "random", "random"],
            "claim": "none",
        },
        headers=_bearer(a),
    ).json()["id"]

    seat = client.post(f"/api/games/{game}/join", json={}, headers=_bearer(b)).json()[
        "seat"
    ]
    snap = client.get(f"/api/games/{game}", headers=_bearer(b)).json()
    assert snap["your_seats"] == [seat]


def test_account_seat_ownership_survives_a_restart() -> None:
    """The seat<->account tie is journalled, so a restored game still knows it."""
    with tempfile.TemporaryDirectory() as state_dir:
        first = TestClient(create_app(state_dir=state_dir, warm=False))
        token = _token(first, "a@example.com")
        game = first.post(
            "/api/games",
            json={"seed": 0, "seats": ["human", "random", "random", "random"]},
            headers=_bearer(token),
        ).json()["id"]

        # A fresh app restores games + accounts from the same state dir.
        restored = TestClient(create_app(state_dir=state_dir, warm=False))
        token2 = str(
            restored.post(
                "/api/auth/login",
                data={"username": "a@example.com", "password": "password1"},
            ).json()["access_token"]
        )
        snap = restored.get(f"/api/games/{game}", headers=_bearer(token2)).json()
        assert snap["your_seats"] == [0]
