"""The admin status page's data endpoint (superuser only)."""

from collections.abc import Iterator

import pytest
from _helpers import bot_registry
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(
        create_app(
            GameRegistry(),
            providers=bot_registry(),
            admin_emails=frozenset({"admin@example.com"}),
        )
    ) as c:
        yield c


def _login(client: TestClient, email: str) -> str:
    client.post("/api/auth/register", json={"email": email, "password": "password1"})
    return str(
        client.post(
            "/api/auth/login", data={"username": email, "password": "password1"}
        ).json()["access_token"]
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_status_requires_a_superuser(client: TestClient) -> None:
    assert client.get("/api/admin/status").status_code == 401
    user = _bearer(_login(client, "user@example.com"))
    assert client.get("/api/admin/status", headers=user).status_code == 403


def test_status_reports_server_and_games(client: TestClient) -> None:
    admin = _bearer(_login(client, "admin@example.com"))
    client.post(
        "/api/games",
        json={
            "seed": 0,
            "n_players": 2,
            "seats": ["human", "human"],
            "claim": "first",
            "listed": True,
            "searchable": True,
        },
        headers=admin,
    )

    st = client.get("/api/admin/status", headers=admin).json()
    assert st["uptime_seconds"] >= 0
    assert st["games_active"] == 1 and st["games_total"] == 1
    assert st["games_capacity"] >= 1
    (game,) = st["games"]
    assert game["n_players"] == 2
    assert game["open_seats"] == 1  # the creator claimed seat 0 only
    assert game["listed"] and game["searchable"] and not game["terminal"]
