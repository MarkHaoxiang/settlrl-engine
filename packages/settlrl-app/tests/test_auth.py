"""Tests for the accounts / authentication layer (fastapi-users).

Each app gets a fresh in-memory async db (``create_app`` with no state dir), so
nothing leaks between tests. The ``with`` form runs the lifespan that creates
the tables.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(
        create_app(GameRegistry(), admin_emails=frozenset({"boss@example.com"}))
    ) as client:
        yield client


def _register(client: TestClient, email: str, password: str = "hunter2pw") -> int:
    return client.post(
        "/api/auth/register", json={"email": email, "password": password}
    ).status_code


def _login(client: TestClient, email: str, password: str = "hunter2pw") -> str:
    resp = client.post(
        "/api/auth/login", data={"username": email, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["access_token"])


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_register_login_me_logout(client: TestClient) -> None:
    assert _register(client, "alice@example.com") == 201
    token = _login(client, "alice@example.com")
    me = client.get("/api/users/me", headers=_bearer(token))
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com" and not me.json()["is_superuser"]

    assert client.post("/api/auth/logout", headers=_bearer(token)).status_code == 204
    assert client.get("/api/users/me", headers=_bearer(token)).status_code == 401


def test_me_requires_a_valid_token(client: TestClient) -> None:
    assert client.get("/api/users/me").status_code == 401
    assert client.get("/api/users/me", headers=_bearer("bogus")).status_code == 401


def test_duplicate_registration_is_rejected(client: TestClient) -> None:
    assert _register(client, "dup@example.com") == 201
    assert _register(client, "dup@example.com") == 400


def test_short_password_rejected(client: TestClient) -> None:
    assert _register(client, "x@example.com", "short") == 400


def test_bad_login_is_rejected(client: TestClient) -> None:
    _register(client, "bob@example.com")
    assert (
        client.post(
            "/api/auth/login",
            data={"username": "bob@example.com", "password": "WRONGpw1"},
        ).status_code
        == 400
    )


def test_configured_admin_email_becomes_superuser(client: TestClient) -> None:
    assert _register(client, "boss@example.com") == 201
    token = _login(client, "boss@example.com")
    assert client.get("/api/users/me", headers=_bearer(token)).json()["is_superuser"]
