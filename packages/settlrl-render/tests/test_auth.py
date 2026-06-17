"""Tests for the accounts / authentication layer.

Cover the unit store (hashing, sessions, admin) and the routes through their
own app, plus that the dependencies gate as expected and the anonymous flow is
untouched. Each app gets a fresh in-memory user db (``create_app`` with no
state dir), so nothing leaks between tests.
"""

from collections.abc import Iterator

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from settlrl_render.auth import (
    Auth,
    AuthError,
    UserStore,
    hash_password,
    verify_password,
)
from settlrl_render.games import GameRegistry
from settlrl_render.server import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    yield TestClient(
        create_app(GameRegistry(), admin_emails=frozenset({"boss@example.com"}))
    )


def _register(client: TestClient, email: str, password: str = "hunter2pw") -> None:
    assert (
        client.post(
            "/api/auth/register", json={"email": email, "password": password}
        ).status_code
        == 201
    )


def _login(client: TestClient, email: str, password: str = "hunter2pw") -> str:
    resp = client.post(
        "/api/auth/login", data={"username": email, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["access_token"])


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# -- password hashing ---------------------------------------------------------


def test_hash_roundtrips_and_salts() -> None:
    h = hash_password("correct horse")
    assert verify_password("correct horse", h)
    assert not verify_password("wrong", h)
    # Distinct salts -> distinct digests for the same password.
    assert h != hash_password("correct horse")


def test_verify_rejects_garbage_hash() -> None:
    assert not verify_password("x", "not-a-hash")
    assert not verify_password("x", "bcrypt$1$2$3$4$5")


# -- store --------------------------------------------------------------------


def test_store_sessions_and_admin() -> None:
    store = UserStore(None)
    user = store.create_user("A@Example.com", "password1")
    assert user.email == "a@example.com" and not user.is_admin
    assert store.authenticate("a@example.com", "password1") == user
    assert store.authenticate("a@example.com", "nope") is None

    token = store.mint_session(user.id, ttl=100.0)
    assert store.user_for_token(token) == user
    store.revoke(token)
    assert store.user_for_token(token) is None

    # Expired sessions don't authenticate.
    expired = store.mint_session(user.id, ttl=-1.0)
    assert store.user_for_token(expired) is None

    promoted = store.set_admin("a@example.com")
    assert promoted is not None and promoted.is_admin
    assert store.set_admin("ghost@example.com") is None


def test_store_rejects_duplicate_email() -> None:
    store = UserStore(None)
    store.create_user("a@example.com", "password1")
    with pytest.raises(AuthError):
        store.create_user("A@example.com", "password2")


# -- routes -------------------------------------------------------------------


def test_register_login_me_logout(client: TestClient) -> None:
    _register(client, "alice@example.com")
    token = _login(client, "alice@example.com")
    me = client.get("/api/auth/me", headers=_bearer(token))
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com" and not me.json()["is_admin"]

    assert client.post("/api/auth/logout", headers=_bearer(token)).status_code == 204
    assert client.get("/api/auth/me", headers=_bearer(token)).status_code == 401


def test_me_requires_a_valid_token(client: TestClient) -> None:
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers=_bearer("bogus")).status_code == 401


def test_duplicate_registration_conflicts(client: TestClient) -> None:
    _register(client, "dup@example.com")
    resp = client.post(
        "/api/auth/register", json={"email": "DUP@example.com", "password": "hunter2pw"}
    )
    assert resp.status_code == 409


def test_short_password_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/auth/register", json={"email": "x@example.com", "password": "short"}
    )
    assert resp.status_code == 422


def test_bad_login_is_401(client: TestClient) -> None:
    _register(client, "bob@example.com")
    assert (
        client.post(
            "/api/auth/login", data={"username": "bob@example.com", "password": "WRONG"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login", data={"username": "ghost@example.com", "password": "x"}
        ).status_code
        == 401
    )


def test_configured_admin_email_is_promoted(client: TestClient) -> None:
    _register(client, "boss@example.com", "bosspass1")
    token = _login(client, "boss@example.com", "bosspass1")
    assert client.get("/api/auth/me", headers=_bearer(token)).json()["is_admin"]


def test_admin_dependency_gates_a_route() -> None:
    """The admin_user dependency 401s anonymous, 403s non-admins, admits admins."""
    auth = Auth(UserStore(None), admin_emails=frozenset({"boss@example.com"}))
    app = FastAPI()
    app.include_router(auth.router)

    @app.get("/admin-only")
    def _admin_only(user: object = Depends(auth.admin_user)) -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/admin-only").status_code == 401

    _register(client, "user@example.com")
    assert (
        client.get(
            "/admin-only", headers=_bearer(_login(client, "user@example.com"))
        ).status_code
        == 403
    )

    _register(client, "boss@example.com", "bosspass1")
    assert (
        client.get(
            "/admin-only",
            headers=_bearer(_login(client, "boss@example.com", "bosspass1")),
        ).status_code
        == 200
    )
