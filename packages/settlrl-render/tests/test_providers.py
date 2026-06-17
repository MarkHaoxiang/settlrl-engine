"""Tests for the bot-provider layer: the registry, the admin API that mutates
it, and end-to-end remote dispatch through the driver.

In-process remote services are reached with a FastAPI ``TestClient`` as the
registry's HTTP client (a sync client bound to an ASGI app), so no sockets are
opened.
"""

import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from settlrl_render.bot_service import create_bot_app
from settlrl_render.games import GameRegistry
from settlrl_render.providers import ActRequest, ProviderRegistry, RemoteBotError
from settlrl_render.server import create_app


def _catalog_only_app(kind: str, counts: tuple[int, ...] = (2, 3, 4)) -> FastAPI:
    """A minimal remote service that only advertises a kind (no real play)."""
    app = FastAPI()

    @app.get("/catalog")
    def catalog() -> dict[str, dict[str, object]]:
        return {kind: {"counts": list(counts), "description": "stub", "params": {}}}

    return app


def _erroring_act_app(kind: str) -> FastAPI:
    """A remote service that advertises a kind but fails every move request."""
    app = _catalog_only_app(kind)

    @app.post("/act")
    def act(_: ActRequest) -> dict[str, int]:
        raise HTTPException(status_code=500, detail="boom")

    return app


# -- registry -----------------------------------------------------------------


def test_register_merges_catalog_and_routes_kind() -> None:
    reg = ProviderRegistry(client=TestClient(_catalog_only_app("alphacatan")))
    reg.register("svc", "http://svc")
    catalog = reg.catalog()
    assert "alphacatan" in catalog  # remote kind joined
    assert "random" in catalog  # local kinds still there
    assert reg.remote_for("alphacatan") is not None
    assert reg.remote_for("random") is None  # local
    assert reg.providers()[0]["name"] == "svc"


def test_register_rejects_kind_clash_with_local() -> None:
    reg = ProviderRegistry(client=TestClient(_catalog_only_app("random")))
    with pytest.raises(RemoteBotError, match="already provided"):
        reg.register("svc", "http://svc")


def test_register_unreachable_service_errors() -> None:
    # A client pointed at an app with no /catalog route -> 404 -> RemoteBotError.
    reg = ProviderRegistry(client=TestClient(FastAPI()))
    with pytest.raises(RemoteBotError):
        reg.register("svc", "http://svc")


def test_unregister() -> None:
    reg = ProviderRegistry(client=TestClient(_catalog_only_app("alphacatan")))
    reg.register("svc", "http://svc")
    assert reg.unregister("svc")
    assert "alphacatan" not in reg.catalog()
    assert not reg.unregister("svc")


def test_local_bots_off_offers_only_remote_kinds() -> None:
    reg = ProviderRegistry(
        client=TestClient(_catalog_only_app("alphacatan")), local_bots=False
    )
    assert reg.catalog() == {}  # nothing until a provider registers
    reg.register("svc", "http://svc")
    assert set(reg.catalog()) == {"alphacatan"}  # no built-ins leaked


# -- admin API ----------------------------------------------------------------


def _app_with_admin() -> tuple[TestClient, str]:
    """A game-server client plus a bearer token for an admin user."""
    reg = ProviderRegistry(client=TestClient(_catalog_only_app("alphacatan")))
    client = TestClient(
        create_app(GameRegistry(), providers=reg, admin_emails=frozenset({"a@x.com"}))
    )
    client.post(
        "/api/auth/register", json={"email": "a@x.com", "password": "password1"}
    )
    token = client.post(
        "/api/auth/login", data={"username": "a@x.com", "password": "password1"}
    ).json()["access_token"]
    return client, token


def test_admin_register_provider_requires_admin() -> None:
    client, token = _app_with_admin()
    body = {"name": "svc", "base_url": "http://svc"}
    # Anonymous and non-admin are refused.
    assert client.post("/api/admin/bot-providers", json=body).status_code == 401
    client.post(
        "/api/auth/register", json={"email": "u@x.com", "password": "password1"}
    )
    utok = client.post(
        "/api/auth/login", data={"username": "u@x.com", "password": "password1"}
    ).json()["access_token"]
    assert (
        client.post(
            "/api/admin/bot-providers",
            json=body,
            headers={"Authorization": f"Bearer {utok}"},
        ).status_code
        == 403
    )
    # The admin can register; the kind then shows in the public catalog.
    h = {"Authorization": f"Bearer {token}"}
    assert (
        client.post("/api/admin/bot-providers", json=body, headers=h).status_code == 201
    )
    assert "alphacatan" in client.get("/api/bots").json()
    assert client.get("/api/admin/bot-providers", headers=h).json()[0]["name"] == "svc"
    assert client.delete("/api/admin/bot-providers/svc", headers=h).status_code == 204
    assert client.delete("/api/admin/bot-providers/svc", headers=h).status_code == 404
    assert "alphacatan" not in client.get("/api/bots").json()


# -- end-to-end dispatch ------------------------------------------------------


def _drive_to_terminal(
    client: TestClient, body: dict[str, object]
) -> dict[str, Any]:
    gid = client.post("/api/games", json=body).json()["id"]
    snap: dict[str, Any] = {}
    for _ in range(200):
        snap = client.get(f"/api/games/{gid}").json()
        if snap["status"]["terminal"]:
            break
        time.sleep(0.05)
    return snap


@pytest.fixture()
def remote_only_client() -> Iterator[TestClient]:
    """A game server with no local agent execution; all bots run in a separate
    in-process bot service."""
    reg = ProviderRegistry(client=TestClient(create_bot_app()), local_bots=False)
    reg.register("svc", "http://svc")
    yield TestClient(
        create_app(GameRegistry(), providers=reg, bot_delay=0.0, warm=False)
    )


def test_remote_bots_play_a_game_to_completion(remote_only_client: TestClient) -> None:
    snap = _drive_to_terminal(
        remote_only_client,
        {"seed": 1, "n_players": 2, "seats": ["random", "random"], "claim": "none"},
    )
    assert snap["status"]["terminal"] and snap["status"]["winner"] in (0, 1)


def test_create_rejects_unknown_remote_kind(remote_only_client: TestClient) -> None:
    # With local bots off and only "svc" registered, a built-in-only kind that
    # the service doesn't advertise is unknown -> 422 (and 'human' stays valid).
    resp = remote_only_client.post(
        "/api/games", json={"seed": 1, "n_players": 2, "seats": ["nope", "random"]}
    )
    assert resp.status_code == 422


def test_remote_failure_falls_back_and_game_progresses() -> None:
    """A remote service that errors on every move must not stall the game: the
    driver falls back to a local random move."""
    reg = ProviderRegistry(
        client=TestClient(_erroring_act_app("flaky")), local_bots=False
    )
    reg.register("svc", "http://svc")
    client = TestClient(
        create_app(GameRegistry(), providers=reg, bot_delay=0.0, warm=False)
    )
    snap = _drive_to_terminal(
        client,
        {"seed": 2, "n_players": 2, "seats": ["flaky", "flaky"], "claim": "none"},
    )
    assert snap["status"]["terminal"]
