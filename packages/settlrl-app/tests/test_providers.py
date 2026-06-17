"""Tests for the bot-provider layer: the registry, the admin API that mutates
it, and end-to-end remote dispatch through the driver.

In-process remote services are reached with an ``httpx.AsyncClient`` over an
``ASGITransport`` bound to a bot-service app, so no sockets are opened. The
registry's ``register`` is async; the unit tests drive it under ``asyncio.run``,
while the end-to-end tests register through the admin route so everything runs
on the app's own event loop.
"""

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from settlrl_agents.service.app import create_bot_app
from settlrl_app.bots.providers import ActRequest, ProviderRegistry, RemoteBotError
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app


def _asgi_client(app: FastAPI) -> httpx.AsyncClient:
    """An async client that dispatches in-process to ``app`` (no sockets)."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://svc"
    )


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
    async def go() -> None:
        reg = ProviderRegistry(client=_asgi_client(_catalog_only_app("alphacatan")))
        await reg.register("svc", "http://svc")
        assert "alphacatan" in reg.catalog()  # the remote kind joined
        assert reg.remote_for("alphacatan") is not None
        assert reg.remote_for("nope") is None  # unknown kind
        assert reg.providers()[0]["name"] == "svc"
        await reg.aclose()

    asyncio.run(go())


def test_register_rejects_kind_clash_between_providers() -> None:
    async def go() -> None:
        reg = ProviderRegistry(client=_asgi_client(_catalog_only_app("dup")))
        await reg.register("a", "http://svc")
        with pytest.raises(RemoteBotError, match="already provided"):
            await reg.register("b", "http://svc")  # same kind from another name
        await reg.aclose()

    asyncio.run(go())


def test_register_unreachable_service_errors() -> None:
    async def go() -> None:
        # A client pointed at an app with no /catalog route -> 404 -> error.
        reg = ProviderRegistry(client=_asgi_client(FastAPI()))
        with pytest.raises(RemoteBotError):
            await reg.register("svc", "http://svc")
        await reg.aclose()

    asyncio.run(go())


def test_unregister() -> None:
    async def go() -> None:
        reg = ProviderRegistry(client=_asgi_client(_catalog_only_app("alphacatan")))
        await reg.register("svc", "http://svc")
        assert reg.unregister("svc")
        assert "alphacatan" not in reg.catalog()
        assert not reg.unregister("svc")
        await reg.aclose()

    asyncio.run(go())


def test_offers_only_remote_kinds() -> None:
    async def go() -> None:
        reg = ProviderRegistry(client=_asgi_client(_catalog_only_app("alphacatan")))
        assert reg.catalog() == {}  # nothing until a provider registers
        await reg.register("svc", "http://svc")
        assert set(reg.catalog()) == {"alphacatan"}  # no built-ins
        await reg.aclose()

    asyncio.run(go())


# -- admin API ----------------------------------------------------------------


def _login(client: TestClient, email: str) -> str:
    client.post("/api/auth/register", json={"email": email, "password": "password1"})
    return str(
        client.post(
            "/api/auth/login", data={"username": email, "password": "password1"}
        ).json()["access_token"]
    )


def _register_remote(client: TestClient, name: str = "svc") -> None:
    """Register the remote service via the admin route (on the app's loop)."""
    tok = _login(client, "a@x.com")
    resp = client.post(
        "/api/admin/bot-providers",
        json={"name": name, "base_url": "http://svc"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert resp.status_code == 201, resp.text


def test_admin_register_provider_requires_admin() -> None:
    reg = ProviderRegistry(client=_asgi_client(_catalog_only_app("alphacatan")))
    with TestClient(
        create_app(
            GameRegistry(),
            providers=reg,
            admin_emails=frozenset({"a@x.com"}),
        )
    ) as client:
        body = {"name": "svc", "base_url": "http://svc"}
        # Anonymous and non-admin are refused.
        assert client.post("/api/admin/bot-providers", json=body).status_code == 401
        utok = _login(client, "u@x.com")
        assert (
            client.post(
                "/api/admin/bot-providers",
                json=body,
                headers={"Authorization": f"Bearer {utok}"},
            ).status_code
            == 403
        )
        # The admin can register; the kind then shows in the public catalog.
        h = {"Authorization": f"Bearer {_login(client, 'a@x.com')}"}
        assert (
            client.post("/api/admin/bot-providers", json=body, headers=h).status_code
            == 201
        )
        assert "alphacatan" in client.get("/api/bots").json()
        assert (
            client.get("/api/admin/bot-providers", headers=h).json()[0]["name"] == "svc"
        )
        assert (
            client.delete("/api/admin/bot-providers/svc", headers=h).status_code == 204
        )
        assert (
            client.delete("/api/admin/bot-providers/svc", headers=h).status_code == 404
        )
        assert "alphacatan" not in client.get("/api/bots").json()
    asyncio.run(reg.aclose())


# -- end-to-end dispatch ------------------------------------------------------


def _drive_to_terminal(client: TestClient, body: dict[str, object]) -> dict[str, Any]:
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
    in-process bot service, registered via the admin route once the loop is up."""
    reg = ProviderRegistry(client=_asgi_client(create_bot_app()))
    with TestClient(
        create_app(
            GameRegistry(),
            providers=reg,
            bot_delay=0.0,
            admin_emails=frozenset({"a@x.com"}),
        )
    ) as client:
        _register_remote(client)
        yield client
    asyncio.run(reg.aclose())


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
    driver falls back to a local random move, so moves keep coming."""
    reg = ProviderRegistry(client=_asgi_client(_erroring_act_app("flaky")))
    with TestClient(
        create_app(
            GameRegistry(),
            providers=reg,
            bot_delay=0.0,
            admin_emails=frozenset({"a@x.com"}),
        )
    ) as client:
        _register_remote(client)
        gid = client.post(
            "/api/games",
            json={
                "seed": 2,
                "n_players": 2,
                "seats": ["flaky", "flaky"],
                "claim": "none",
            },
        ).json()["id"]
        # A failing remote round-trip per move is slow, so assert clear progress
        # (well past the setup phase) rather than full completion — the point is
        # that the game is advancing via the fallback, not stalled.
        snap: dict[str, Any] = {}
        for _ in range(200):
            snap = client.get(f"/api/games/{gid}").json()
            if snap["status"]["terminal"] or len(snap["log"]) > 12:
                break
            time.sleep(0.05)
        assert snap["status"]["terminal"] or len(snap["log"]) > 12
    asyncio.run(reg.aclose())
