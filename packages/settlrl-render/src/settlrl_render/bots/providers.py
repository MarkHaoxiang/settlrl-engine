"""Bot providers: where a seat's moves are computed.

The built-in provider runs the ``settlrl-agents`` policies in-process, exactly
as the server always has. A **remote** provider is a separate bot service
(:mod:`settlrl_render.bots.bot_service`) reached over HTTP, so the agent-running code
can be deployed and scaled apart from the game server; an admin registers one at
runtime and its bot kinds join the catalog.

The wire contract is deliberately small and engine-version-stable: a move is
requested by sending the game's setup plus its flat move list so far (the same
data a ``settlrl_engine.record`` carries) and the service replays them and
returns the chosen flat action — no engine observation pytree crosses the wire,
so the two sides only have to agree on the (stable) record format and the flat
action indexing.

:class:`ProviderRegistry` maps each bot kind to where it runs; local kinds stay
in-process (the driver's fast path on the live env), remote kinds dispatch to
their service over an async HTTP client. It is mutated and read only on the
event loop (admin routes, the driver), so it needs no lock.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from settlrl_render.bots.bots import bot_catalog

# A bot move request / reply on the standardized wire (the bot service's /act).
# `game_id` only keys the service's replay cache; `setup` + `moves` fully
# determine the position.


class ActRequest(BaseModel):
    game_id: str
    setup: dict[str, Any]
    moves: list[int]
    seat: int


class ActResponse(BaseModel):
    flat: int


class RemoteBotError(Exception):
    """A remote bot service was unreachable or answered unusably."""


# HTTP timeout (seconds) for the catalog fetch and each move request, applied as
# the client's default. A move request is awaited on the driver's turn, so it is
# kept short — a slow service falls back to a local random move rather than
# stalling the game.
_DEFAULT_TIMEOUT = 5.0


class RemoteBotProvider:
    """A registered remote bot service and the kinds it offers."""

    def __init__(
        self,
        name: str,
        base_url: str,
        catalog: dict[str, dict[str, object]],
        client: httpx.AsyncClient,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._catalog = catalog
        self._client = client

    @classmethod
    async def connect(
        cls, name: str, base_url: str, client: httpx.AsyncClient
    ) -> RemoteBotProvider:
        """Fetch a service's catalog to register it (raising
        :class:`RemoteBotError` if it can't be reached or speaks nonsense)."""
        url = base_url.rstrip("/") + "/catalog"
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            catalog = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RemoteBotError(
                f"cannot reach bot service at {base_url}: {exc}"
            ) from exc
        if not isinstance(catalog, dict) or not all(
            isinstance(k, str) and isinstance(v, dict) for k, v in catalog.items()
        ):
            raise RemoteBotError(f"bot service at {base_url} returned a bad catalog")
        return cls(name, base_url, catalog, client)

    @property
    def kinds(self) -> set[str]:
        return set(self._catalog)

    def catalog(self) -> dict[str, dict[str, object]]:
        return dict(self._catalog)

    async def act(
        self, game_id: str, setup: dict[str, Any], moves: list[int], seat: int
    ) -> int:
        """The flat move the service picks for ``seat`` (raises
        :class:`RemoteBotError` on any transport / protocol failure)."""
        req = ActRequest(game_id=game_id, setup=setup, moves=moves, seat=seat)
        try:
            resp = await self._client.post(
                self.base_url + "/act", json=req.model_dump()
            )
            resp.raise_for_status()
            return ActResponse(**resp.json()).flat
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise RemoteBotError(f"bot service {self.name!r} failed: {exc}") from exc


class ProviderRegistry:
    """Bot kinds -> where they run. Built-in kinds (``settlrl-agents``) run
    locally; admins register remote services whose kinds join the catalog.

    Set ``local_bots=False`` to run the game server with **no** in-process agent
    execution: it then offers only registered remote providers' kinds, so the
    agent-running code lives entirely in the bot service(s). (The abandoned-turn
    auto-play still uses a trivial local random move as a liveness fallback.)

    ``client`` is the shared async HTTP client for remote calls (tests inject one
    wired to a bot-service app via ``ASGITransport``)."""

    def __init__(
        self, client: httpx.AsyncClient | None = None, local_bots: bool = True
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        self._local_bots = local_bots
        self._remotes: dict[str, RemoteBotProvider] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    def _local_catalog(self) -> dict[str, dict[str, object]]:
        return dict(bot_catalog()) if self._local_bots else {}

    def catalog(self) -> dict[str, dict[str, object]]:
        """Every offered bot kind — local first, then each remote provider's —
        in the shape ``GET /api/bots`` returns."""
        out = self._local_catalog()
        for prov in self._remotes.values():
            out.update(prov.catalog())
        return out

    def remote_for(self, kind: str) -> RemoteBotProvider | None:
        """The remote provider that owns ``kind``, or None when it is local (or
        unknown — the create route rejects unknown kinds first)."""
        for prov in self._remotes.values():
            if kind in prov.kinds:
                return prov
        return None

    def remote_kinds(self) -> frozenset[str]:
        """All kinds served remotely (the session's ``external_kinds``)."""
        return frozenset(k for prov in self._remotes.values() for k in prov.kinds)

    async def register(self, name: str, base_url: str) -> RemoteBotProvider:
        """Register (or replace) a remote provider by name. Raises
        :class:`RemoteBotError` if it is unreachable or any of its kinds clash
        with a local or another remote provider's kind."""
        provider = await RemoteBotProvider.connect(name, base_url, self._client)
        local = set(self._local_catalog())
        others = {k for n, p in self._remotes.items() if n != name for k in p.kinds}
        clash = provider.kinds & (local | others)
        if clash:
            raise RemoteBotError(
                f"bot kind(s) already provided: {', '.join(sorted(clash))}"
            )
        self._remotes[name] = provider
        return provider

    def unregister(self, name: str) -> bool:
        return self._remotes.pop(name, None) is not None

    def providers(self) -> list[dict[str, object]]:
        """Registered remote providers (name, base url, kinds) for the admin API."""
        return [
            {"name": n, "base_url": p.base_url, "kinds": sorted(p.kinds)}
            for n, p in self._remotes.items()
        ]
