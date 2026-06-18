"""Bot providers: the remote bot services that compute a seat's moves.

The game server runs no bot policies itself. Each registered provider is a
separate **one-bot** service (:mod:`settlrl_agents.service`) reached over HTTP, so
the agent-running code is deployed and scaled apart from the game server; an admin
registers one by base URL and its bot's ``name`` becomes a seatable kind.

The wire is structured and incremental (:mod:`settlrl_game.botproto`): a move is
requested by sending only the moves the service has not seen yet — those after the
``base`` cursor this provider keeps per game — as :class:`MoveModel`s in board
coordinates; the service applies them to the game it tracks and returns the chosen
move. If the service has fallen behind/ahead (a restart), it answers ``409`` with
the move count it actually holds and the request is replayed from there.

:class:`ProviderRegistry` maps each bot kind (the service's bot name) to its
provider. It is mutated and read only on the event loop, so it needs no lock.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import httpx
from settlrl_game.actions import flat_for_move, move_for_flat
from settlrl_game.botproto import ActRequest, ActResponse, BotInfo

__all__ = [
    "ProviderRegistry",
    "RemoteBotError",
    "RemoteBotProvider",
]


class RemoteBotError(Exception):
    """A remote bot service was unreachable or answered unusably."""


# HTTP timeout (seconds) for the info fetch and each move request, applied as the
# client's default. A move request is awaited on the driver's turn, so it is kept
# short — a slow service falls back to a local random move rather than stalling.
_DEFAULT_TIMEOUT = 5.0

# Per-provider cap on tracked per-game cursors; past it the oldest are dropped (a
# dropped cursor just costs one resync round-trip on that game's next move).
_CURSOR_CAP = 256


class RemoteBotProvider:
    """A registered remote bot service and the single bot it offers."""

    def __init__(self, base_url: str, info: BotInfo, client: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.info = info
        self._client = client
        # game_id -> move count the service is assumed to already hold.
        self._sent: OrderedDict[str, int] = OrderedDict()

    @classmethod
    async def connect(
        cls, base_url: str, client: httpx.AsyncClient
    ) -> RemoteBotProvider:
        """Fetch a service's :class:`BotInfo` to register it (raising
        :class:`RemoteBotError` if it can't be reached or speaks nonsense)."""
        url = base_url.rstrip("/") + "/info"
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            info = BotInfo(**resp.json())
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise RemoteBotError(
                f"cannot reach bot service at {base_url}: {exc}"
            ) from exc
        if not info.name:
            raise RemoteBotError(f"bot service at {base_url} reported no name")
        return cls(base_url, info, client)

    @property
    def name(self) -> str:
        return self.info.name

    def catalog(self) -> dict[str, dict[str, object]]:
        return {
            self.name: {
                "title": self.info.title,
                "description": self.info.description,
                "counts": self.info.counts,
            }
        }

    async def act(
        self, game_id: str, setup: dict[str, Any], history: list[int], seat: int
    ) -> int:
        """The flat move the service picks for ``seat``, sending only the moves
        after this provider's cursor (replaying from the service's reported count
        on a ``409`` resync). Raises :class:`RemoteBotError` on any transport /
        protocol failure."""
        base = self._sent.get(game_id, 0)
        if not 0 <= base <= len(history):
            base = 0
        try:
            for resync in (False, True):
                moves = [move_for_flat(f) for f in history[base:]]
                req = ActRequest(
                    game_id=game_id, seat=seat, setup=setup, base=base, moves=moves
                )
                resp = await self._client.post(
                    self.base_url + "/act", json=req.model_dump()
                )
                if resp.status_code == 409 and not resync:
                    detail = resp.json().get("detail")
                    if isinstance(detail, dict) and detail.get("resync"):
                        base = max(0, min(int(detail.get("have", 0)), len(history)))
                        continue
                resp.raise_for_status()
                flat = flat_for_move(ActResponse(**resp.json()).move)
                self._remember(game_id, len(history))
                return flat
            raise RemoteBotError(f"bot service {self.name!r} kept asking to resync")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise RemoteBotError(f"bot service {self.name!r} failed: {exc}") from exc

    def _remember(self, game_id: str, count: int) -> None:
        self._sent[game_id] = count
        self._sent.move_to_end(game_id)
        while len(self._sent) > _CURSOR_CAP:
            self._sent.popitem(last=False)


class ProviderRegistry:
    """Bot kind (a service's bot name) -> the remote service that serves it. The
    game server runs **no** bots in-process; admins register one-bot services
    whose names form the whole catalog (an unreachable bot still falls back to a
    server-side random move for liveness, but that is not a selectable kind).

    ``client`` is the shared async HTTP client for remote calls (tests inject one
    wired to a bot-service app via ``ASGITransport``)."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        self._remotes: dict[str, RemoteBotProvider] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    def catalog(self) -> dict[str, dict[str, object]]:
        """Every offered bot kind, in the shape ``GET /api/bots`` returns."""
        out: dict[str, dict[str, object]] = {}
        for prov in self._remotes.values():
            out.update(prov.catalog())
        return out

    def remote_for(self, kind: str) -> RemoteBotProvider | None:
        """The provider serving ``kind`` (None if no service offers it)."""
        return self._remotes.get(kind)

    def remote_kinds(self) -> frozenset[str]:
        """All kinds served remotely (the session's ``external_kinds``)."""
        return frozenset(self._remotes)

    async def register(self, base_url: str) -> RemoteBotProvider:
        """Register (or replace) a remote bot service by base URL; its bot name
        becomes the seatable kind. Raises :class:`RemoteBotError` if it is
        unreachable."""
        provider = await RemoteBotProvider.connect(base_url, self._client)
        self._remotes[provider.name] = provider
        return provider

    def unregister(self, name: str) -> bool:
        return self._remotes.pop(name, None) is not None

    def providers(self) -> list[dict[str, object]]:
        """Registered bot services (name, base url) for the admin API."""
        return [{"name": n, "base_url": p.base_url} for n, p in self._remotes.items()]
