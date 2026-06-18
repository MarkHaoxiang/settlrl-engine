"""Shared test helpers.

The game server runs no bots itself, so any test that seats or plays a bot needs
in-process bot services registered with it. Each service hosts ONE bot, so
``bot_app`` mounts a one-bot service per kind under ``/<kind>`` (one ASGI client
then reaches them all), and ``bot_registry`` stuffs a provider per bot straight
into a registry (white-box, no admin/HTTP/event-loop dance at setup time).
``BOT_KINDS`` is what such seats validate against when a ``GameSession`` is built
directly.
"""

from collections.abc import Sequence

import httpx
from fastapi import FastAPI
from settlrl_agents.service.app import create_app as create_bot_app
from settlrl_agents.service.bots import BUNDLED, make_bot
from settlrl_app.bots.providers import ProviderRegistry, RemoteBotProvider

# The bot kinds the in-process services offer (used as ``external_kinds`` when a
# test constructs a GameSession with bot seats directly).
BOT_KINDS = frozenset(BUNDLED)


def asgi_client(app: FastAPI) -> httpx.AsyncClient:
    """An async client dispatching in-process to ``app`` (no sockets)."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://svc"
    )


def bot_app(kinds: Sequence[str] = BUNDLED) -> FastAPI:
    """One ASGI app mounting a one-bot service per kind under ``/<kind>``."""
    root = FastAPI()
    for kind in kinds:
        root.mount(f"/{kind}", create_bot_app(make_bot(kind)))
    return root


def bot_registry(kinds: Sequence[str] = BUNDLED) -> ProviderRegistry:
    """A provider registry with an in-process one-bot service per kind, so a game
    server built with it can seat and play those kinds (random, greedy, …)."""
    client = asgi_client(bot_app(kinds))
    reg = ProviderRegistry(client=client)
    for kind in kinds:
        reg._remotes[kind] = RemoteBotProvider(
            f"http://svc/{kind}", make_bot(kind).info(), client
        )
    return reg
