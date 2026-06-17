"""Shared test helpers.

The game server runs no bots itself, so any test that seats or plays a bot needs
an in-process bot service registered with it. ``bot_registry`` wires one up
(white-box: the provider is stuffed in directly, so no admin/HTTP/event-loop
dance is needed at setup time), and ``BOT_KINDS`` is what such seats validate
against when a ``GameSession`` is built directly.
"""

import httpx
from fastapi import FastAPI
from settlrl_render.bots.bot_service import create_bot_app
from settlrl_render.bots.bots import bot_catalog
from settlrl_render.bots.providers import ProviderRegistry, RemoteBotProvider

# The bot kinds the in-process service offers (used as ``external_kinds`` when a
# test constructs a GameSession with bot seats directly).
BOT_KINDS = frozenset(bot_catalog())


def asgi_client(app: FastAPI) -> httpx.AsyncClient:
    """An async client dispatching in-process to ``app`` (no sockets)."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://svc"
    )


def bot_registry(app: FastAPI | None = None) -> ProviderRegistry:
    """A provider registry with an in-process bot service pre-registered, so a
    game server built with it can seat and play bot kinds (random, greedy, …)."""
    app = app or create_bot_app()
    client = asgi_client(app)
    reg = ProviderRegistry(client=client)
    reg._remotes["svc"] = RemoteBotProvider("svc", "http://svc", bot_catalog(), client)
    return reg
