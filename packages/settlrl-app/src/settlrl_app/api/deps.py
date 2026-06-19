"""Shared request helpers and the runtime context the routers close over.

``create_app`` builds one :class:`Deps` and hands it to each router factory in
:mod:`settlrl_app.api.routers`; the routers read everything game-related through
it (the registry, bot providers, the auth dependencies, the replay slot, and the
driver spawner), so they hold no module-level state and tests stay isolated.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Header, HTTPException
from settlrl_game.session import HUMAN

from settlrl_app.bots.providers import ProviderRegistry
from settlrl_app.game.games import GameHandle, GameRegistry
from settlrl_app.game.replay import ReplaySession
from settlrl_app.storage.auth import Auth
from settlrl_app.storage.db import User
from settlrl_app.storage.store import GameStore

if TYPE_CHECKING:
    from settlrl_app.game.matchmaking import Matchmaker

# The per-request seat-ownership proof: a comma-separated list of seat tokens.
SeatTokens = Annotated[str | None, Header(alias="X-Seat-Tokens")]


def tokens(header: str | None) -> list[str]:
    return [t.strip() for t in (header or "").split(",") if t.strip()]


def uid(user: User | None) -> str | None:
    return str(user.id) if user is not None else None


def needs_driver(handle: GameHandle, turn_timeout: float) -> bool:
    """A non-terminal game needs the server-side driver when it has a bot seat
    to pace, or a turn timeout to enforce on human seats (started on create,
    restarted for restored games)."""
    session = handle.session
    if session.terminal():
        return False
    return turn_timeout > 0 or any(k != HUMAN for k in session.seats)


class ReplaySlot:
    """The loaded replay, if any (server-wide tooling; one at a time)."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.session: ReplaySession | None = None


@dataclass
class Deps:
    """The shared runtime the routers operate on."""

    registry: GameRegistry
    bots: ProviderRegistry
    auth: Auth
    replays: ReplaySlot
    spawn_driver: Callable[[GameHandle], None]
    turn_timeout: float
    store: GameStore | None = None
    matchmaker: Matchmaker | None = None

    def handle_of(self, game_id: str) -> GameHandle:
        handle = self.registry.get(game_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="no such game")
        return handle
