"""Shared request helpers and the runtime context the routers close over.

``create_app`` builds one :class:`Deps` and hands it to each router factory in
:mod:`settlrl_app.api.routers`; the routers read everything game-related through
it (the registry, bot providers, the auth dependencies, the replay slot, and the
driver spawner), so they hold no module-level state and tests stay isolated.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from fastapi import Header, HTTPException
from settlrl_game.session import HUMAN

from settlrl_app.bots.providers import ProviderRegistry
from settlrl_app.game.games import GameHandle, GameRegistry
from settlrl_app.game.lobbies import LobbyRegistry
from settlrl_app.game.replay import ReplaySession
from settlrl_app.storage.auth import Auth
from settlrl_app.storage.db import User
from settlrl_app.storage.store import GameStore

if TYPE_CHECKING:
    from settlrl_app.game.matchmaking import Matchmaker

# The per-request seat-ownership proof: a comma-separated list of seat tokens.
SeatTokens = Annotated[str | None, Header(alias="X-Seat-Tokens")]

# A stable anonymous per-browser id, so guests (with no account) are held to one
# game at a time and never matched against themselves. Not an ownership proof —
# only an identity hint for the one-game guard and matchmaker self-pair dedup.
ClientId = Annotated[str | None, Header(alias="X-Client-Id")]


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
    lobbies: LobbyRegistry = field(default_factory=LobbyRegistry)
    store: GameStore | None = None
    matchmaker: Matchmaker | None = None
    # Wall-clock process start, for the admin status page's uptime.
    started_at: float = field(default_factory=time.time)

    def handle_of(self, game_id: str) -> GameHandle:
        handle = self.registry.get(game_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="no such game")
        return handle

    def guard_one_game(
        self,
        user: User | None,
        client_id: str | None = None,
        allow: str | None = None,
    ) -> None:
        """Enforce one live game per player: raise 409 (carrying the existing
        game's id, so the client can offer to resume it) when the caller already
        holds a seat in a live game other than ``allow``. A signed-in ``user`` is
        keyed by account; a guest by ``client_id`` (its browser). A guest with no
        client id sent (an old client) passes — the limit is only as strong as
        its identity."""
        if user is not None:
            key = str(user.id)
            game = self.registry.live_game_for_user(key)
            lobby = self.lobbies.live_for_user(key)
        elif client_id:
            game = self.registry.live_game_for_client(client_id)
            lobby = self.lobbies.live_for_client(client_id)
        else:
            return
        for kind, existing in (("game", game), ("lobby", lobby)):
            if existing is not None and existing.id != allow:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "you are already in a game",
                        "id": existing.id,
                        "kind": kind,
                    },
                )
