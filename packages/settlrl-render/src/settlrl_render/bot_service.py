"""The bot service: the agent-running half of the renderer, behind the
standardized bot API (:mod:`settlrl_render.providers`).

It runs the ``settlrl-agents`` policies and answers two requests:

* ``GET /catalog`` — the bot kinds it offers (same shape as the game server's
  ``GET /api/bots``).
* ``POST /act`` — given a game's setup and its flat moves so far, replay them
  and return the move the acting seat's bot would play.

It holds no game registry and no per-seat secrets; it is a pure function of the
game record, so it can be deployed and scaled independently of the game server.
A small per-``game_id`` LRU cache keeps a replayed session around so a move
request only has to apply the new tail rather than the whole game each time.

Run it with ``settlrl-render-bot`` (``BOT_HOST`` / ``BOT_PORT`` / ``RELOAD``).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, HTTPException

from .bots import bot_catalog
from .providers import ActRequest, ActResponse
from .session import GameSession, GameSetup, IllegalActionError

# Distinct games kept warm for incremental replay; past this the
# least-recently-used is dropped (its next request rebuilds from setup).
_CACHE_CAP = 64


class _SessionCache:
    """One replayed :class:`GameSession` per ``game_id``, at its applied move
    count, so a request only replays moves beyond what's cached. Bounded LRU;
    thread-safe (the worker handles requests in a threadpool)."""

    def __init__(self, cap: int = _CACHE_CAP) -> None:
        self._cap = cap
        self._by_id: OrderedDict[str, tuple[GameSession, int]] = OrderedDict()
        self._lock = threading.Lock()

    def session_for(
        self, game_id: str, setup: dict[str, Any], moves: list[int]
    ) -> GameSession:
        with self._lock:
            cached = self._by_id.get(game_id)
            if cached is not None and cached[1] <= len(moves):
                session, applied = cached  # extend the warm session
            else:
                session, applied = GameSession.from_setup(GameSetup.from_dict(setup)), 0
            for flat in moves[applied:]:
                session.apply(flat)
            self._by_id[game_id] = (session, len(moves))
            self._by_id.move_to_end(game_id)
            while len(self._by_id) > self._cap:
                self._by_id.popitem(last=False)
            return session


def create_bot_app() -> FastAPI:
    app = FastAPI(title="Settlrl Bot Service")
    cache = _SessionCache()

    @app.get("/catalog")
    def catalog() -> dict[str, dict[str, object]]:
        return bot_catalog()

    @app.post("/act")
    def act(req: ActRequest) -> ActResponse:
        try:
            session = cache.session_for(req.game_id, req.setup, req.moves)
        except (KeyError, ValueError, TypeError, IllegalActionError) as exc:
            raise HTTPException(
                status_code=422, detail=f"cannot reconstruct game: {exc}"
            ) from exc
        if session.acting_seat() != req.seat:
            raise HTTPException(status_code=409, detail="requested seat is not acting")
        flat = session.bot_choice()
        if flat is None:
            raise HTTPException(
                status_code=409, detail="no bot move available for that seat"
            )
        return ActResponse(flat=flat)

    return app


app = create_bot_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "settlrl_render.bot_service:app",
        host=os.environ.get("BOT_HOST", "0.0.0.0"),
        port=int(os.environ.get("BOT_PORT", "8100")),
        reload=bool(int(os.environ.get("RELOAD", "0"))),
    )
