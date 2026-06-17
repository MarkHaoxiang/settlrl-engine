"""The bot service: the agents' move-serving endpoint, behind the standardized
bot wire protocol (:mod:`settlrl_game.botproto`).

It runs the ``settlrl-agents`` policies and answers two requests:

* ``GET /catalog`` — the bot kinds it offers (same shape as the game server's
  ``GET /api/bots``).
* ``POST /act`` — given a game's setup and its flat moves so far, replay them
  and return the move the acting seat's bot would play.

It holds no game registry and no per-seat secrets; it is a pure function of the
game record, so it can be deployed and scaled independently of the game server.
A small per-``game_id`` LRU cache keeps a replayed session around so a move
request only has to apply the new tail rather than the whole game each time.

Run it with ``settlrl-bot-service`` (``BOT_HOST`` / ``BOT_PORT`` / ``RELOAD``).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

import anyio.to_thread
import jax
from fastapi import FastAPI, HTTPException
from settlrl_game.botproto import ActRequest, ActResponse
from settlrl_game.session import (
    HUMAN,
    GameSession,
    GameSetup,
    IllegalActionError,
)

from settlrl_agents.service.bots import bot_act, bot_catalog
from settlrl_agents.service.bridge import engine_env, game_flat

# The agent kinds this service plays; the replayed game's bot seats are accepted
# as these (the game server stores them verbatim and never plays them itself).
_KINDS = frozenset(bot_catalog())

# Distinct games kept warm for incremental replay; past this the
# least-recently-used is dropped (its next request rebuilds from setup).
_CACHE_CAP = 64


class _SessionCache:
    """One replayed :class:`GameSession` per ``game_id``, at its applied move
    count, so a request only replays moves beyond what's cached. Bounded LRU;
    thread-safe (replay is offloaded to a worker thread)."""

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
                session = GameSession.from_setup(
                    GameSetup.from_dict(setup), external_kinds=_KINDS
                )
                applied = 0
            for flat in moves[applied:]:
                session.apply(flat)
            self._by_id[game_id] = (session, len(moves))
            self._by_id.move_to_end(game_id)
            while len(self._by_id) > self._cap:
                self._by_id.popitem(last=False)
            return session


def _choose(session: GameSession, seat: int) -> int | None:
    """The flat move the seat's bot plays in ``session`` (None when no bot move
    is due: the game is over, the seat is human, or it has no legal move).

    The agent reasons on an engine env bridged from the reference game, then its
    chosen engine action is translated back to the game's flat index.
    """
    if session.terminal() or session.seats[seat] == HUMAN:
        return None
    if not session.legal_flat():
        return None
    env = engine_env(session.game, session.belief_state)
    # Reproducible per position, independent of the bridged env's own key.
    key = jax.random.fold_in(jax.random.key(0), len(session.moves_flat()))
    engine_flat = bot_act(
        session.seats[seat], session.seat_params[seat], key, env, seat
    )
    return game_flat(engine_flat)


def create_bot_app() -> FastAPI:
    app = FastAPI(title="Settlrl Bot Service")
    cache = _SessionCache()

    @app.get("/catalog")
    def catalog() -> dict[str, dict[str, object]]:
        return bot_catalog()

    @app.post("/act")
    async def act(req: ActRequest) -> ActResponse:
        try:
            session = await anyio.to_thread.run_sync(
                cache.session_for, req.game_id, req.setup, req.moves
            )
        except (KeyError, ValueError, TypeError, IllegalActionError) as exc:
            raise HTTPException(
                status_code=422, detail=f"cannot reconstruct game: {exc}"
            ) from exc
        if session.acting_seat() != req.seat:
            raise HTTPException(status_code=409, detail="requested seat is not acting")
        flat = await anyio.to_thread.run_sync(_choose, session, req.seat)
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
        "settlrl_agents.service.app:app",
        host=os.environ.get("BOT_HOST", "0.0.0.0"),
        port=int(os.environ.get("BOT_PORT", "8100")),
        reload=bool(int(os.environ.get("RELOAD", "0"))),
    )
