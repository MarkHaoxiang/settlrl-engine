"""The bot service: a FastAPI app serving one :class:`~settlrl_agents.service.sdk.Bot`
over the bot wire protocol (:mod:`settlrl_game.botproto`).

It answers two requests:

* ``GET /info`` — the bot's identity (:class:`~settlrl_game.botproto.BotInfo`).
* ``POST /act`` — apply the moves the bot has not seen yet to the game it is
  tracking, then return the move the acting seat plays.

A :class:`_Tracker` keeps one replayed game per ``game_id`` (a bounded LRU), so a
turn only applies the few new moves rather than the whole game. When its tracked
game is not at the request's ``base`` (a fresh or restarted service), ``/act``
answers ``409 {"resync": true, "have": n}`` and the client re-sends from move ``n``.

Run it with ``settlrl-bot-service --bot KIND`` (env ``BOT_HOST`` / ``BOT_PORT`` /
``RELOAD`` / ``SETTLRL_BOT``).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Annotated

import anyio.to_thread
import typer
from fastapi import FastAPI, HTTPException
from settlrl_game.actions import flat_for_move
from settlrl_game.botproto import ActRequest, ActResponse, BotInfo
from settlrl_game.session import HUMAN, GameSession, GameSetup, IllegalActionError

from settlrl_agents.service.sdk import Bot, GameView

# Distinct games kept warm for incremental replay; past this the least-recently-
# used is dropped (its next request resyncs from scratch).
_CACHE_CAP = 64


class _Resync(Exception):
    """The tracker is not at the request's ``base``; the client must re-send from
    ``have`` (the moves the tracker actually holds)."""

    def __init__(self, have: int) -> None:
        self.have = have


class _Tracker:
    """One replayed :class:`GameSession` per ``game_id``, at its applied move
    count, so a request only applies moves beyond what's tracked. Bounded LRU;
    thread-safe (replay is offloaded to a worker thread)."""

    def __init__(self, bot: Bot, cap: int = _CACHE_CAP) -> None:
        self._bot = bot
        self._cap = cap
        self._games: OrderedDict[str, GameSession] = OrderedDict()
        self._lock = threading.Lock()

    def view_for(self, req: ActRequest) -> GameView:
        """Advance ``game_id`` by the request's moves and return the seat's view.

        Raises :class:`_Resync` when the tracked game is not at ``req.base``."""
        with self._lock:
            session = self._games.get(req.game_id)
            if session is None:
                if req.base != 0:
                    raise _Resync(0)
                setup = GameSetup.from_dict(req.setup)
                external = frozenset(s for s in setup.seats if s != HUMAN)
                session = GameSession.from_setup(setup, external_kinds=external)
                self._bot.new_game(req.game_id, setup, req.seat)
                self._games[req.game_id] = session
            applied = len(session.moves_flat())
            if applied != req.base:
                raise _Resync(applied)
            for move in req.moves:
                session.apply(flat_for_move(move))
            self._games.move_to_end(req.game_id)
            while len(self._games) > self._cap:
                dropped, _ = self._games.popitem(last=False)
                self._bot.end_game(dropped)
            return GameView(req.game_id, req.seat, session)


def create_app(bot: Bot) -> FastAPI:
    app = FastAPI(title=f"Settlrl Bot — {bot.title}")
    tracker = _Tracker(bot)

    @app.get("/info")
    def info() -> BotInfo:
        return bot.info()

    @app.post("/act")
    async def act(req: ActRequest) -> ActResponse:
        try:
            view = await anyio.to_thread.run_sync(tracker.view_for, req)
        except _Resync as resync:
            raise HTTPException(
                status_code=409, detail={"resync": True, "have": resync.have}
            ) from resync
        except (KeyError, ValueError, TypeError, IllegalActionError) as exc:
            raise HTTPException(
                status_code=422, detail=f"cannot apply moves: {exc}"
            ) from exc
        if view.session.terminal() or view.session.acting_seat() != req.seat:
            raise HTTPException(status_code=409, detail="requested seat is not acting")
        legal = set(view.session.legal_flat())
        if not legal:  # the acting seat always has a move; guard a NaN-prone mask
            raise HTTPException(status_code=409, detail="no legal move for that seat")
        move = await anyio.to_thread.run_sync(bot.act, view)
        try:
            if flat_for_move(move) not in legal:
                raise ValueError("move is not legal in this position")
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"bot returned an illegal move: {exc}"
            ) from exc
        return ActResponse(move=move)

    return app


def bundled_app() -> FastAPI:
    """A bundled bot service for ``$SETTLRL_BOT`` (the reload worker's factory)."""
    from settlrl_agents.service.bots import make_bot

    return create_app(make_bot(os.environ.get("SETTLRL_BOT", "greedy")))


def serve(
    bot: Annotated[
        str, typer.Option(envvar="SETTLRL_BOT", help="the bundled bot to serve")
    ] = "greedy",
    host: Annotated[str, typer.Option(envvar="BOT_HOST")] = "0.0.0.0",
    port: Annotated[int, typer.Option(envvar="BOT_PORT")] = 8100,
    reload: Annotated[
        bool, typer.Option(envvar="RELOAD", help="dev autoreload")
    ] = False,
) -> None:
    """Run a Settlrl bot service for one bundled bot."""
    import uvicorn

    from settlrl_agents.service.bots import BUNDLED, make_bot

    if bot not in BUNDLED:
        raise typer.BadParameter(
            f"unknown bot {bot!r} (choose from {sorted(BUNDLED)})", param_hint="--bot"
        )
    os.environ["SETTLRL_BOT"] = bot  # so a reload worker's factory builds the same bot
    if reload:
        uvicorn.run(
            "settlrl_agents.service.app:bundled_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(make_bot(bot)), host=host, port=port)


def main() -> None:
    typer.run(serve)
