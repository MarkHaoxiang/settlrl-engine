"""Game routes for a game in progress — read / act / event-stream / chat /
record — plus an all-bot create and loading a finished game into replay.

Human games are staged in a lobby and materialised on start (see
:mod:`settlrl_app.api.routers.lobbies`); the only game *created* here is an
all-bot one (bots playing each other, for spectating or tooling), which has no
human host to go through a lobby. Routes stay thin: resolve the game, check seat
tokens, hold the per-game lock, offload the blocking engine step, map errors to
status codes.
"""

from collections.abc import AsyncIterator
from typing import Annotated

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from settlrl_game.models import GameModel, ReplayStateModel
from settlrl_game.record import GameRecord
from settlrl_game.session import HUMAN, GameSession, IllegalActionError
from sse_starlette.sse import EventSourceResponse
from starlette.responses import Response

from settlrl_app.api.deps import Deps, SeatTokens, needs_driver, tokens, uid
from settlrl_app.api.routers.replay import load_replay
from settlrl_app.api.views import game_model
from settlrl_app.game.games import RegistryFullError, win_threshold
from settlrl_app.storage.db import User


class _BotGameRequest(BaseModel):
    seed: int = 0
    # Every seat is a bot kind from GET /api/bots — a human seat must be hosted
    # through a lobby instead.
    seats: list[str]


class _CreatedModel(BaseModel):
    id: str


class _ActionRequest(BaseModel):
    flat: int


class _ChatRequest(BaseModel):
    text: str
    # Seat the message belongs to (must be owned); None for a spectator.
    player: int | None = None


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    bots = deps.bots
    CurrentUser = Annotated[User | None, Depends(deps.auth.optional_user)]

    @router.post("/api/games")
    async def post_bot_game(req: _BotGameRequest) -> _CreatedModel:
        """Create an all-bot game (bots playing each other). Human games are
        hosted through a lobby, so a ``human`` seat here is ``422``; an unknown
        bot kind or one that doesn't support the count is ``422`` too; ``503``
        when every game slot is taken."""
        n = len(req.seats)
        if n not in (2, 3, 4):
            raise HTTPException(status_code=422, detail="a game seats 2-4 players")
        catalog = bots.catalog()
        for kind in req.seats:
            if kind == HUMAN:
                raise HTTPException(
                    status_code=422, detail="host a lobby for human seats"
                )
            spec = catalog.get(kind)
            if spec is None:
                raise HTTPException(status_code=422, detail=f"unknown bot: {kind!r}")
            counts = spec.get("counts", [])
            if isinstance(counts, list) and n not in counts:
                raise HTTPException(
                    status_code=422, detail=f"{kind} is not available at {n} players"
                )
        session = GameSession(seed=req.seed, n_players=n)
        await anyio.to_thread.run_sync(
            lambda: session.reset(
                req.seed,
                seats=req.seats,
                external_kinds=bots.remote_kinds(),
                victory_points_to_win=win_threshold(n),
            )
        )
        try:
            handle = deps.registry.create(session)
        except RegistryFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if needs_driver(handle, deps.turn_timeout):
            deps.spawn_driver(handle)
        handle.bump()
        return _CreatedModel(id=handle.id)

    @router.get("/api/games/{game_id}")
    async def get_game(
        game_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> GameModel:
        handle = deps.handle_of(game_id)
        async with handle.lock:
            return game_model(
                handle, handle.owned_seats(tokens(x_seat_tokens), uid(user))
            )

    @router.post("/api/games/{game_id}/action")
    async def post_action(
        game_id: str,
        req: _ActionRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> GameModel:
        """Apply the acting seat's move; the request must prove that seat."""
        handle = deps.handle_of(game_id)
        async with handle.lock:
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            if handle.session.acting_seat() not in owned:
                raise HTTPException(status_code=403, detail="not your seat")
            try:
                await anyio.to_thread.run_sync(handle.session.apply, req.flat)
            except IllegalActionError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            handle.bot_move = None
            handle.bump()
            return game_model(handle, owned)

    @router.get("/api/games/{game_id}/events")
    async def get_events(
        game_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> EventSourceResponse:
        """Server-sent events: the requester's snapshot now, then again on every
        state change (moves, bot plays, chat). ``EventSourceResponse`` adds the
        SSE framing, keepalive pings, and client-disconnect teardown."""
        handle = deps.handle_of(game_id)
        seat_tokens = tokens(x_seat_tokens)
        user_id = uid(user)

        async def stream() -> AsyncIterator[str]:
            seen = -1
            while True:
                changed = handle._changed  # capture before serialising/waiting
                async with handle.lock:
                    if handle.closed:
                        return
                    if handle.version != seen:
                        seen = handle.version
                        body: str | None = game_model(
                            handle, handle.owned_seats(seat_tokens, user_id)
                        ).model_dump_json()
                    else:
                        body = None
                if body is not None:
                    yield body  # EventSourceResponse wraps it as a `data:` event
                else:
                    await changed.wait()  # idle: pings come from EventSourceResponse

        return EventSourceResponse(stream(), ping=15)

    @router.post("/api/games/{game_id}/chat")
    async def post_chat(
        game_id: str,
        req: _ChatRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> GameModel:
        """Append a chat message to the game log."""
        text = req.text.strip()
        if not text or len(text) > 500:
            raise HTTPException(
                status_code=422, detail="chat text must be 1-500 characters"
            )
        handle = deps.handle_of(game_id)
        async with handle.lock:
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            if req.player is not None and req.player not in owned:
                raise HTTPException(status_code=403, detail="not your seat")
            handle.session.add_chat(req.player, text)
            if handle.journal is not None:
                handle.journal.chat(req.player, text)
            handle.bump()
            return game_model(handle, owned)

    async def _finished_record(game_id: str) -> GameRecord:
        """A finished game's replayable record — from the live handle, or rebuilt
        from the journal store when the handle has been evicted. 409 while still
        running (replaying reconstructs hidden hands); 404 if unknown."""
        handle = deps.registry.get(game_id)
        if handle is not None:
            async with handle.lock:
                if not handle.session.terminal():
                    raise HTTPException(status_code=409, detail="game still running")
                return handle.session.record()
        if deps.store is not None:
            record = await deps.store.finished_record(game_id)
            if record is not None:
                return record
        raise HTTPException(status_code=404, detail="no such game")

    @router.get("/api/games/{game_id}/record")
    async def get_record(game_id: str) -> Response:
        """The finished game as ``GameRecord`` JSON -- a self-contained,
        replayable transcript (served for past games too, rebuilt from the
        store). 409 while running; 404 if unknown."""
        record = await _finished_record(game_id)
        return Response(content=record.to_json(), media_type="application/json")

    @router.post("/api/games/{game_id}/replay")
    async def post_replay_from_game(game_id: str) -> ReplayStateModel:
        """Load a finished game for replay (a past game too, rebuilt from the
        store). 409 while it is still running; 404 if unknown."""
        return await load_replay(deps.replays, await _finished_record(game_id))

    return router
