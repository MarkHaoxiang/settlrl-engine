"""The FastAPI app. Routes stay thin: resolve the game, check seat tokens,
hold the per-game lock, map errors to status codes — game logic lives in
``session``, seat claims in ``games``, and what a requester may see in
``views``. ``create_app`` is the composition root, so tests build isolated
apps with their own registries instead of sharing module state.
"""

import json
import os
import secrets
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import anyio.to_thread
from catan_engine.record import GameRecord, ReplayError
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import Scope

from .bots import bot_catalog
from .driver import start_bot_driver
from .games import GameHandle, GameRegistry, RegistryFullError
from .models import GameModel, ReplayStateModel
from .replay import ReplaySession
from .session import HUMAN, GameSession, IllegalActionError
from .views import game_model


def _warm_jit_cache() -> None:
    """Play throwaway moves so XLA compiles before the first real click.

    The first engine step in a fresh process compiles for a couple of seconds,
    once per seat count (the compiled shapes depend on n_players). Scratch
    sessions take that hit at startup instead of the user's first placement;
    the in-process jit cache is shared, so live sessions then step in
    milliseconds.
    """
    for n_players in (4, 2):
        scratch = GameSession(seed=0, n_players=n_players)
        scratch.apply(int(scratch.legal_flat()[0]))  # compiles the env step
        scratch.apply(int(scratch.legal_flat()[0]))
        scratch.bot_step()  # compiles the default bot policy


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=_warm_jit_cache, daemon=True).start()
    # Each event-stream subscriber occupies a threadpool thread for its whole
    # connection; the anyio default (40) would cap concurrent clients and
    # then starve ordinary requests.
    anyio.to_thread.current_default_thread_limiter().total_tokens = 160
    yield


SeatTokens = Annotated[str | None, Header(alias="X-Seat-Tokens")]
CreateKey = Annotated[str | None, Header(alias="X-Create-Key")]

# Replaying a submitted record steps the engine once per move; cap the count so
# an untrusted POST /api/replay can't hand us an arbitrarily long game to grind
# through. Well above any real game (random games run a few thousand moves).
_MAX_REPLAY_MOVES = 20_000


def _tokens(header: str | None) -> list[str]:
    return [t.strip() for t in (header or "").split(",") if t.strip()]


class _ReplaySlot:
    """The loaded replay, if any (server-wide tooling; one at a time)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.session: ReplaySession | None = None


class _SeatSpec(BaseModel):
    """A configured bot seat: its kind plus knob overrides from the catalog."""

    kind: str
    params: dict[str, float | int | bool] = {}


class _CreateRequest(BaseModel):
    seed: int = 0
    # Seats in the new game. The engine supports 2-4; the renderer offers 2
    # and 4 for now.
    n_players: Literal[2, 4] = 4
    number_placement: Literal["random", "spiral"] = "random"
    # What controls each seat: "human", a bot kind from GET /api/bots, or a
    # configured bot {"kind", "params"}; no seat has to be human. None seats a
    # human on seat 0 and "random" bots elsewhere.
    seats: list[str | _SeatSpec] | None = None
    # Human seats the creator claims immediately: every one ("all", the
    # hotseat default), just the first ("first", online play — others join
    # via POST /join), or none.
    claim: Literal["all", "first", "none"] = "all"


class _CreatedModel(BaseModel):
    """A fresh game: its id and the creator's seat tokens."""

    id: str
    seats: list[str]
    tokens: dict[int, str]


class _JoinRequest(BaseModel):
    # A specific human seat, or None for the first unclaimed one.
    seat: int | None = None


class _JoinedModel(BaseModel):
    id: str
    seat: int
    token: str


class _ActionRequest(BaseModel):
    flat: int


class _ChatRequest(BaseModel):
    text: str
    # Seat the message belongs to (must be owned); None for a spectator.
    player: int | None = None


def create_app(
    games: GameRegistry | None = None,
    create_key: str | None = None,
    bot_delay: float = 0.65,
    root_path: str = "",
    max_streams: int = 64,
    max_body_bytes: int = 2 * 1024 * 1024,
) -> FastAPI:
    """Build the app around its own registry (tests pass theirs in).

    ``create_key`` gates game creation: when set, ``POST /api/games`` requires
    a matching ``X-Create-Key`` header. Public deployments set it so strangers
    can't spam games and exhaust the registry. ``bot_delay`` paces the
    server-side bot driver (seconds between bot moves, so clients can animate
    each one). ``root_path`` is the proxy prefix the app is served under.
    ``max_streams`` caps concurrent event-stream subscribers (each pins a
    threadpool thread, so this must stay well under the pool size or idle
    streams starve ordinary requests); ``max_body_bytes`` rejects oversized
    request bodies before they are parsed.
    """
    registry = games if games is not None else GameRegistry()
    replays = _ReplaySlot()
    # Each live event stream holds one permit for its whole connection; past
    # the cap, new subscribers get 503 rather than exhausting the threadpool.
    sse_gate = threading.Semaphore(max_streams)
    app = FastAPI(title="Catan Render", lifespan=_lifespan, root_path=root_path)

    @app.middleware("http")
    async def _limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if (
            declared is not None
            and declared.isdigit()
            and int(declared) > max_body_bytes
        ):
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    def handle_of(game_id: str) -> GameHandle:
        handle = registry.get(game_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="no such game")
        return handle

    @app.post("/api/games")
    def post_create(
        req: _CreateRequest, x_create_key: CreateKey = None
    ) -> _CreatedModel:
        if create_key is not None and not (
            x_create_key is not None
            and secrets.compare_digest(x_create_key, create_key)
        ):
            raise HTTPException(
                status_code=403, detail="creation requires the host key"
            )
        seats = (
            [s if isinstance(s, str) else s.model_dump() for s in req.seats]
            if req.seats is not None
            else None
        )
        try:
            session = GameSession(seed=req.seed, n_players=req.n_players)
            session.reset(req.seed, number_placement=req.number_placement, seats=seats)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            handle = registry.create(session)
        except RegistryFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        humans = handle.human_seats()
        claiming = (
            humans if req.claim == "all" else humans[:1] if req.claim == "first" else []
        )
        tokens = dict(handle.claim(seat) for seat in claiming)
        if any(kind != HUMAN for kind in session.seats):
            start_bot_driver(handle, bot_delay)
        return _CreatedModel(id=handle.id, seats=session.seats, tokens=tokens)

    @app.post("/api/games/{game_id}/join")
    def post_join(game_id: str, req: _JoinRequest) -> _JoinedModel:
        handle = handle_of(game_id)
        with handle.lock:
            try:
                seat, token = handle.claim(req.seat)
            except (LookupError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            handle.bump()
        return _JoinedModel(id=game_id, seat=seat, token=token)

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str, x_seat_tokens: SeatTokens = None) -> GameModel:
        handle = handle_of(game_id)
        with handle.lock:
            return game_model(handle, handle.owned_seats(_tokens(x_seat_tokens)))

    @app.post("/api/games/{game_id}/action")
    def post_action(
        game_id: str, req: _ActionRequest, x_seat_tokens: SeatTokens = None
    ) -> GameModel:
        """Apply the acting seat's move; the request must prove that seat."""
        handle = handle_of(game_id)
        with handle.lock:
            owned = handle.owned_seats(_tokens(x_seat_tokens))
            if handle.session.acting_seat() not in owned:
                raise HTTPException(status_code=403, detail="not your seat")
            try:
                handle.session.apply(req.flat)
            except IllegalActionError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            handle.bot_move = None
            handle.bump()
            return game_model(handle, owned)

    @app.get("/api/games/{game_id}/events")
    def get_events(game_id: str, x_seat_tokens: SeatTokens = None) -> StreamingResponse:
        """Server-sent events: the requester's snapshot now, then again on
        every state change (moves, bot plays, chat, joins). Comment lines
        keep idle connections alive."""
        handle = handle_of(game_id)
        tokens = _tokens(x_seat_tokens)
        if not sse_gate.acquire(blocking=False):
            raise HTTPException(
                status_code=503, detail="too many event streams; try again"
            )

        def stream() -> Iterator[str]:
            seen = -1
            try:
                while True:
                    with handle.lock:
                        if handle.version == seen:
                            handle.lock.wait(timeout=15.0)
                        if handle.closed:
                            return
                        if handle.version == seen:
                            body = None  # idle: fall through to a keepalive
                        else:
                            seen = handle.version
                            model = game_model(handle, handle.owned_seats(tokens))
                            body = model.model_dump_json()
                    yield f"data: {body}\n\n" if body else ": keepalive\n\n"
            finally:
                sse_gate.release()

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/games/{game_id}/chat")
    def post_chat(
        game_id: str, req: _ChatRequest, x_seat_tokens: SeatTokens = None
    ) -> GameModel:
        """Append a chat message to the game log."""
        text = req.text.strip()
        if not text or len(text) > 500:
            raise HTTPException(
                status_code=422, detail="chat text must be 1-500 characters"
            )
        handle = handle_of(game_id)
        with handle.lock:
            owned = handle.owned_seats(_tokens(x_seat_tokens))
            if req.player is not None and req.player not in owned:
                raise HTTPException(status_code=403, detail="not your seat")
            handle.session.add_chat(req.player, text)
            handle.bump()
            return game_model(handle, owned)

    @app.get("/api/games/{game_id}/record")
    def get_record(game_id: str) -> Response:
        """The finished game as ``catan_engine.record`` JSON -- a
        self-contained, replayable transcript. 409 while running: replaying a
        record reconstructs hidden hands, so live games don't export."""
        handle = handle_of(game_id)
        with handle.lock:
            if not handle.session.terminal():
                raise HTTPException(status_code=409, detail="game still running")
            body = handle.session.record().to_json()
        return Response(content=body, media_type="application/json")

    def load_replay(record: GameRecord) -> ReplayStateModel:
        try:
            session = ReplaySession(record)
        except (ReplayError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with replays.lock:
            replays.session = session
            return session.state(0)

    @app.post("/api/games/{game_id}/replay")
    def post_replay_from_game(game_id: str) -> ReplayStateModel:
        """Load a finished game for replay (409 while it is still running:
        replaying reconstructs hidden hands)."""
        handle = handle_of(game_id)
        with handle.lock:
            if not handle.session.terminal():
                raise HTTPException(status_code=409, detail="game still running")
            record = handle.session.record()
        return load_replay(record)

    @app.post("/api/replay")
    def post_replay(doc: dict[str, object]) -> ReplayStateModel:
        """Load a game record (the ``catan_engine.record`` JSON document) for
        replay; returns the opening state. ``422`` if the record is malformed
        or fails replay validation."""
        try:
            record = GameRecord.from_json(json.dumps(doc))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"bad record: {exc}") from exc
        if len(record.moves) > _MAX_REPLAY_MOVES:
            raise HTTPException(
                status_code=422,
                detail=f"record has too many moves (max {_MAX_REPLAY_MOVES})",
            )
        return load_replay(record)

    @app.get("/api/replay/state")
    def get_replay_state(move: int = 0) -> ReplayStateModel:
        """The loaded replay after ``move`` moves (0 = the opening board).

        ``404`` when no replay is loaded; ``422`` when ``move`` is out of
        range.
        """
        with replays.lock:
            if replays.session is None:
                raise HTTPException(status_code=404, detail="no replay loaded")
            try:
                return replays.session.state(move)
            except IndexError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/replay/record")
    def get_replay_record() -> Response:
        """The loaded replay's record JSON (e.g. to save it to a file)."""
        with replays.lock:
            if replays.session is None:
                raise HTTPException(status_code=404, detail="no replay loaded")
            body = replays.session.record.to_json()
        return Response(content=body, media_type="application/json")

    @app.get("/api/bots")
    def get_bots() -> dict[str, dict[str, object]]:
        """Bot kinds available for seats (catan-agents names), each with the
        player counts it supports and its configurable build parameters."""
        return bot_catalog()

    if _dist.exists():
        app.mount("/", _SPAStaticFiles(directory=_dist, html=True), name="static")
    return app


class _SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for unknown paths.

    The frontend uses client-side routing (/play, /replay/:id), so a deep link
    or a refresh on those paths must still return the SPA entry point rather
    than a 404. Only extension-less paths fall back; a request for a missing
    file (e.g. a stale /assets/*.js) still returns 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in path.rsplit("/", 1)[-1]:
                return FileResponse(_dist / "index.html")
            raise


# Serve built frontend when it exists (src/catan_render/server.py -> package root)
_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"

# The uvicorn entry point (catan_render.server:app). ROOT_PATH is the proxy
# prefix when served under a path (e.g. /catan behind Caddy's handle_path).
# An empty CATAN_RENDER_CREATE_KEY means "no key" (compose substitutes an empty
# string for an unset value), not a key that happens to be the empty string.
app = create_app(
    create_key=os.environ.get("CATAN_RENDER_CREATE_KEY") or None,
    root_path=os.environ.get("ROOT_PATH", ""),
    max_streams=int(os.environ.get("CATAN_RENDER_MAX_STREAMS", "64")),
)
