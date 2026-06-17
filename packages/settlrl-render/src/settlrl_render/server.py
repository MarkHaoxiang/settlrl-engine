"""The FastAPI app. Routes stay thin: resolve the game, check seat tokens,
hold the per-game lock, map errors to status codes — game logic lives in
``session``, seat claims in ``games``, and what a requester may see in
``views``. ``create_app`` is the composition root, so tests build isolated
apps with their own registries instead of sharing module state.
"""

import json
import os
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import anyio.to_thread
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from settlrl_engine.record import GameRecord, ReplayError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import Scope

from .auth import Auth, UserStore
from .bots import bot_catalog
from .driver import start_game_driver
from .games import (
    GameHandle,
    GameRegistry,
    QueuePosition,
    RegistryFullError,
    restore_registry,
)
from .models import GameModel, ReplayStateModel
from .replay import ReplaySession
from .session import HUMAN, GameSession, IllegalActionError
from .store import GameStore
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


def _build_registry(
    games: GameRegistry | None, state_dir: str | None, max_active: int
) -> GameRegistry:
    """The app's registry: the one passed in (tests), one restored from the
    store, or a fresh in-memory one."""
    if games is not None:
        return games
    if state_dir:
        return restore_registry(GameStore(state_dir), max_active=max_active)
    return GameRegistry(max_active=max_active)


def _user_store(user_db: str | None, state_dir: str | None) -> UserStore:
    """The accounts db: an explicit path, ``users.db`` under the state dir when
    persistence is on, or an ephemeral in-memory db (tests / stateless runs)."""
    if user_db is not None:
        return UserStore(user_db)
    if state_dir:
        return UserStore(str(Path(state_dir) / "users.db"))
    return UserStore(None)


def _needs_driver(handle: GameHandle, turn_timeout: float) -> bool:
    """A non-terminal game needs the server-side driver when it has a bot seat
    to pace, or a turn timeout to enforce on human seats (started on create,
    restarted for restored games)."""
    session = handle.session
    if session.terminal():
        return False
    return turn_timeout > 0 or any(k != HUMAN for k in session.seats)


SeatTokens = Annotated[str | None, Header(alias="X-Seat-Tokens")]

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
    # The caller's place in line from a prior queued response, re-sent each poll
    # while waiting for a free slot; None on the first attempt.
    ticket: str | None = None


class _CreatedModel(BaseModel):
    """A fresh game: its id and the creator's seat tokens."""

    id: str
    seats: list[str]
    tokens: dict[int, str]


class _QueuedModel(BaseModel):
    """The server is at its concurrency cap: the caller's place in line. They
    re-POST with ``ticket`` until they get a :class:`_CreatedModel` back."""

    queued: Literal[True] = True
    ticket: str
    position: int
    total: int


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
    bot_delay: float = 0.65,
    root_path: str = "",
    max_streams: int = 64,
    max_body_bytes: int = 2 * 1024 * 1024,
    state_dir: str | None = None,
    turn_timeout: float = 0.0,
    max_active: int = 16,
    warm: bool = True,
    user_db: str | None = None,
    admin_emails: frozenset[str] = frozenset(),
) -> FastAPI:
    """Build the app around its own registry (tests pass theirs in).

    ``bot_delay`` paces the server-side bot driver (seconds between bot moves,
    so clients can animate each one). ``root_path`` is the proxy prefix the app
    is served under.
    ``max_streams`` caps concurrent event-stream subscribers (each pins a
    threadpool thread, so this must stay well under the pool size or idle
    streams starve ordinary requests); ``max_body_bytes`` rejects oversized
    request bodies before they are parsed. ``state_dir`` turns on persistence:
    games are journalled there and replayed back on the next startup (ignored
    when ``games`` is passed). ``turn_timeout`` (seconds, 0 = off) auto-plays a
    human turn that has gone idle that long, so an abandoned game still finishes.
    ``max_active`` caps how many games run at once; beyond it, new creators are
    queued (``POST /api/games`` returns their place in line). ``warm``
    pre-compiles the engine at startup (off in tests, which compile lazily and
    don't want the background contention). ``user_db`` is the accounts SQLite
    path (defaults to ``users.db`` under ``state_dir``, else in-memory);
    ``admin_emails`` are granted admin on register / login.
    """
    registry = _build_registry(games, state_dir, max_active)
    auth = Auth(_user_store(user_db, state_dir), admin_emails=admin_emails)
    replays = _ReplaySlot()
    # Each live event stream holds one permit for its whole connection; past
    # the cap, new subscribers get 503 rather than exhausting the threadpool.
    sse_gate = threading.Semaphore(max_streams)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if warm:
            threading.Thread(target=_warm_jit_cache, daemon=True).start()
        # Each event-stream subscriber occupies a threadpool thread for its
        # whole connection; the anyio default (40) would cap concurrent clients
        # and then starve ordinary requests.
        anyio.to_thread.current_default_thread_limiter().total_tokens = 160
        # Resume pacing/timeouts for any games replayed in from the store.
        for handle in registry.all_handles():
            if _needs_driver(handle, turn_timeout):
                start_game_driver(handle, bot_delay, turn_timeout)
        yield

    app = FastAPI(title="Settlrl Render", lifespan=lifespan, root_path=root_path)

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
        req: _CreateRequest, response: Response
    ) -> _CreatedModel | _QueuedModel:
        """Create a game, or return the caller's place in line when the server
        is at its concurrency cap (a ``202`` they re-POST with ``ticket``)."""
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
            seated = registry.admit(session, req.ticket)
        except RegistryFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(seated, QueuePosition):
            response.status_code = 202
            return _QueuedModel(
                ticket=seated.ticket, position=seated.position, total=seated.total
            )
        humans = seated.human_seats()
        claiming = (
            humans if req.claim == "all" else humans[:1] if req.claim == "first" else []
        )
        tokens = dict(seated.claim(seat) for seat in claiming)
        if _needs_driver(seated, turn_timeout):
            start_game_driver(seated, bot_delay, turn_timeout)
        return _CreatedModel(id=seated.id, seats=session.seats, tokens=tokens)

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
            if handle.journal is not None:
                handle.journal.chat(req.player, text)
            handle.bump()
            return game_model(handle, owned)

    @app.get("/api/games/{game_id}/record")
    def get_record(game_id: str) -> Response:
        """The finished game as ``settlrl_engine.record`` JSON -- a
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
        """Load a game record (the ``settlrl_engine.record`` JSON document) for
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
        """Bot kinds available for seats (settlrl-agents names), each with the
        player counts it supports and its configurable build parameters."""
        return bot_catalog()

    app.include_router(auth.router)

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


# Serve built frontend when it exists (src/settlrl_render/server.py -> package root)
_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"


# The uvicorn entry point (settlrl_render.server:app). ROOT_PATH is the proxy
# prefix when served under a path (e.g. /settlrl behind Caddy's handle_path).
def _admin_emails() -> frozenset[str]:
    raw = os.environ.get("SETTLRL_RENDER_ADMIN_EMAILS", "")
    return frozenset(e.strip() for e in raw.split(",") if e.strip())


app = create_app(
    root_path=os.environ.get("ROOT_PATH", ""),
    max_streams=int(os.environ.get("SETTLRL_RENDER_MAX_STREAMS", "64")),
    state_dir=os.environ.get("SETTLRL_RENDER_STATE_DIR") or None,
    turn_timeout=float(os.environ.get("SETTLRL_RENDER_TURN_TIMEOUT_S", "0")),
    max_active=int(os.environ.get("SETTLRL_RENDER_MAX_ACTIVE", "16")),
    user_db=os.environ.get("SETTLRL_RENDER_USER_DB") or None,
    admin_emails=_admin_emails(),
)
