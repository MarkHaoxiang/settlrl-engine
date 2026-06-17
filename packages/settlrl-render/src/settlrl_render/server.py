"""The FastAPI app: the composition root. ``create_app`` wires the one async
:class:`~settlrl_render.db.Database`, the game registry and its journal store,
the auth system, the bot providers, and the bot/timeout driver tasks into a
shared :class:`~settlrl_render.deps.Deps`, then mounts the routers
(:mod:`settlrl_render.routers`) and the SPA. Tests build isolated apps with
their own registries instead of sharing module state.
"""

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import Scope

from . import routers
from .auth import Auth
from .db import Database
from .deps import Deps, ReplaySlot, needs_driver
from .driver import start_game_driver
from .games import GameHandle, GameRegistry, restore_games
from .providers import ProviderRegistry
from .session import GameSession
from .store import GameStore


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
    games: GameRegistry | None,
    database: Database,
    state_dir: str | None,
    max_active: int,
) -> tuple[GameRegistry, GameStore | None]:
    """The app's registry and (when persistence is on) its journal store: the
    registry passed in by tests, a fresh persistent one backed by ``database``,
    or a fresh in-memory one. The store's games are replayed in at startup."""
    if games is not None:
        return games, None
    if state_dir:
        store = GameStore(database)
        return GameRegistry(max_active=max_active, store=store), store
    return GameRegistry(max_active=max_active), None


def _database(user_db: str | None, state_dir: str | None) -> Database:
    """The one async db: an explicit path, ``settlrl.db`` under the state dir
    when persistence is on, or an ephemeral in-memory db (tests / stateless)."""
    if user_db is not None:
        return Database(user_db)
    if state_dir:
        return Database(str(Path(state_dir) / "settlrl.db"))
    return Database(None)


def create_app(
    games: GameRegistry | None = None,
    bot_delay: float = 0.65,
    root_path: str = "",
    max_body_bytes: int = 2 * 1024 * 1024,
    state_dir: str | None = None,
    turn_timeout: float = 0.0,
    max_active: int = 16,
    warm: bool = True,
    user_db: str | None = None,
    admin_emails: frozenset[str] = frozenset(),
    providers: ProviderRegistry | None = None,
    local_bots: bool = True,
) -> FastAPI:
    """Build the app around its own registry (tests pass theirs in).

    ``bot_delay`` paces the server-side bot driver (seconds between bot moves,
    so clients can animate each one). ``root_path`` is the proxy prefix the app
    is served under. ``max_body_bytes`` rejects oversized request bodies before
    they are parsed. ``state_dir`` turns on persistence: games are journalled
    there and replayed back on the next startup (ignored when ``games`` is
    passed). ``turn_timeout`` (seconds, 0 = off) auto-plays a human turn that has
    gone idle that long, so an abandoned game still finishes. ``max_active`` caps
    how many games run at once; beyond it, new creators are queued
    (``POST /api/games`` returns their place in line). ``warm`` pre-compiles the
    engine at startup (off in tests, which compile lazily and don't want the
    background contention). ``user_db`` overrides the shared SQLite path
    (defaults to ``settlrl.db`` under ``state_dir``, else in-memory);
    ``admin_emails`` are granted admin on register / login.
    """
    database = _database(user_db, state_dir)
    registry, store = _build_registry(games, database, state_dir, max_active)
    auth = Auth(database, admin_emails=admin_emails)
    bots = (
        providers if providers is not None else ProviderRegistry(local_bots=local_bots)
    )
    bots_owned = providers is None  # only close the HTTP client we created
    # The live bot/timeout driver tasks, so the lifespan can cancel them on
    # shutdown; each removes itself when it ends (game over or evicted).
    drivers: set[asyncio.Task[None]] = set()

    def spawn_driver(handle: GameHandle) -> None:
        task = start_game_driver(handle, bot_delay, turn_timeout, bots)
        drivers.add(task)
        task.add_done_callback(drivers.discard)

    deps = Deps(
        registry=registry,
        bots=bots,
        auth=auth,
        replays=ReplaySlot(),
        spawn_driver=spawn_driver,
        turn_timeout=turn_timeout,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await database.init()
        if store is not None:
            store.start()
            await restore_games(registry, store)
        if warm:
            threading.Thread(target=_warm_jit_cache, daemon=True).start()
        # Resume pacing/timeouts for any games replayed in from the store.
        for handle in registry.all_handles():
            if needs_driver(handle, turn_timeout):
                spawn_driver(handle)
        yield
        for task in drivers:
            task.cancel()
        if store is not None:
            await store.aclose()  # flush queued writes before the engine closes
        if bots_owned:
            await bots.aclose()
        await database.dispose()

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

    for module in (routers.games, routers.replay, routers.bots, routers.me):
        app.include_router(module.build(deps))
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
    state_dir=os.environ.get("SETTLRL_RENDER_STATE_DIR") or None,
    turn_timeout=float(os.environ.get("SETTLRL_RENDER_TURN_TIMEOUT_S", "0")),
    max_active=int(os.environ.get("SETTLRL_RENDER_MAX_ACTIVE", "16")),
    user_db=os.environ.get("SETTLRL_RENDER_USER_DB") or None,
    admin_emails=_admin_emails(),
    local_bots=os.environ.get("SETTLRL_RENDER_LOCAL_BOTS", "1") != "0",
)
