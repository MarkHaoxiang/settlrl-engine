import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from catan_engine.record import GameRecord, ReplayError
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from .actions import decode_actions
from .bots import bot_catalog
from .convert import board_to_model
from .models import BoardModel, BotMoveModel, GameModel, ReplayStateModel
from .replay import ReplaySession
from .session import GameSession, IllegalActionError


def _warm_jit_cache() -> None:
    """Play throwaway moves so XLA compiles before the first real click.

    The first engine step in a fresh process compiles for a couple of seconds,
    once per seat count (the compiled shapes depend on n_players). Scratch
    sessions take that hit at startup instead of the user's first placement;
    the in-process jit cache is shared, so the live session then steps in
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
    yield


app = FastAPI(title="Catan Render", lifespan=_lifespan)

# A single live game; each seat is a human or a bot per the last reset (an
# all-bot game is watchable). One game at a time; POST /api/game/reset starts
# a fresh one.
_SESSION = GameSession()

# FastAPI runs sync endpoints in a threadpool, and the session is not
# thread-safe (concurrent requests would race its legality check and step the
# engine twice) -- every session access is serialised.
_LOCK = threading.Lock()


def _game_model(bot_move: BotMoveModel | None = None) -> GameModel:
    """The full Play-view snapshot: board + turn status + the human's legal moves."""
    status = _SESSION.status()
    actions = (
        decode_actions([int(f) for f in _SESSION.legal_flat()])
        if status.your_turn
        else []
    )
    return GameModel(
        board=board_to_model(_SESSION.board),
        status=status,
        actions=actions,
        bot_move=bot_move,
        log=_SESSION.log(),
    )


@app.get("/api/board")
def get_board() -> BoardModel:
    """Board geometry + player stats (used by the Replay view and shared hook)."""
    with _LOCK:
        return board_to_model(_SESSION.board)


@app.get("/api/game")
def get_game() -> GameModel:
    with _LOCK:
        return _game_model()


class _ActionRequest(BaseModel):
    flat: int


@app.post("/api/game/action")
def post_action(req: _ActionRequest) -> GameModel:
    """Apply the acting human's move (bot replies are stepped via /api/game/bot)."""
    with _LOCK:
        try:
            _SESSION.apply(req.flat)
        except IllegalActionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _game_model()


@app.post("/api/game/bot")
def post_bot_step() -> GameModel:
    """Play one due bot move; the snapshot plus what was played (``bot_move``).

    ``bot_move`` is null when no bot move is due (a human seat is acting or the
    game is over) -- the client polls this until ``status.your_turn``.
    """
    with _LOCK:
        seat = _SESSION.acting_seat()
        flat = _SESSION.bot_step()
        move = (
            None
            if flat is None
            else BotMoveModel(player=seat, action=decode_actions([flat])[0])
        )
        return _game_model(bot_move=move)


@app.get("/api/game/record")
def get_record() -> Response:
    """The current game as ``catan_engine.record`` JSON -- a self-contained,
    replayable transcript (``winner`` is null while the game is running)."""
    with _LOCK:
        body = _SESSION.record().to_json()
    return Response(content=body, media_type="application/json")


# The loaded replay, if any (independent of the live game; one at a time).
_REPLAY: ReplaySession | None = None
_REPLAY_LOCK = threading.Lock()


def _load_replay(record: GameRecord) -> ReplayStateModel:
    """Replay ``record`` into a fresh session and return its opening state."""
    global _REPLAY
    try:
        session = ReplaySession(record)
    except (ReplayError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with _REPLAY_LOCK:
        _REPLAY = session
        return session.state(0)


@app.post("/api/replay")
def post_replay(doc: dict[str, object]) -> ReplayStateModel:
    """Load a game record (the ``catan_engine.record`` JSON document) for
    replay; returns the opening state. ``422`` if the record is malformed or
    fails replay validation."""
    try:
        record = GameRecord.from_json(json.dumps(doc))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"bad record: {exc}") from exc
    return _load_replay(record)


@app.post("/api/replay/from-game")
def post_replay_from_game() -> ReplayStateModel:
    """Load the live game (as played so far) for replay."""
    with _LOCK:
        record = _SESSION.record()
    return _load_replay(record)


@app.get("/api/replay/state")
def get_replay_state(move: int = 0) -> ReplayStateModel:
    """The loaded replay after ``move`` moves (0 = the opening board).

    ``404`` when no replay is loaded; ``422`` when ``move`` is out of range.
    """
    with _REPLAY_LOCK:
        if _REPLAY is None:
            raise HTTPException(status_code=404, detail="no replay loaded")
        try:
            return _REPLAY.state(move)
        except IndexError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/replay/record")
def get_replay_record() -> Response:
    """The loaded replay's record JSON (e.g. to save it to a file)."""
    with _REPLAY_LOCK:
        if _REPLAY is None:
            raise HTTPException(status_code=404, detail="no replay loaded")
        body = _REPLAY.record.to_json()
    return Response(content=body, media_type="application/json")


@app.get("/api/bots")
def get_bots() -> dict[str, dict[str, object]]:
    """Bot kinds available for seats (catan-agents names), each with the
    player counts it supports and its configurable build parameters."""
    return bot_catalog()


class _ChatRequest(BaseModel):
    text: str
    # Seat the message belongs to; None for a spectator.
    player: int | None = None


@app.post("/api/game/chat")
def post_chat(req: _ChatRequest) -> GameModel:
    """Append a chat message to the game log."""
    text = req.text.strip()
    if not text or len(text) > 500:
        raise HTTPException(
            status_code=422, detail="chat text must be 1-500 characters"
        )
    with _LOCK:
        if req.player is not None and not 0 <= req.player < _SESSION.n_players:
            raise HTTPException(status_code=422, detail="no such seat")
        _SESSION.add_chat(req.player, text)
        return _game_model()


class _SeatSpec(BaseModel):
    """A configured bot seat: its kind plus knob overrides from the catalog."""

    kind: str
    params: dict[str, float | int | bool] = {}


class _ResetRequest(BaseModel):
    seed: int = 0
    # Seats in the new game. The engine supports 2-4; the renderer offers 2
    # and 4 for now.
    n_players: Literal[2, 4] = 4
    number_placement: Literal["random", "spiral"] = "random"
    # What controls each seat: "human" (hotseat), a bot kind from
    # GET /api/bots, or a configured bot {"kind", "params"}; no seat has to
    # be human. None seats a human on seat 0 and "random" bots elsewhere.
    seats: list[str | _SeatSpec] | None = None


@app.post("/api/game/reset")
def post_reset(req: _ResetRequest) -> GameModel:
    with _LOCK:
        try:
            _SESSION.reset(
                req.seed,
                n_players=req.n_players,
                number_placement=req.number_placement,
                seats=[s if isinstance(s, str) else s.model_dump() for s in req.seats]
                if req.seats is not None
                else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _game_model()


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
if _dist.exists():
    app.mount("/", _SPAStaticFiles(directory=_dist, html=True), name="static")
