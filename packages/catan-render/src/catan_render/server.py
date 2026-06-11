import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from catan_engine.record import GameRecord, ReplayError
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from .actions import decode_actions
from .bots import bot_catalog
from .convert import board_to_model
from .games import GameHandle, GameRegistry
from .models import BotMoveModel, GameModel, ReplayStateModel
from .replay import ReplaySession
from .session import GameSession, IllegalActionError


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
    yield


app = FastAPI(title="Catan Render", lifespan=_lifespan)

# The live games, addressed by id. Claiming a human seat (create / join)
# issues a token; privileged requests prove seat ownership by presenting
# their tokens in the X-Seat-Tokens header (comma-separated — a hotseat
# client holds one per local seat).
_GAMES = GameRegistry()

SeatTokens = Annotated[str | None, Header(alias="X-Seat-Tokens")]


def _tokens(header: str | None) -> list[str]:
    return [t.strip() for t in (header or "").split(",") if t.strip()]


def _handle(game_id: str) -> GameHandle:
    handle = _GAMES.get(game_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="no such game")
    return handle


def _game_model(
    handle: GameHandle, owned: set[int], bot_move: BotMoveModel | None = None
) -> GameModel:
    """The snapshot as one requester sees it.

    ``owned`` is the requester's proven seats: it decides ``your_turn``, which
    legal actions ship, whose hands stay unredacted, and the belief observer.
    Spectators (no seats) get the public view: counts, board, log — no hands.
    """
    session = handle.session
    status = session.status()
    status.your_turn = (not status.terminal) and status.acting_player in owned
    actions = (
        decode_actions([int(f) for f in session.legal_flat()])
        if status.your_turn
        else []
    )
    observer = (
        status.acting_player
        if status.acting_player in owned
        else min(owned)
        if owned
        else None
    )
    board = board_to_model(session.board)
    for player in board.players:
        if player.player not in owned:
            player.resources = None
            player.dev_card_types = None
    return GameModel(
        id=handle.id,
        board=board,
        status=status,
        actions=actions,
        bot_move=bot_move,
        log=session.log(),
        belief=session.belief(observer) if observer is not None else None,
        seats_claimed=sorted(handle.claims),
    )


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
    # hotseat default), or none (others join via POST /join).
    claim: Literal["all", "none"] = "all"


class _CreatedModel(BaseModel):
    """A fresh game: its id and the creator's seat tokens."""

    id: str
    seats: list[str]
    tokens: dict[int, str]


@app.post("/api/games")
def post_create(req: _CreateRequest) -> _CreatedModel:
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
    handle = _GAMES.create(session)
    tokens = (
        dict(handle.claim(seat) for seat in handle.human_seats())
        if req.claim == "all"
        else {}
    )
    return _CreatedModel(id=handle.id, seats=session.seats, tokens=tokens)


class _JoinRequest(BaseModel):
    # A specific human seat, or None for the first unclaimed one.
    seat: int | None = None


class _JoinedModel(BaseModel):
    id: str
    seat: int
    token: str


@app.post("/api/games/{game_id}/join")
def post_join(game_id: str, req: _JoinRequest) -> _JoinedModel:
    handle = _handle(game_id)
    with handle.lock:
        try:
            seat, token = handle.claim(req.seat)
        except LookupError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _JoinedModel(id=game_id, seat=seat, token=token)


@app.get("/api/games/{game_id}")
def get_game(game_id: str, x_seat_tokens: SeatTokens = None) -> GameModel:
    handle = _handle(game_id)
    with handle.lock:
        return _game_model(handle, handle.owned_seats(_tokens(x_seat_tokens)))


class _ActionRequest(BaseModel):
    flat: int


@app.post("/api/games/{game_id}/action")
def post_action(
    game_id: str, req: _ActionRequest, x_seat_tokens: SeatTokens = None
) -> GameModel:
    """Apply the acting seat's move; the request must prove it owns that seat."""
    handle = _handle(game_id)
    with handle.lock:
        owned = handle.owned_seats(_tokens(x_seat_tokens))
        if handle.session.acting_seat() not in owned:
            raise HTTPException(status_code=403, detail="not your seat")
        try:
            handle.session.apply(req.flat)
        except IllegalActionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _game_model(handle, owned)


@app.post("/api/games/{game_id}/bot")
def post_bot_step(game_id: str, x_seat_tokens: SeatTokens = None) -> GameModel:
    """Play one due bot move; the snapshot plus what was played (``bot_move``).

    ``bot_move`` is null when no bot move is due (a human seat is acting or the
    game is over) -- clients poll this until ``status.your_turn``.
    """
    handle = _handle(game_id)
    with handle.lock:
        seat = handle.session.acting_seat()
        flat = handle.session.bot_step()
        move = (
            None
            if flat is None
            else BotMoveModel(player=seat, action=decode_actions([flat])[0])
        )
        return _game_model(
            handle, handle.owned_seats(_tokens(x_seat_tokens)), bot_move=move
        )


class _ChatRequest(BaseModel):
    text: str
    # Seat the message belongs to (must be owned); None for a spectator.
    player: int | None = None


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
    handle = _handle(game_id)
    with handle.lock:
        owned = handle.owned_seats(_tokens(x_seat_tokens))
        if req.player is not None and req.player not in owned:
            raise HTTPException(status_code=403, detail="not your seat")
        handle.session.add_chat(req.player, text)
        return _game_model(handle, owned)


@app.get("/api/games/{game_id}/record")
def get_record(game_id: str) -> Response:
    """The finished game as ``catan_engine.record`` JSON -- a self-contained,
    replayable transcript. 409 while the game is running: replaying a record
    reconstructs hidden hands, so live games don't export."""
    handle = _handle(game_id)
    with handle.lock:
        if not handle.session.terminal():
            raise HTTPException(status_code=409, detail="game still running")
        body = handle.session.record().to_json()
    return Response(content=body, media_type="application/json")


# The loaded replay, if any (server-wide tooling; one at a time).
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


@app.post("/api/games/{game_id}/replay")
def post_replay_from_game(game_id: str) -> ReplayStateModel:
    """Load a finished game for replay (409 while it is still running:
    replaying reconstructs hidden hands)."""
    handle = _handle(game_id)
    with handle.lock:
        if not handle.session.terminal():
            raise HTTPException(status_code=409, detail="game still running")
        record = handle.session.record()
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
