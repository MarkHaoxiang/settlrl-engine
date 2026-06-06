from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from .actions import decode_actions
from .convert import board_to_model
from .models import BoardModel, GameModel
from .session import GameSession, IllegalActionError

app = FastAPI(title="Catan Render")

# A single live game vs. random bots (human plays seat 0). One game at a time;
# POST /api/game/reset starts a fresh one.
_SESSION = GameSession()


def _game_model() -> GameModel:
    """The full Play-view snapshot: board + turn status + the human's legal moves."""
    status = _SESSION.status()
    actions = (
        decode_actions([int(f) for f in _SESSION.legal_flat()]) if status.your_turn else []
    )
    return GameModel(board=board_to_model(_SESSION.board), status=status, actions=actions)


@app.get("/api/board")
def get_board() -> BoardModel:
    """Board geometry + player stats (used by the Replay view and shared hook)."""
    return board_to_model(_SESSION.board)


@app.get("/api/game")
def get_game() -> GameModel:
    return _game_model()


class _ActionRequest(BaseModel):
    flat: int


@app.post("/api/game/action")
def post_action(req: _ActionRequest) -> GameModel:
    try:
        _SESSION.apply(req.flat)
    except IllegalActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _game_model()


class _ResetRequest(BaseModel):
    seed: int = 0
    # Seats in the new game (the human + bots). The engine supports 2-4; the
    # renderer offers 2 and 4 for now.
    n_players: Literal[2, 4] = 4


@app.post("/api/game/reset")
def post_reset(req: _ResetRequest) -> GameModel:
    _SESSION.reset(req.seed, n_players=req.n_players)
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
