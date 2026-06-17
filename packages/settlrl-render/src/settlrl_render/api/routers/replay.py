"""Replay tooling routes (``/api/replay*``) and the shared replay loader.

One replay is loaded server-wide at a time (the :class:`~settlrl_render.api.deps.
ReplaySlot`); :func:`load_replay` builds it — offloading the engine replay to a
worker thread — and is reused by the game router's replay-from-a-finished-game
route.
"""

import json

import anyio.to_thread
from fastapi import APIRouter, HTTPException
from settlrl_engine.record import GameRecord, ReplayError
from starlette.responses import Response

from settlrl_render.api.deps import Deps, ReplaySlot
from settlrl_render.api.models import ReplayStateModel
from settlrl_render.game.replay import ReplaySession

# Replaying a submitted record steps the engine once per move; cap the count so
# an untrusted POST /api/replay can't hand us an arbitrarily long game to grind
# through. Well above any real game (random games run a few thousand moves).
_MAX_REPLAY_MOVES = 20_000


async def load_replay(replays: ReplaySlot, record: GameRecord) -> ReplayStateModel:
    """Replay ``record`` into the shared slot and return its opening state."""
    try:
        session = await anyio.to_thread.run_sync(ReplaySession, record)
    except (ReplayError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with replays.lock:
        replays.session = session
        return session.state(0)


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    replays = deps.replays

    @router.post("/api/replay")
    async def post_replay(doc: dict[str, object]) -> ReplayStateModel:
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
        return await load_replay(replays, record)

    @router.get("/api/replay/state")
    async def get_replay_state(move: int = 0) -> ReplayStateModel:
        """The loaded replay after ``move`` moves (0 = the opening board).

        ``404`` when no replay is loaded; ``422`` when ``move`` is out of
        range.
        """
        async with replays.lock:
            if replays.session is None:
                raise HTTPException(status_code=404, detail="no replay loaded")
            try:
                return replays.session.state(move)
            except IndexError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/replay/record")
    async def get_replay_record() -> Response:
        """The loaded replay's record JSON (e.g. to save it to a file)."""
        async with replays.lock:
            if replays.session is None:
                raise HTTPException(status_code=404, detail="no replay loaded")
            body = replays.session.record.to_json()
        return Response(content=body, media_type="application/json")

    return router
