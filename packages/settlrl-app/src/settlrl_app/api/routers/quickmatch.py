"""Quick Match: pair near-rated players into one game, bots filling the rest.

Unlike a hosted lobby (:mod:`settlrl_app.api.routers.lobbies`), a match forms a
full table at once and goes straight to a game — there is no staging step.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from settlrl_app.api.deps import ClientId, Deps, uid
from settlrl_app.storage.db import User


class _MatchRequest(BaseModel):
    n_players: Literal[2, 4] = 2
    # The caller's place in line from a prior poll, re-sent each time; None first.
    ticket: str | None = None


class _MatchQueued(BaseModel):
    """Still finding a game: re-POST with ``ticket`` until matched."""

    queued: Literal[True] = True
    ticket: str
    waiting: int  # players waiting in this bucket (the caller included)


class _MatchFound(BaseModel):
    """Matched: the seat claimed for the caller in a freshly created game."""

    id: str
    seat: int
    token: str


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    CurrentUser = Annotated[User | None, Depends(deps.auth.optional_user)]

    @router.post("/api/matchmake")
    async def matchmake(
        req: _MatchRequest, user: CurrentUser = None, x_client_id: ClientId = None
    ) -> _MatchQueued | _MatchFound:
        """Find an Elo-matched game (bots filling the rest of the table), or return
        the caller's place in line to re-poll. A signed-in caller is matched on
        their rating; anonymous callers match at a fresh player's rating."""
        if deps.matchmaker is None:
            raise HTTPException(status_code=503, detail="matchmaking is unavailable")
        deps.guard_one_game(user, x_client_id)
        match = await deps.matchmaker.matchmake(
            req.n_players, req.ticket, uid(user), x_client_id
        )
        if match.result is not None:
            game_id, seat, token = match.result
            return _MatchFound(id=game_id, seat=seat, token=token)
        return _MatchQueued(ticket=match.ticket, waiting=match.waiting)

    return router
