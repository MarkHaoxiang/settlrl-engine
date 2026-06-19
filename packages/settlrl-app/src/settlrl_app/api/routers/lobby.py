"""The public game lobby: open games anyone can join, plus Elo Quick Match."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from settlrl_app.api.deps import Deps, uid
from settlrl_app.storage.db import User


class _LobbyGameModel(BaseModel):
    """One joinable game in the lobby."""

    id: str
    n_players: int
    number_placement: str
    seats: list[str]  # seat kinds; "human" seats are the open ones
    claimed: list[int]  # seats already taken
    open_seats: int  # unclaimed human seats a joiner can take
    created_at: float


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

    @router.get("/api/lobby")
    def lobby() -> list[_LobbyGameModel]:
        """Listed, joinable games (newest first) — each has an open human seat."""
        return [
            _LobbyGameModel(
                id=h.id,
                n_players=h.session.n_players,
                number_placement=h.session.number_placement,
                seats=list(h.session.seats),
                claimed=sorted(h.claims),
                open_seats=len(h.open_human_seats()),
                created_at=h.created_at,
            )
            for h in deps.registry.open_games()
        ]

    @router.post("/api/matchmake")
    async def matchmake(
        req: _MatchRequest, user: CurrentUser = None
    ) -> _MatchQueued | _MatchFound:
        """Find an Elo-matched game (bots filling the rest of the table), or return
        the caller's place in line to re-poll. A signed-in caller is matched on
        their rating; anonymous callers match at a fresh player's rating."""
        if deps.matchmaker is None:
            raise HTTPException(status_code=503, detail="matchmaking is unavailable")
        match = await deps.matchmaker.matchmake(req.n_players, req.ticket, uid(user))
        if match.result is not None:
            game_id, seat, token = match.result
            return _MatchFound(id=game_id, seat=seat, token=token)
        return _MatchQueued(ticket=match.ticket, waiting=match.waiting)

    return router
