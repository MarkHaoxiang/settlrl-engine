"""The signed-in user's own view: the live games their account owns a seat in,
and the finished games kept in their history."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from settlrl_app.api.deps import Deps
from settlrl_app.storage.db import User


class _MyGameModel(BaseModel):
    """A live game the signed-in user owns a seat in."""

    id: str
    seats: list[int]


class _PastGameModel(BaseModel):
    """A finished game in the user's history."""

    id: str
    seats: list[int]
    n_players: int
    winner: int | None
    finished_at: float


def build(deps: Deps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/me/games")
    def my_games(
        user: Annotated[User, Depends(deps.auth.current_user)],
    ) -> list[_MyGameModel]:
        """The signed-in user's games — those still live where their account
        owns a seat — so they can resume on any device without a seat token."""
        return [
            _MyGameModel(id=handle.id, seats=seats)
            for handle in deps.registry.all_handles()
            if (seats := handle.seats_for_user(str(user.id)))
        ]

    @router.get("/api/me/history")
    async def my_history(
        user: Annotated[User, Depends(deps.auth.current_user)],
    ) -> list[_PastGameModel]:
        """The signed-in user's finished games (newest first) — their seats, the
        winner, and the player count — each replayable / downloadable by id."""
        if deps.store is None:
            return []
        uid = str(user.id)
        return [
            _PastGameModel(
                id=game.id,
                seats=sorted(game.owners[uid]),
                n_players=cast(int, game.header.get("n_players", 0)),
                winner=game.winner,
                finished_at=game.finished_at,
            )
            for game in await deps.store.history()
            if uid in game.owners
        ]

    return router
