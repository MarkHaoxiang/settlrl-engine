"""The signed-in user's own view: the live games their account owns a seat in."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..db import User
from ..deps import Deps


class _MyGameModel(BaseModel):
    """A live game the signed-in user owns a seat in."""

    id: str
    seats: list[int]


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

    return router
