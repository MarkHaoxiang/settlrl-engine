"""The public game lobby: open games anyone can join (Elo quick-match follows)."""

from fastapi import APIRouter
from pydantic import BaseModel

from settlrl_app.api.deps import Deps


class _LobbyGameModel(BaseModel):
    """One joinable game in the lobby."""

    id: str
    n_players: int
    number_placement: str
    seats: list[str]  # seat kinds; "human" seats are the open ones
    claimed: list[int]  # seats already taken
    open_seats: int  # unclaimed human seats a joiner can take
    created_at: float


def build(deps: Deps) -> APIRouter:
    router = APIRouter()

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

    return router
