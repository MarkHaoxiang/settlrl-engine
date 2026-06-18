"""The public Elo leaderboard: accounts and bots ranked per player count."""

from fastapi import APIRouter
from pydantic import BaseModel

from settlrl_app.api.deps import Deps


class _LeaderboardEntry(BaseModel):
    """One ranked subject in a player-count bucket."""

    n_players: int
    kind: str  # "account" | "bot"
    name: str
    rating: float
    games: int
    wins: int


def build(deps: Deps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/leaderboard")
    async def leaderboard() -> list[_LeaderboardEntry]:
        """Every rated subject, ordered by player-count bucket then rating
        (best first). Public; clients split the buckets into ladders."""
        if deps.store is None:
            return []
        return [
            _LeaderboardEntry(
                n_players=e.n_players,
                kind=e.kind,
                name=e.name,
                rating=e.rating,
                games=e.games,
                wins=e.wins,
            )
            for e in await deps.store.leaderboard()
        ]

    return router
