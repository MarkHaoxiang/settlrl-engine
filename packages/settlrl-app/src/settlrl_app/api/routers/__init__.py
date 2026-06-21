"""API routers, each built from the shared :class:`~settlrl_app.api.deps.Deps`."""

from settlrl_app.api.routers import (
    admin,
    bots,
    games,
    leaderboard,
    lobbies,
    me,
    quickmatch,
    replay,
)

__all__ = [
    "admin",
    "bots",
    "games",
    "leaderboard",
    "lobbies",
    "me",
    "quickmatch",
    "replay",
]
