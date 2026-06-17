"""API routers, each built from the shared :class:`~settlrl_app.api.deps.Deps`."""

from settlrl_app.api.routers import bots, games, me, replay

__all__ = ["bots", "games", "me", "replay"]
