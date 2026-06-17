"""API routers, each built from the shared :class:`~settlrl_render.api.deps.Deps`."""

from settlrl_render.api.routers import bots, games, me, replay

__all__ = ["bots", "games", "me", "replay"]
