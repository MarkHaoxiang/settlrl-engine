"""API routers, each built from the shared :class:`~settlrl_render.deps.Deps`."""

from . import bots, games, me, replay

__all__ = ["bots", "games", "me", "replay"]
