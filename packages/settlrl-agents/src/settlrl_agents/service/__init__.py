"""The bot service: subclass :class:`Bot`, implement :meth:`Bot.act`, and serve it
with :func:`create_app` behind the bot wire protocol
(:mod:`settlrl_game.botproto`). Optional ``[service]`` extra (FastAPI).
"""

from settlrl_agents.service.app import create_app
from settlrl_agents.service.sdk import Bot, GameView

__all__ = ["Bot", "GameView", "create_app"]
