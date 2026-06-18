"""The one-bot SDK: subclass :class:`Bot`, implement :meth:`Bot.act`, serve it.

A bot service hosts exactly one bot. The framework (:mod:`settlrl_agents.service.app`)
tracks every game in flight as a :class:`~settlrl_game.session.GameSession` and, on
the bot's turn, hands it a :class:`GameView` from the acting seat's perspective; the
bot returns the :class:`~settlrl_game.botproto.MoveModel` it wants to play. The
lifecycle hooks (:meth:`Bot.new_game` / :meth:`Bot.end_game`) let a stateful bot keep
per-game memory across turns.

A minimal bot only needs ``name``/``title`` and ``act``::

    class FirstLegal(Bot):
        name, title = "first", "First legal"

        def act(self, view: GameView) -> MoveModel:
            return view.legal[0]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

from settlrl_game.actions import legal_moves
from settlrl_game.botproto import BotInfo, MoveModel
from settlrl_game.convert import board_to_model
from settlrl_game.models import BeliefModel, BoardModel, GameStatusModel
from settlrl_game.session import GameSession, GameSetup

__all__ = ["Bot", "GameView"]


class GameView:
    """What a bot sees on its turn, from the acting seat's perspective.

    ``session`` is the live game (engine-free); advanced bots may read it
    directly. The structured ``board`` / ``legal`` / ``belief`` / ``status`` are
    built on demand for bots that reason over the wire shapes (cube coordinates,
    redaction and card-counting already applied for ``seat``).
    """

    def __init__(self, game_id: str, seat: int, session: GameSession) -> None:
        self.game_id = game_id
        self.seat = seat
        self.session = session

    @cached_property
    def board(self) -> BoardModel:
        return board_to_model(self.session.game)

    @cached_property
    def legal(self) -> list[MoveModel]:
        return legal_moves(self.session.game)

    @cached_property
    def belief(self) -> BeliefModel | None:
        return self.session.belief(self.seat)

    @cached_property
    def status(self) -> GameStatusModel:
        return self.session.status()


class Bot(ABC):
    """A single bot a service hosts. Set the class attributes and implement
    :meth:`act`; the framework does the HTTP, the game tracking, and the move
    translation."""

    name: str
    title: str
    description: str = ""
    counts: list[int] = [2, 3, 4]  # noqa: RUF012  (an overridable per-bot default)

    def info(self) -> BotInfo:
        return BotInfo(
            name=self.name,
            title=self.title,
            description=self.description,
            counts=list(self.counts),
        )

    def new_game(self, game_id: str, setup: GameSetup, seat: int) -> None:  # noqa: B027
        """Called the first time the service is asked to act in ``game_id`` (for
        a stateful bot to allocate per-game memory). Default: nothing."""

    def end_game(self, game_id: str) -> None:  # noqa: B027
        """Called when a tracked game is dropped (finished or evicted). Default:
        nothing."""

    @abstractmethod
    def act(self, view: GameView) -> MoveModel:
        """The move to play for ``view.seat`` — one of ``view.legal``."""
