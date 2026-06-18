"""The bot-service wire protocol, shared by the client (the app's bot registry)
and the server (a bot service).

A service hosts exactly one bot, described by :class:`BotInfo`. A move is
requested by replaying the game incrementally: the request carries the moves the
bot has not seen yet (those after ``base``) as structured :class:`MoveModel`s in
the board's cube/axial coordinates, the service applies them to the game it is
tracking, and returns the chosen move. No engine indices or observation pytrees
cross the wire — only the (stable) record geometry and action shapes.

When the service's tracked game is not at ``base`` (a fresh or restarted
service), ``/act`` answers ``409`` with ``{"resync": true, "have": <n>}`` and the
client re-requests from move ``n``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from settlrl_game.models import CubeModel, EdgeModel, HexModel


class BotInfo(BaseModel):
    """The single bot a service offers.

    ``name`` is its stable id (the seat kind the game stores); ``counts`` are the
    player counts it can play.
    """

    name: str
    title: str
    description: str = ""
    counts: list[int] = [2, 3, 4]


class MoveModel(BaseModel):
    """One action in the board's coordinates. ``type`` is the lowercased action
    type; depending on it, at most one geometry/resource group below is set.

    The same shape is used for the moves replayed to the bot and for the move it
    returns.
    """

    type: str
    # Placement target: a vertex (settlement/city), an edge (road), or a tile
    # (robber / knight, with the optional victim seat to steal from).
    vertex: CubeModel | None = None
    edge: EdgeModel | None = None
    tile: HexModel | None = None
    victim: int | None = None
    # Resource choices: discard / monopoly (one), year-of-plenty (two), maritime
    # trade and the domestic proposal (give/receive plus the proposed-to partner).
    resource: str | None = None
    resources: list[str] | None = None
    give: str | None = None
    receive: str | None = None
    partner: int | None = None


class ActRequest(BaseModel):
    # `setup` + the moves replayed so far fully determine the position; `base` is
    # how many of those moves the bot is assumed to already hold, so only the tail
    # `moves` are sent.
    game_id: str
    seat: int
    setup: dict[str, Any]
    base: int = 0
    moves: list[MoveModel] = []


class ActResponse(BaseModel):
    move: MoveModel
