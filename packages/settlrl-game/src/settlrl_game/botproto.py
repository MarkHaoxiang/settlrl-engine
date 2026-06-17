"""The bot-service wire protocol, shared by the client (the app's provider
registry) and the server (the agents' bot service).

A move is requested by sending the game's setup plus its flat move list so far —
the same data a record carries — and the service replays them and returns the
chosen flat action. No engine observation crosses the wire, so the two sides
only have to agree on the (stable) record format and the flat action indexing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ActRequest(BaseModel):
    # `game_id` only keys the service's replay cache; `setup` + `moves` fully
    # determine the position.
    game_id: str
    setup: dict[str, Any]
    moves: list[int]
    seat: int


class ActResponse(BaseModel):
    flat: int
