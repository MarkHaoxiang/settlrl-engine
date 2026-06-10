"""Shared fixtures/helpers for the catan-reference unit tests.

These tests hand-craft small ``Game`` / ``Layout`` states to exercise subtle rule
branches in isolation (the kind random differential play under-exercises). The
board *geometry* is fixed (generated in ``board.py``); only the ``Layout`` (per-
tile resource + number token, and the ports) and the occupancy on a ``Game``
vary, so the helpers below build just enough of those by hand.
"""

from __future__ import annotations

from catan_reference import board
from catan_reference.board import Layout, Port
from catan_reference.game import Game, Player
from catan_reference.types import (
    N_PLAYERS,
    Building,
    Phase,
    PortType,
    Resource,
)


def make_layout(
    tile_resource: dict[int, Resource | None] | None = None,
    tile_number: dict[int, int] | None = None,
    ports: tuple[Port, ...] = (),
) -> Layout:
    """A ``Layout`` of all-desert tiles, overridden per-tile by the given maps.

    Defaults keep every tile a desert (resource ``None``, number ``0``) so a test
    only has to name the few tiles it cares about.
    """
    tile_resource = tile_resource or {}
    tile_number = tile_number or {}
    return Layout(
        tile_resource=tuple(tile_resource.get(t, None) for t in range(board.N_TILES)),
        tile_number=tuple(tile_number.get(t, 0) for t in range(board.N_TILES)),
        ports=ports,
    )


def make_game(layout: Layout, robber: int = 18, phase: Phase = Phase.MAIN) -> Game:
    """A fresh ``Game`` on ``layout`` with empty boards, in ``phase``.

    ``robber`` defaults to a tile far from the ones the production tests use so it
    never accidentally blocks them.
    """
    return Game(
        layout=layout,
        robber=robber,
        players=[Player() for _ in range(N_PLAYERS)],
        phase=phase,
        has_rolled=True,
    )


def set_resource(game: Game, player: int, resource: Resource, amount: int) -> None:
    game.players[player].resources[resource] = amount


def place(game: Game, vertex: int, player: int, kind: Building) -> None:
    game.buildings[vertex] = (player, kind)


def place_road_path(game: Game, vertices: list[int], player: int) -> list[int]:
    """Lay ``player`` roads along the consecutive-vertex path; return the edges."""
    edges = [
        board.edge_between(vertices[i], vertices[i + 1])
        for i in range(len(vertices) - 1)
    ]
    for e in edges:
        game.roads[e] = player
    return edges


__all__ = [
    "PortType",
    "make_game",
    "make_layout",
    "place",
    "place_road_path",
    "set_resource",
]
