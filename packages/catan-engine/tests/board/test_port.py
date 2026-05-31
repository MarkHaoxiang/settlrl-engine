"""Tests for the Port enum: resource ports mirror Tile, plus the 3:1 general."""

from __future__ import annotations

from catan_engine.board.port import Port
from catan_engine.board.tile import Tile


def test_resource_ports_mirror_tile_values() -> None:
    assert Port.SHEEP.value == Tile.SHEEP.value
    assert Port.WHEAT.value == Tile.WHEAT.value
    assert Port.WOOD.value == Tile.WOOD.value
    assert Port.BRICK.value == Tile.BRICK.value
    assert Port.ORE.value == Tile.ORE.value


def test_general_is_the_extra_member() -> None:
    assert Port.GENERAL.value == 5
    assert len(Port) == 6


def test_labels() -> None:
    assert [str(p) for p in Port] == ["SHP", "WHT", "WOD", "BRK", "ORE", "3:1"]
