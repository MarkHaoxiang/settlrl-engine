"""Tests for the Port enum: resource ports mirror Tile, plus the 3:1 general."""

from __future__ import annotations

from settlrl_engine.board.port import Port
from settlrl_engine.board.tile import Tile


def test_resource_ports_mirror_tile_values() -> None:
    assert Port.SHEEP.value == Tile.SHEEP.value
    assert Port.WHEAT.value == Tile.WHEAT.value
    assert Port.WOOD.value == Tile.WOOD.value
    assert Port.BRICK.value == Tile.BRICK.value
    assert Port.ORE.value == Tile.ORE.value
    # The 3:1 general port is the one extra member.
    assert Port.GENERAL.value == 5
    assert len(Port) == 6
