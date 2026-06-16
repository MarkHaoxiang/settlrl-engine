"""Tests for the Tile enum: the values resource arrays are indexed by."""

from __future__ import annotations

from settlrl_engine.board.tile import Tile


def test_values_are_zero_through_five() -> None:
    assert [t.value for t in Tile] == [0, 1, 2, 3, 4, 5]
    assert Tile.DESERT.value == 5  # the one non-resource tile
