"""Tests for the Tile enum: values and labels."""

from __future__ import annotations

from catan_engine.tile import Tile


def test_values_are_zero_through_five() -> None:
    assert [t.value for t in Tile] == [0, 1, 2, 3, 4, 5]
    assert Tile.DESERT.value == 5  # the one non-resource tile


def test_labels() -> None:
    assert [str(t) for t in Tile] == ["SHP", "WHT", "WOD", "BRK", "ORE", "DST"]
