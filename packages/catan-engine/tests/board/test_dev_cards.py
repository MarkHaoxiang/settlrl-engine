"""Tests for dev_cards.py: the DevCard enum, deck composition, and purchase cost."""

from __future__ import annotations

from catan_engine.board.dev_cards import (
    DEV_CARD_COST,
    DEV_CARD_COUNTS,
    DevCard,
    N_DEV_CARD_TYPES,
)


def test_enum_covers_every_type() -> None:
    assert [c.value for c in DevCard] == [0, 1, 2, 3, 4]
    assert len(DevCard) == N_DEV_CARD_TYPES


def test_labels() -> None:
    assert [str(c) for c in DevCard] == ["KNT", "RDB", "YOP", "MNP", "VPT"]


def test_deck_is_a_standard_25_card_deck() -> None:
    assert len(DEV_CARD_COUNTS) == N_DEV_CARD_TYPES
    assert sum(DEV_CARD_COUNTS) == 25
    assert DEV_CARD_COUNTS[DevCard.KNIGHT] == 14


def test_cost_is_sheep_wheat_ore() -> None:
    # [sheep, wheat, wood, brick, ore]
    assert DEV_CARD_COST == (1, 1, 0, 0, 1)
