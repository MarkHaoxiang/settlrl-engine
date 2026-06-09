"""Tests for dev_cards.py: the DevCard enum, deck composition, and purchase cost."""

from __future__ import annotations

from catan_engine.board.dev_cards import (
    DEV_CARD_COST,
    DEV_CARD_COUNTS,
    DevCard,
    N_DEV_CARD_TYPES,
)


def test_matches_rulebook() -> None:
    assert [c.value for c in DevCard] == [0, 1, 2, 3, 4]
    assert len(DevCard) == N_DEV_CARD_TYPES
    # The standard 25-card deck.
    assert len(DEV_CARD_COUNTS) == N_DEV_CARD_TYPES
    assert sum(DEV_CARD_COUNTS) == 25
    assert DEV_CARD_COUNTS[DevCard.KNIGHT] == 14
    # Cost: [sheep, wheat, wood, brick, ore]
    assert DEV_CARD_COST == (1, 1, 0, 0, 1)
