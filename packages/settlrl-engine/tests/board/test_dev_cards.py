"""Tests for dev_cards.py: the DevCard enum, deck composition, and purchase cost."""

from __future__ import annotations

from settlrl_engine.board.dev_cards import (
    DEV_CARD_COST,
    DEV_CARD_COUNTS,
    N_DEV_CARD_TYPES,
    DevCard,
)


def test_matches_rulebook() -> None:
    # The standard 25-card deck.
    assert len(DEV_CARD_COUNTS) == N_DEV_CARD_TYPES
    assert sum(DEV_CARD_COUNTS) == 25
    assert DEV_CARD_COUNTS[DevCard.KNIGHT] == 14
    # Cost: [sheep, wheat, wood, brick, ore]
    assert DEV_CARD_COST == (1, 1, 0, 0, 1)
