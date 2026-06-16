"""Production under a depleted bank (rulebook p.4 / Almanac).

The branch under test (``Game.production``): for each resource, if the total
demanded this roll exceeds the bank's remaining stock, then a *single* claimant
still receives ``min(demand, stock)``, but when *two or more* players each claim
a share of the depleted resource, **nobody** receives any of it.
"""

from __future__ import annotations

from conftest import make_game, make_layout, place, set_resource
from settlrl_reference.types import BANK_INITIAL, Building, Resource


def test_single_claimant_capped_at_remaining_stock() -> None:
    # Tile 0 produces WOOD on a roll of 6. Player 0 has a city (2 cards) there.
    layout = make_layout({0: Resource.WOOD}, {0: 6})
    game = make_game(layout)
    place(game, vertex=0, player=0, kind=Building.CITY)  # corner of tile 0

    # Drain the bank so only 1 WOOD remains, while the city demands 2.
    set_resource(game, player=1, resource=Resource.WOOD, amount=BANK_INITIAL - 1)
    assert game.bank(Resource.WOOD) == 1

    granted = game.production(6)
    # Single claimant -> capped at the remaining stock, not zeroed.
    assert granted[0][Resource.WOOD] == 1


def test_multiple_claimants_on_depleted_resource_get_nothing() -> None:
    # Tile 0 produces WOOD on 6; players 0 and 1 each own a settlement on it,
    # so each demands 1 (total demand 2).
    layout = make_layout({0: Resource.WOOD}, {0: 6})
    game = make_game(layout)
    place(game, vertex=0, player=0, kind=Building.SETTLEMENT)
    place(game, vertex=3, player=1, kind=Building.SETTLEMENT)  # non-adjacent corner

    # Leave only 1 WOOD in the bank: total demand 2 > stock 1, and two claimants.
    set_resource(game, player=2, resource=Resource.WOOD, amount=BANK_INITIAL - 1)
    assert game.bank(Resource.WOOD) == 1

    granted = game.production(6)
    # Tie on a depleted resource -> nobody receives any.
    assert granted[0][Resource.WOOD] == 0
    assert granted[1][Resource.WOOD] == 0


def test_sufficient_stock_pays_everyone_even_when_multiple_claim() -> None:
    # Control: with enough stock both claimants are paid normally.
    layout = make_layout({0: Resource.WOOD}, {0: 6})
    game = make_game(layout)
    place(game, vertex=0, player=0, kind=Building.SETTLEMENT)
    place(game, vertex=3, player=1, kind=Building.SETTLEMENT)
    assert game.bank(Resource.WOOD) == BANK_INITIAL

    granted = game.production(6)
    assert granted[0][Resource.WOOD] == 1
    assert granted[1][Resource.WOOD] == 1
