"""Resolve the stochastic outcomes a driver must inject into actions.

The game itself takes realised outcomes on its actions (``Roll.value``, the
drawn ``BuyDevelopmentCard.card``, the robber's ``stolen`` card) so a
differential test can feed it the engine's. A live driver (the renderer)
samples them instead; these helpers do that from an RNG. Each assumes the
matching action is legal (the deck has a card, the victim has a hand).
"""

from __future__ import annotations

from random import Random

from settlrl_reference.game import Game
from settlrl_reference.types import DevCard, Resource


def roll_dice(rng: Random) -> int:
    """A two-dice roll, 2..12."""
    return rng.randint(1, 6) + rng.randint(1, 6)


def draw_dev_card(game: Game, rng: Random) -> DevCard:
    """A uniformly random card from the remaining development deck."""
    deck = [card for card, n in game.dev_deck.items() for _ in range(n)]
    return rng.choice(deck)


def steal(game: Game, victim: int, rng: Random) -> Resource:
    """A uniformly random card from the victim's hand."""
    hand = [r for r, n in game.players[victim].resources.items() for _ in range(n)]
    return rng.choice(hand)
