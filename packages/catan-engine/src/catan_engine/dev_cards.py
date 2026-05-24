from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, UInt8

from catan_engine.resources import N_PLAYERS, N_RESOURCES

N_DEV_CARD_TYPES = 5

# Remaining counts per type in a full, undrawn deck.
# Order matches DevCard enum values.
DEV_CARD_COUNTS: tuple[int, ...] = (14, 2, 2, 2, 5)  # sums to 25
N_DEV_CARDS: int = sum(DEV_CARD_COUNTS)  # 25

# Cost to purchase one development card: [sheep, wheat, wood, brick, ore]
# Indices match Tile enum (SHEEP=0, WHEAT=1, WOOD=2, BRICK=3, ORE=4).
DEV_CARD_COST: tuple[int, ...] = (1, 1, 0, 0, 1)

DevCardDeckArray = UInt8[Array, f"batch dev_card_types={N_DEV_CARD_TYPES}"]
PlayerDevCardHandArray = UInt8[
    Array, f"batch players={N_PLAYERS} dev_card_types={N_DEV_CARD_TYPES}"
]
PlayerPlayedKnightsArray = UInt8[Array, f"batch players={N_PLAYERS}"]


class DevCard(IntEnum):
    KNIGHT = 0
    ROAD_BUILDING = 1
    YEAR_OF_PLENTY = 2
    MONOPOLY = 3
    VICTORY_POINT = 4

    def __str__(self) -> str:
        return ("KNT", "RDB", "YOP", "MNP", "VPT")[self]


class DevCardState(NamedTuple):
    """Development card state for a batch of games.

    deck:           remaining count of each card type in the draw pile.
    player_hand:    cards held (not yet played) per player per type.
    knights_played: cumulative knights played per player (for Largest Army).
    """

    deck: jax.Array          # (batch, N_DEV_CARD_TYPES)
    player_hand: jax.Array   # (batch, N_PLAYERS, N_DEV_CARD_TYPES)
    knights_played: jax.Array  # (batch, N_PLAYERS)


def make_dev_card_state(batch_size: int = 1) -> DevCardState:
    B = batch_size
    deck_counts = jnp.array(DEV_CARD_COUNTS, dtype=jnp.uint8)
    return DevCardState(
        deck=jnp.broadcast_to(deck_counts, (B, N_DEV_CARD_TYPES)).copy(),
        player_hand=jnp.zeros((B, N_PLAYERS, N_DEV_CARD_TYPES), dtype=jnp.uint8),
        knights_played=jnp.zeros((B, N_PLAYERS), dtype=jnp.uint8),
    )
