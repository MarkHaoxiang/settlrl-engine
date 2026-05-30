"""Development-card rules: playability and weighted draws (single-game, traceable).

Separate from ``dev_cards.py`` (which holds the ``DevCard`` enum and deck counts)
because these rules operate on ``BoardState`` and ``state`` already imports
``dev_cards`` -- colocating would create an import cycle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.dev_cards import N_DEV_CARD_TYPES
from catan_engine.state import BoardState


def playable_dev(state: BoardState, player: jax.Array, card: int) -> jax.Array:
    """True if ``player`` holds a playable copy of ``card`` (not bought this turn)."""
    held = state.dev_hand[player, card].astype(jnp.int32)
    bought = state.dev_bought[card].astype(jnp.int32)
    return held - bought > 0


def draw_dev_card(key: jax.Array, dev_deck: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Draw one card type from ``dev_deck`` weighted by remaining counts.

    Returns ``(advanced key, card index)``. The probabilities fall back to
    uniform when the deck is empty so the draw is always well defined under a
    trace; callers gate the actual application on deck availability.
    """
    deck = dev_deck.astype(jnp.float32)
    total = deck.sum()
    probs = jnp.where(
        total > 0,
        deck / jnp.maximum(total, 1.0),
        jnp.full((N_DEV_CARD_TYPES,), 1.0 / N_DEV_CARD_TYPES),
    )
    key, sub = jax.random.split(key)
    card = jax.random.choice(sub, N_DEV_CARD_TYPES, p=probs)
    return key, card.astype(jnp.int32)
