"""Root determinization: sample a world consistent with what the player knows.

The only hidden state in a two-player game is dev-card identities. The unseen
pool — the full deck composition minus the player's own hand and every card
seen played — equals ``dev_deck + opponent's hand`` by conservation, so it is
computable without reading anything private. Re-dealing the opponent's hand
from that pool (and leaving the rest as the deck) replaces the true hidden
identities with an honest posterior sample.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.board.dev_cards import DEV_CARD_COUNTS, N_DEV_CARD_TYPES
from catan_engine.board.state import BoardState, IntScalar

_DECK_SIZE = sum(DEV_CARD_COUNTS)
# Card-slot view of the deck composition: each of the 25 interchangeable cards
# as (its type, its index within that type).
_CARD_TYPE = jnp.asarray([t for t, c in enumerate(DEV_CARD_COUNTS) for _ in range(c)])
_CARD_RANK = jnp.asarray([r for c in DEV_CARD_COUNTS for r in range(c)])


def redeal_dev_cards(key: jax.Array, state: BoardState, player: IntScalar) -> BoardState:
    """Resample the opponent's dev hand from ``player``'s unseen pool (one
    two-player game); the pool's remainder becomes the deck. Hand sizes, and
    therefore everything public, are unchanged."""
    opponent = 1 - player
    pool = state.dev_deck + state.dev_hand[opponent]
    n_held = state.dev_hand[opponent].astype(jnp.int32).sum()
    # Draw n_held cards uniformly without replacement: noise the pool's card
    # slots and keep the top n_held (the pool contains the hand, so it always
    # has enough).
    in_pool = _CARD_RANK < pool[_CARD_TYPE]
    noise = jnp.where(in_pool, jax.random.uniform(key, (_DECK_SIZE,)), -1.0)
    order = jnp.argsort(-noise)
    taken = jnp.zeros((_DECK_SIZE,), bool).at[order].set(jnp.arange(_DECK_SIZE) < n_held)
    hand = (
        jnp.zeros((N_DEV_CARD_TYPES,), jnp.uint8)
        .at[_CARD_TYPE]
        .add(taken.astype(jnp.uint8))
    )
    return state._replace(
        dev_hand=state.dev_hand.at[opponent].set(hand),
        dev_deck=pool - hand,
    )
