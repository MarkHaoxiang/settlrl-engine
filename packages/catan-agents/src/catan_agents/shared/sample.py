"""Determinization: turn a censored state + belief into one concrete world.

``sample_world`` is the only road from a player's honest view (see
``catan_engine.belief``) back to a playable ``BoardState``: every field
``censor`` removed is filled with a sample consistent with the belief —
opponents' dev hands dealt from the unseen pool, their resource hands dealt
within the proven ``[lo, hi]`` bounds to their public sizes against the public
per-type pool, and a fresh PRNG key (the search samples its own dice / steals
/ draws instead of foreseeing the environment's). Model-based agents search in
the sample; nothing hidden can reach them because nothing hidden was there to
begin with.

The resource deal draws cards one at a time proportionally to the remaining
per-type headroom — a reasonable surrogate for the exact posterior, not the
posterior itself (it ignores *how* the bounds arose).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.belief import PlayerBelief
from catan_engine.board.dev_cards import DEV_CARD_COUNTS, N_DEV_CARD_TYPES
from catan_engine.board.resources import N_RESOURCES
from catan_engine.board.state import BoardState, IntScalar, to_u8

_DECK_SIZE = sum(DEV_CARD_COUNTS)
# Card-slot view of the deck composition: each of the 25 interchangeable cards
# as (its type, its index within that type).
_CARD_TYPE = jnp.asarray([t for t, c in enumerate(DEV_CARD_COUNTS) for _ in range(c)])
_CARD_RANK = jnp.asarray([r for c in DEV_CARD_COUNTS for r in range(c)])

# Upper bound on cards dealt to one hand (the bank holds 19 of each of the 5
# types, so no hand can exceed this; real hands are far smaller).
_MAX_DEAL = 5 * 19


def _deal_dev_hands(
    key: jax.Array, state: BoardState, belief: PlayerBelief, player: IntScalar
) -> BoardState:
    """Deal every opponent's dev hand from the censored deck (the unseen pool),
    uniformly without replacement; the remainder becomes the deck."""
    pool = state.dev_deck
    need = jnp.where(
        jnp.arange(state.n_players) == player, 0, belief.dev_count
    )  # (P,)
    # Noise the pool's card slots; the top slots are taken, opponent by
    # opponent in seat order (exchangeable, so the order doesn't matter).
    in_pool = _CARD_RANK < pool[_CARD_TYPE]
    noise = jnp.where(in_pool, jax.random.uniform(key, (_DECK_SIZE,)), -1.0)
    rank = jnp.argsort(jnp.argsort(-noise))  # rank 0 = highest noise
    owner = jnp.searchsorted(jnp.cumsum(need), rank, side="right")  # (25,)
    taken = rank < need.sum()
    hands = (
        jnp.zeros((state.n_players, N_DEV_CARD_TYPES), jnp.int32)
        .at[jnp.where(taken, owner, 0), _CARD_TYPE]
        .add(taken.astype(jnp.int32))
    )
    dealt = hands.sum(axis=0)
    new_hand = state.dev_hand.astype(jnp.int32) + hands
    return state._replace(
        dev_hand=to_u8(new_hand),
        dev_deck=to_u8(pool.astype(jnp.int32) - dealt),
    )


def _deal_resources(
    key: jax.Array, state: BoardState, belief: PlayerBelief, player: IntScalar
) -> BoardState:
    """Deal every opponent's unknown resource cards within ``[lo, hi]``.

    The censored rows already sit at ``lo``; each opponent draws
    ``hand_size - lo.sum()`` more, one at a time, weighted by the per-type
    headroom ``hi - current`` capped by the remaining public pool (with the
    ``hi`` cap relaxed if it ever leaves no choice).
    """
    res = state.player_resources.astype(jnp.int32)  # (P, R), opponents at lo
    pool = belief.res_total - res.sum(axis=0)  # (R,) cards left to place
    need = belief.hand_size - res.sum(axis=1)  # (P,) 0 for `player`

    def deal_player(
        p: int, carry: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        res, pool, key = carry
        hi = belief.res_hi[p].astype(jnp.int32)

        def deal_one(
            i: jax.Array, inner: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            res, pool, key = inner
            key, k = jax.random.split(key)
            active = i < need[p]
            w = jnp.minimum(jnp.clip(hi - res[p], 0, None), pool)
            w = jnp.where(w.sum() > 0, w, pool)  # infeasible bounds: relax hi
            r = jax.random.categorical(k, jnp.log(w.astype(jnp.float32)))
            r = jnp.clip(r, 0, N_RESOURCES - 1)
            add = active.astype(jnp.int32)
            return res.at[p, r].add(add), pool.at[r].add(-add), key

        return cast(
            "tuple[jax.Array, jax.Array, jax.Array]",
            jax.lax.fori_loop(0, _MAX_DEAL, deal_one, (res, pool, key)),
        )

    carry = (res, pool, key)
    for p in range(state.n_players):
        carry = deal_player(p, carry)
    res, _, _ = carry
    return state._replace(player_resources=to_u8(res))


def sample_world(
    key: jax.Array, state: BoardState, belief: PlayerBelief, player: IntScalar
) -> BoardState:
    """Sample one concrete world from ``player``'s censored ``state`` + belief.

    Public fields are untouched; hand sizes, dev counts, and per-type totals
    all match the public record, so the sample is indistinguishable from the
    truth on everything ``player`` can see.
    """
    k_key, k_dev, k_res = jax.random.split(key, 3)
    state = state._replace(key=k_key)
    state = _deal_dev_hands(k_dev, state, belief, player)
    return _deal_resources(k_res, state, belief, player)
