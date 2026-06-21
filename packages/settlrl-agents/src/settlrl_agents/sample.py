"""Determinization: turn a player's honest view into one concrete world.

``sample_world`` is the only road from a
:class:`~settlrl_engine.belief.BeliefView` back to a playable ``BoardState``:
public fields are copied through, every hidden field is filled with a sample
consistent with the belief, and the PRNG key is fresh (the search samples its
own dice / steals / draws instead of foreseeing the environment's). The
proportional-headroom resource deal is a surrogate for the exact posterior,
not the posterior itself.
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, UInt8
from settlrl_engine.belief import BeliefView, ResBoundsVec, ResTotalVec
from settlrl_engine.board.dev_cards import DEV_CARD_COUNTS, N_DEV_CARD_TYPES, DevDeckVec
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import (
    BoardState,
    BoolScalar,
    IntScalar,
    KeyScalar,
    Player,
    to_u8,
)

_DECK_SIZE = sum(DEV_CARD_COUNTS)
# Card-slot view of the deck composition: each of the 25 interchangeable cards
# as (its type, its index within that type).
_CARD_TYPE = jnp.asarray([t for t, c in enumerate(DEV_CARD_COUNTS) for _ in range(c)])
_CARD_RANK = jnp.asarray([r for c in DEV_CARD_COUNTS for r in range(c)])

# Upper bound on cards dealt to one hand (the bank holds 19 of each of the 5
# types, so no hand can exceed this; real hands are far smaller).
_MAX_DEAL = 5 * 19


def _deal_dev_hands(
    key: KeyScalar, view: BeliefView, player: Player
) -> tuple[UInt8[Array, f"players dev_card_types={N_DEV_CARD_TYPES}"], DevDeckVec]:
    """Deal every opponent's dev hand from the unseen pool, uniformly without
    replacement; returns ``(dev_hand, dev_deck)`` with the remainder as the
    deck and the observer's own hand in its row."""
    pool = view.unseen_dev
    need = jnp.where(
        jnp.arange(view.n_players) == player, 0, view.belief.dev_count
    )  # (P,)
    # Noise the pool's card slots; the top slots are taken, opponent by
    # opponent in seat order (exchangeable, so the order doesn't matter).
    in_pool = pool[_CARD_TYPE] > _CARD_RANK
    noise = jnp.where(in_pool, jax.random.uniform(key, (_DECK_SIZE,)), -1.0)
    rank = jnp.argsort(jnp.argsort(-noise))  # rank 0 = highest noise
    owner = jnp.searchsorted(jnp.cumsum(need), rank, side="right")  # (25,)
    taken = rank < need.sum()
    hands = (
        jnp.zeros((view.n_players, N_DEV_CARD_TYPES), jnp.int32)
        .at[jnp.where(taken, owner, 0), _CARD_TYPE]
        .add(taken.astype(jnp.int32))
    )
    dealt = hands.sum(axis=0)
    dev_hand = hands.at[player].add(view.own_dev.astype(jnp.int32))
    return to_u8(dev_hand), to_u8(pool.astype(jnp.int32) - dealt)


_DealCarry: TypeAlias = tuple[IntScalar, ResBoundsVec, ResTotalVec, KeyScalar]


def _deal_resources(key: KeyScalar, view: BeliefView) -> ResBoundsVec:
    """Deal every opponent's unknown resource cards within ``[lo, hi]``.

    Rows start at ``lo`` (the observer's own row is already exact there);
    each player draws ``hand_size - lo.sum()`` more, one at a time, weighted
    by the per-type headroom ``hi - current`` capped by the remaining public
    pool (with the ``hi`` cap relaxed if it ever leaves no choice).
    """
    res = view.belief.res_lo.astype(jnp.int32)  # (P, R), opponents at lo
    pool = view.belief.res_total - res.sum(axis=0)  # (R,) cards left to place
    need = view.belief.hand_size - res.sum(axis=1)  # (P,) 0 where lo is exact

    def deal_player(p: int, carry: _DealCarry) -> _DealCarry:
        hi = view.belief.res_hi[p].astype(jnp.int32)

        def owes(c: _DealCarry) -> BoolScalar:
            return c[0] < jnp.minimum(need[p], _MAX_DEAL)

        def deal_one(c: _DealCarry) -> _DealCarry:
            i, res, pool, key = c
            key, k = jax.random.split(key)
            w = jnp.minimum(jnp.clip(hi - res[p], 0, None), pool)
            w = jnp.where(w.sum() > 0, w, pool)  # infeasible bounds: relax hi
            r = jax.random.categorical(k, jnp.log(w.astype(jnp.float32)))
            r = jnp.clip(r, 0, N_RESOURCES - 1)
            return i + 1, res.at[p, r].add(1), pool.at[r].add(-1), key

        i, res, pool, key = carry
        return jax.lax.while_loop(owes, deal_one, (jnp.zeros_like(i), res, pool, key))

    carry = (jnp.int32(0), res, pool, key)
    for p in range(view.n_players):
        carry = deal_player(p, carry)
    _, res, _, _ = carry
    return res


def sample_world(key: KeyScalar, view: BeliefView, player: Player) -> BoardState:
    """Sample one concrete world from ``player``'s :class:`BeliefView`.

    Public fields are copied through; hand sizes, dev counts, and per-type
    totals all match the public record, so the sample is indistinguishable
    from the truth on everything ``player`` can see.
    """
    k_key, k_dev, k_res = jax.random.split(key, 3)
    dev_hand, dev_deck = _deal_dev_hands(k_dev, view, player)
    resources = _deal_resources(k_res, view)
    pub = view.public
    return BoardState(
        # Public record, copied through.
        vertex_owner=pub.vertex_owner,
        vertex_type=pub.vertex_type,
        edge_road=pub.edge_road,
        robber=pub.robber,
        victory_points=pub.victory_points,
        knights_played=pub.knights_played,
        phase=pub.phase,
        current_player=pub.current_player,
        setup_index=pub.setup_index,
        dice_roll=pub.dice_roll,
        has_rolled=pub.has_rolled,
        dev_played=pub.dev_played,
        free_roads=pub.free_roads,
        pending_discard=pub.pending_discard,
        trade_partner=pub.trade_partner,
        trade_give=pub.trade_give,
        trade_receive=pub.trade_receive,
        longest_road_owner=pub.longest_road_owner,
        largest_army_owner=pub.largest_army_owner,
        longest_road_len=pub.longest_road_len,
        # Hidden fields, sampled (or the observer's own knowledge).
        player_resources=to_u8(resources),
        dev_hand=dev_hand,
        dev_deck=dev_deck,
        dev_bought=view.own_bought,
        key=k_key,
    )
