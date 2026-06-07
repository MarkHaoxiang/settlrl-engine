"""Card counting: what each player can legitimately know about hidden hands.

An optional companion to ``BoardState`` (see ``BatchedCatanEnv(track_beliefs=
True)``): a ``BeliefState`` carries, for every observer, proven per-type bounds
on every player's resource hand, plus the public count of played development
cards. Everything in it is derivable from public information, so handing it to
an agent never leaks.

What is hidden in this engine's Catan: the *type* of a card moved by a robber
steal (third parties see only that a card moved; thief and victim see it) and
the *identity* of held development cards (buys draw face down; plays are
public). Everything else -- production, build/trade costs, discards (cards
returned to the face-up bank), Monopoly surrenders, hand and dev-card counts,
the bank itself -- is public. Per-type *totals* across all hands stay public
because every bank flow is public and steals only move cards between players.

The update is diff-based: ``update_belief`` watches a ``(before, after)``
transition plus the action that caused it, applies the exact public delta for
flows the observer sees, and decays/widens the bounds for hidden steals. A
final constraint-propagation pass (hand sizes, public per-type totals) tightens
both bounds; with two players it pins the opponent's hand exactly, recovering
"2p Catan is perfect-information up to dev-card identities" as a derived
property rather than an assumption.

``censor`` produces the matching leak-free ``BoardState``: the observer's own
rows are exact, opponents' resources collapse to the proven minima, opponents'
development cards return to the deck (which becomes the observer's *unseen
pool*), and the PRNG key is a constant. A censored state is a container for
public knowledge, not a playable position -- agents rebuild a concrete world
from it by sampling (see catan-agents' ``sample_world``).
"""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, UInt8

from catan_engine.board.dev_cards import DEV_CARD_COUNTS, N_DEV_CARD_TYPES, DevCard
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import BoardState, IntScalar, to_u8
from catan_engine.mechanics.action import ActionParams, ActionType

__all__ = [
    "BeliefState",
    "PlayerBelief",
    "make_belief",
    "update_belief",
    "player_belief",
    "censor",
]

# Batched aliases (BeliefState fields), following the BoardState convention of a
# leading variable `batch` axis; the functions below are single-game (the env
# vmaps them), so they annotate with the batch-free *Vec aliases.
ResBoundsArray = UInt8[Array, f"batch observers players resources={N_RESOURCES}"]
DevPlayedArray = UInt8[Array, f"batch dev_card_types={N_DEV_CARD_TYPES}"]
ResBoundsVec = UInt8[Array, f"observers players resources={N_RESOURCES}"]
DevPlayedVec = UInt8[Array, f"dev_card_types={N_DEV_CARD_TYPES}"]
PlayerResBoundsVec = UInt8[Array, f"players resources={N_RESOURCES}"]
PlayerCountVec = Int[Array, "players"]
ResTotalVec = Int[Array, f"resources={N_RESOURCES}"]

_DEV_COUNTS = jnp.asarray(DEV_CARD_COUNTS, dtype=jnp.int32)


class BeliefState(NamedTuple):
    """Per-observer knowledge about a batch of games (all publicly derivable).

    ``res_lo[b, o, p, r]`` / ``res_hi[b, o, p, r]`` bound player ``p``'s count
    of resource ``r`` as proven by observer ``o``'s public information;
    ``lo <= true <= hi`` always, with ``lo == hi`` where ``o`` knows the count
    exactly (in particular each observer's own row). ``dev_played`` counts
    publicly played development cards by type (observer-independent), which by
    conservation gives every observer's unseen dev pool:
    ``DEV_CARD_COUNTS - dev_played - own hand``.
    """

    res_lo: ResBoundsArray
    res_hi: ResBoundsArray
    dev_played: DevPlayedArray


class PlayerBelief(NamedTuple):
    """One observer's slice of the belief plus the public card counts.

    The form consumed by agents (single game; batch by vmapping): per-type
    resource bounds for every player, public hand / dev-card sizes, and the
    public per-type resource totals across all hands (the bank holds
    ``BANK_INITIAL - res_total``).
    """

    res_lo: PlayerResBoundsVec
    res_hi: PlayerResBoundsVec
    hand_size: PlayerCountVec
    dev_count: PlayerCountVec
    res_total: ResTotalVec


def make_belief(batch_size: int = 1, n_players: int = N_PLAYERS) -> BeliefState:
    """The belief matching a fresh board: empty hands, nothing played."""
    B, P = batch_size, n_players
    bounds = jnp.zeros((B, P, P, N_RESOURCES), dtype=jnp.uint8)
    return BeliefState(
        res_lo=bounds,
        res_hi=bounds,
        dev_played=jnp.zeros((B, N_DEV_CARD_TYPES), dtype=jnp.uint8),
    )


def _tighten(
    lo: jax.Array, hi: jax.Array, hand: jax.Array, total: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """One constraint-propagation pass over a ``(players, resources)`` bound pair.

    Each rule is individually sound, so applying them once per step converges
    toward the fixpoint over time: a hand can't exceed its public size, a type
    can't exceed the public total minus what others provably hold, and the dual
    floors. All int32.
    """
    hi = jnp.minimum(hi, hand[:, None])
    hi = jnp.minimum(hi, total[None, :] - (lo.sum(axis=0)[None, :] - lo))
    lo = jnp.maximum(lo, hand[:, None] - (hi.sum(axis=1)[:, None] - hi))
    lo = jnp.maximum(lo, total[None, :] - (hi.sum(axis=0)[None, :] - hi))
    return jnp.clip(lo, 0, None), jnp.clip(hi, 0, None)


def update_belief(
    belief: BeliefState,
    before: BoardState,
    after: BoardState,
    action_type: IntScalar,
    params: ActionParams,
) -> BeliefState:
    """Advance one game's belief across a ``(before, after)`` transition.

    Sound for any transition produced by ``apply_action`` with the given
    ``(action_type, params)``, including rejected (``INVALID``) actions, which
    leave the belief unchanged.
    """
    n = before.n_players
    res_a = after.player_resources.astype(jnp.int32)  # (P, R)
    d = res_a - before.player_resources.astype(jnp.int32)
    hand = res_a.sum(axis=1)  # (P,) public
    total = res_a.sum(axis=0)  # (R,) public (all bank flows are public)

    # Publicly played dev cards: the positive part of the hand decrease (a buy
    # *increases* a hand, so hidden draws never register here).
    played = jnp.clip(
        before.dev_hand.astype(jnp.int32) - after.dev_hand.astype(jnp.int32), 0, None
    ).sum(axis=0)
    dev_played = to_u8(belief.dev_played.astype(jnp.int32) + played)

    # A robber steal is the one resource flow third parties can't type.
    thief = before.current_player.astype(jnp.int32)
    victim = jnp.clip(params.target, 0, n - 1)
    robber_act = (action_type == ActionType.MOVE_ROBBER) | (
        action_type == ActionType.PLAY_KNIGHT
    )
    stole = (
        robber_act
        & (params.target >= 0)
        & (hand[victim] < before.player_resources.astype(jnp.int32)[victim].sum())
    )

    # Monopoly publicly reveals every player's exact count of the named type.
    mono_r = jnp.clip(params.idx, 0, N_RESOURCES - 1)
    mono = (action_type == ActionType.PLAY_MONOPOLY) & (
        played[DevCard.MONOPOLY] > 0
    )

    def observer(o: jax.Array, lo8: jax.Array, hi8: jax.Array) -> tuple[
        jax.Array, jax.Array
    ]:
        lo, hi = lo8.astype(jnp.int32), hi8.astype(jnp.int32)
        sees = ~stole | (o == thief) | (o == victim)
        # Seen flows: the typed delta is public, so both bounds track it exactly.
        lo_pub = jnp.clip(lo + d, 0, None)
        hi_pub = jnp.clip(hi + d, 0, None)
        # Hidden steal: the victim provably keeps at least lo - 1 of each type;
        # the thief may now hold one more of any type the victim could have held.
        lo_hid = lo.at[victim].set(jnp.clip(lo[victim] - 1, 0, None))
        hi_hid = hi.at[thief].add((hi[victim] > 0).astype(jnp.int32))
        lo = jnp.where(sees, lo_pub, lo_hid)
        hi = jnp.where(sees, hi_pub, hi_hid)
        # Monopoly pin (the surrendered counts are announced).
        lo = jnp.where(mono, lo.at[:, mono_r].set(res_a[:, mono_r]), lo)
        hi = jnp.where(mono, hi.at[:, mono_r].set(res_a[:, mono_r]), hi)
        # Own row is exact by definition.
        lo = lo.at[o].set(res_a[o])
        hi = hi.at[o].set(res_a[o])
        lo, hi = _tighten(lo, hi, hand, total)
        return to_u8(lo), to_u8(hi)

    obs = jnp.arange(n, dtype=jnp.int32)
    res_lo, res_hi = jax.vmap(observer)(obs, belief.res_lo, belief.res_hi)
    return BeliefState(res_lo=res_lo, res_hi=res_hi, dev_played=dev_played)


def player_belief(
    state: BoardState, belief: BeliefState, observer: IntScalar | int
) -> PlayerBelief:
    """``observer``'s slice of the belief plus the public card counts (one game)."""
    return PlayerBelief(
        res_lo=belief.res_lo[observer],
        res_hi=belief.res_hi[observer],
        hand_size=state.player_resources.astype(jnp.int32).sum(axis=1),
        dev_count=state.dev_hand.astype(jnp.int32).sum(axis=1),
        res_total=state.player_resources.astype(jnp.int32).sum(axis=0),
    )


def censor(
    state: BoardState, belief: BeliefState, observer: IntScalar | int
) -> BoardState:
    """``state`` with everything ``observer`` can't know removed (one game).

    Opponents' resources collapse to the proven minima, their development cards
    return to the deck (making ``dev_deck`` the observer's unseen pool),
    ``dev_bought`` survives only for the acting observer, and the PRNG key is a
    constant. The result is a container for public knowledge -- rebuild a
    playable position with a posterior sample before stepping it.
    """
    lo = belief.res_lo[observer]
    res = lo.at[observer].set(state.player_resources[observer])
    dev = (
        jnp.zeros_like(state.dev_hand).at[observer].set(state.dev_hand[observer])
    )
    pool = (
        _DEV_COUNTS
        - belief.dev_played.astype(jnp.int32)
        - state.dev_hand[observer].astype(jnp.int32)
    )
    own_turn = observer == state.current_player.astype(jnp.int32)
    return state._replace(
        player_resources=res,
        dev_hand=dev,
        dev_deck=to_u8(pool),
        dev_bought=jnp.where(own_turn, state.dev_bought, jnp.zeros_like(state.dev_bought)),
        key=jax.random.key(0),
    )
