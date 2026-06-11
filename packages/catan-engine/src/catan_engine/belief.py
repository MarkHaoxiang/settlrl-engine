"""Card counting: what each player can legitimately know about hidden hands.

An optional companion to ``BoardState`` (``BatchedCatanEnv(track_beliefs=
True)``): per observer, proven per-type bounds on every player's hand plus
the public played-dev-card tally -- everything derivable from public
information, so handing it to an agent never leaks. Hidden in this engine's
Catan: the *type* of a card moved by a robber steal (thief and victim see it)
and the *identity* of held development cards; everything else is public.

``update_belief`` advances one game's belief across a ``(before, after)``
transition; ``belief_view`` projects one observer's knowledge into the
agent-facing ``BeliefView``, which is deliberately *not* a ``BoardState`` and
not steppable -- the road back to a playable position is catan-agents'
``sample_world``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, UInt8

from catan_engine.board.dev_cards import (
    DEV_CARD_COUNTS,
    N_DEV_CARD_TYPES,
    DevCard,
    DevCardDeckArray,
    DevDeckVec,
)
from catan_engine.board.resources import (
    N_PLAYERS,
    N_RESOURCES,
    PlayerResourcesVec,
)
from catan_engine.board.state import (
    BoardState,
    EdgeRoadVec,
    IntScalar,
    PlayerU8Vec,
    U8Scalar,
    VertexOwnerVec,
    VertexTypeVec,
    to_u8,
)
from catan_engine.mechanics.action import ActionParams, ActionType

__all__ = [
    "BeliefState",
    "BeliefView",
    "PlayerBelief",
    "PublicState",
    "belief_view",
    "make_belief",
    "update_belief",
]

# Belief-specific aliases; everything board-shaped reuses the canonical aliases
# from board.state / board.resources / board.dev_cards. The functions below are
# single-game (the env vmaps them), so they annotate with the batch-free forms.
ResBoundsArray = UInt8[Array, f"batch observers players resources={N_RESOURCES}"]
ResBoundsVec = Int[Array, f"players resources={N_RESOURCES}"]  # one observer, int32
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
    dev_played: DevCardDeckArray


class PlayerBelief(NamedTuple):
    """One observer's slice of the belief plus the public card counts.

    Per-type resource bounds for every player (the observer's own row is
    exact), public hand / dev-card sizes, and the public per-type resource
    totals across all hands (the bank holds ``BANK_INITIAL - res_total``).
    Single game; batch by vmapping.
    """

    res_lo: PlayerResourcesVec
    res_hi: PlayerResourcesVec
    hand_size: PlayerCountVec
    dev_count: PlayerCountVec
    res_total: ResTotalVec


class PublicState(NamedTuple):
    """The publicly visible part of a ``BoardState`` (one game).

    Field-for-field copies of the ``BoardState`` fields of the same names; the
    hidden fields (``player_resources``, ``dev_hand``, ``dev_deck``,
    ``dev_bought``, ``key``) have no counterpart here.
    """

    vertex_owner: VertexOwnerVec
    vertex_type: VertexTypeVec
    edge_road: EdgeRoadVec
    robber: U8Scalar
    victory_points: PlayerU8Vec
    knights_played: PlayerU8Vec
    phase: U8Scalar
    current_player: U8Scalar
    setup_index: U8Scalar
    dice_roll: U8Scalar
    has_rolled: U8Scalar
    dev_played: U8Scalar
    free_roads: U8Scalar
    pending_discard: PlayerU8Vec
    trade_partner: U8Scalar
    trade_give: U8Scalar
    trade_receive: U8Scalar
    longest_road_owner: U8Scalar
    largest_army_owner: U8Scalar
    longest_road_len: U8Scalar

    @property
    def n_players(self) -> int:
        """Seated players (2..N_PLAYERS), read off the player axis (static)."""
        return self.victory_points.shape[-1]


class BeliefView(NamedTuple):
    """Everything one observer knows about one game -- the agent-facing seam.

    The public board fields, the observer's proven resource bounds and public
    counts, their own development hand, their own purchases this turn (zeros
    off-turn), and the unseen dev pool (deck + opponents' hands). Not a
    ``BoardState`` and not steppable: rebuild a playable position with a
    posterior sample (catan-agents' ``sample_world``) first.
    """

    public: PublicState
    belief: PlayerBelief
    own_dev: DevDeckVec
    own_bought: DevDeckVec
    unseen_dev: DevDeckVec

    @property
    def n_players(self) -> int:
        """Seated players (2..N_PLAYERS), read off the player axis (static)."""
        return self.public.n_players


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
    lo: ResBoundsVec, hi: ResBoundsVec, hand: PlayerCountVec, total: ResTotalVec
) -> tuple[ResBoundsVec, ResBoundsVec]:
    """One constraint-propagation pass over a bound pair.

    Each rule is individually sound, so applying them once per step converges
    toward the fixpoint over time: a hand can't exceed its public size, a type
    can't exceed the public total minus what others provably hold, and the dual
    floors.
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
    mono = (action_type == ActionType.PLAY_MONOPOLY) & (played[DevCard.MONOPOLY] > 0)

    def observer(
        o: IntScalar, lo8: PlayerResourcesVec, hi8: PlayerResourcesVec
    ) -> tuple[PlayerResourcesVec, PlayerResourcesVec]:
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


def belief_view(
    state: BoardState, belief: BeliefState, observer: IntScalar | int
) -> BeliefView:
    """Project ``observer``'s knowledge of one game into a :class:`BeliefView`."""
    res = state.player_resources.astype(jnp.int32)
    own_dev = state.dev_hand[observer]
    own_turn = observer == state.current_player.astype(jnp.int32)
    return BeliefView(
        public=PublicState(**{f: getattr(state, f) for f in PublicState._fields}),
        belief=PlayerBelief(
            res_lo=belief.res_lo[observer],
            res_hi=belief.res_hi[observer],
            hand_size=res.sum(axis=1),
            dev_count=state.dev_hand.astype(jnp.int32).sum(axis=1),
            res_total=res.sum(axis=0),
        ),
        own_dev=own_dev,
        own_bought=jnp.where(
            own_turn, state.dev_bought, jnp.zeros_like(state.dev_bought)
        ),
        unseen_dev=to_u8(
            _DEV_COUNTS
            - belief.dev_played.astype(jnp.int32)
            - own_dev.astype(jnp.int32)
        ),
    )
