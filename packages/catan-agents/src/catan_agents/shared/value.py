"""State value functions: how good is this board for a given player?

A :class:`ValueFunction` reads only information the player could legitimately
infer at a two-player table: the public board, exact resource counts (all
flows are public with two players), and dev cards as *counts* for opponents
(their composition is a distribution over the known deck) but exact for the
player itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from catan_engine.board.dev_cards import DEV_CARD_COUNTS, DevCard
from catan_engine.board.layout import N_TILES, N_VERTICES, TILE_V, BoardLayout
from catan_engine.board.state import BoardState, IntScalar

Value = Float[Array, ""]
"""A scalar state score for one player: higher is better, arbitrary scale."""


@runtime_checkable
class ValueFunction(Protocol):
    """A single-game state evaluation, pure and ``jit`` / ``vmap`` compatible.

    ``layout`` / ``state`` are one game's board (no batch axis); returns the
    state's value from ``player``'s point of view.
    """

    def __call__(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Value: ...


def tile_pips(tile_number: jax.Array) -> Float[Array, f"tiles={N_TILES}"]:
    """Expected-production weight per tile: 6 - |7 - number| (0 for the desert)."""
    n = tile_number.astype(jnp.int32)
    return jnp.where(n == 0, 0, 6 - jnp.abs(7 - n)).astype(jnp.float32)


def vertex_pips(tile_number: jax.Array) -> Float[Array, f"vertices={N_VERTICES}"]:
    """Summed pips of each vertex's adjacent tiles."""
    pips = tile_pips(tile_number)
    acc = jnp.zeros((N_VERTICES,), jnp.float32)
    return acc.at[TILE_V.reshape(-1)].add(jnp.repeat(pips, TILE_V.shape[1]))


# Heuristic weights. Victory points dominate; production pips reward good
# placements; sqrt of each resource count values a diverse, discard-resistant
# hand; held dev cards carry their expected VP-card share for opponents.
_W_VP = 10.0
_W_PROD = 1.0
_W_HAND = 0.3
_W_DEV = 1.5
_VP_CARD_SHARE = float(DEV_CARD_COUNTS[DevCard.VICTORY_POINT]) / float(
    sum(DEV_CARD_COUNTS)
)


def _strength(
    layout: BoardLayout, state: BoardState, p: jax.Array, exact_dev: jax.Array
) -> jax.Array:
    """One player's heuristic score; ``exact_dev`` switches own/inferred dev info."""
    # Public VP: buildings plus awards.
    vp = state.victory_points[p].astype(jnp.float32)
    vp += 2.0 * (state.longest_road_owner == p)
    vp += 2.0 * (state.largest_army_owner == p)
    # Production: pips of tiles adjacent to own buildings (city counts twice),
    # robber tile blocked.
    pips = tile_pips(layout.tile_number)
    pips = pips * (jnp.arange(N_TILES) != state.robber)
    weight = (state.vertex_owner[TILE_V] == p + 1) * state.vertex_type[TILE_V]
    production = pips @ weight.sum(axis=1).astype(jnp.float32)
    # Hand: diminishing per-resource value favours diversity and makes the
    # cheapest discard the most-held resource.
    hand = jnp.sqrt(state.player_resources[p].astype(jnp.float32)).sum()
    # Dev cards: own hand is exact (VP cards at full VP weight); an opponent's
    # is a count whose VP-card share is its prior over the deck composition.
    n_dev = state.dev_hand[p].astype(jnp.float32).sum()
    own_vp_cards = state.dev_hand[p, DevCard.VICTORY_POINT].astype(jnp.float32)
    dev_vp = jnp.where(exact_dev, own_vp_cards, n_dev * _VP_CARD_SHARE)
    return (
        _W_VP * (vp + dev_vp) + _W_PROD * production + _W_HAND * hand + _W_DEV * n_dev
    )


def heuristic_value(
    layout: BoardLayout, state: BoardState, player: IntScalar
) -> Value:
    """Hand-written heuristic: ``player``'s strength minus the best opponent's.

    Strength weighs victory points (awards and VP cards included), pip-weighted
    production of own buildings (robber-aware), hand diversity, and held dev
    cards.
    """
    players = jnp.arange(state.n_players)
    strengths = jax.vmap(lambda q: _strength(layout, state, q, q == player))(players)
    mine = strengths[player]
    best_other = jnp.max(jnp.where(players == player, -jnp.inf, strengths))
    return mine - best_other
