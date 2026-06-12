"""State value functions: how good is this board for a given player?

A :class:`ValueFunction` scores a concrete board. The search agents only ever
hand it *sampled* worlds (see ``catan_agents.sample``), so the "hidden" fields
it reads there are belief-consistent samples; it still treats opponents' dev
cards as counts (their composition is a distribution over the known deck) and
the player's own as exact.

The shipped heuristic is a *weighting*: the terms themselves live in
``internal.feature_engineering.board_features``, and :func:`make_heuristic`
just takes their dot product with its weight kwargs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import BoardState, BoolScalar, IntScalar
from jaxtyping import Array, Float

from catan_agents.internal.feature_engineering import BoardFeatures, board_features

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


def make_heuristic(
    *,
    w_vp: float = 10.0,
    w_prod: float = 1.0,
    w_hand: float = 0.3,
    w_scarce: float = 1.0,
    w_over: float = 0.4,
    w_dev: float = 1.5,
    w_spot: float = 0.5,
    w_road: float = 0.15,
    w_prog: float = 2.0,
    w_knight: float = 0.5,
    w_diverse: float = 0.6,
    w_port: float = 0.0,
    w_wheat_ore: float = 0.25,
    w_race: float = 0.8,
    w_numbers: float = 0.3,
    w_spots: float = 0.0,
    w_fill: float = 0.0,
    w_kheld: float = 0.8,
    w_second_spot: float = 0.0,
    w_reach: float = 0.0,
    w_army_lead: float = 0.0,
) -> ValueFunction:
    """Build a weighted heuristic value function (see :func:`heuristic_value`).

    Each weight prices one term of
    :class:`~catan_agents.internal.feature_engineering.BoardFeatures` (the
    terms document themselves there). The value is the player's weighted
    strength minus the best opponent's.
    """

    def strength(
        layout: BoardLayout, state: BoardState, p: IntScalar, exact_dev: BoolScalar
    ) -> Value:
        f = board_features(layout, state, p, exact_dev)
        return (
            w_vp * f.vp
            + w_prod * f.production
            + w_diverse * f.diversity
            + w_hand * f.hand
            + w_scarce * f.scarce
            - w_over * f.over
            + w_dev * f.n_dev
            + w_spot * f.best_spot
            + w_road * f.n_roads
            + w_prog * f.progress
            + w_knight * f.knights
            + w_port * f.ports
            + w_wheat_ore * f.wheat_ore
            + w_race * f.race
            + w_numbers * f.numbers
            + w_spots * f.n_spots
            + w_fill * f.fill
            + w_kheld * f.held_knights
            + w_second_spot * f.second_spot
            + w_reach * f.reach
            + w_army_lead * f.army_lead
        )

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        players = jnp.arange(state.n_players)
        strengths = jax.vmap(lambda q: strength(layout, state, q, q == player))(players)
        mine = strengths[player]
        best_other = jnp.max(jnp.where(players == player, -jnp.inf, strengths))
        return mine - best_other

    return value


heuristic_value = make_heuristic()
"""The shipped heuristic, at the weights that won the 2-player CLI tournament."""


def make_linear(weights: Mapping[str, float]) -> ValueFunction:
    """A linear value over named :class:`BoardFeatures` terms.

    ``weights`` maps feature names (``BoardFeatures._fields``) to
    coefficients; unnamed features count zero. The fitted-weights deployment
    seam: any classical fit over the engineered features drops in here.
    """
    unknown = set(weights) - set(BoardFeatures._fields)
    if unknown:
        raise ValueError(f"unknown features: {sorted(unknown)}")
    names = tuple(weights)
    coefs = tuple(float(weights[n]) for n in names)

    def strength(
        layout: BoardLayout, state: BoardState, p: IntScalar, exact_dev: BoolScalar
    ) -> Value:
        f = board_features(layout, state, p, exact_dev)
        out: Value = sum(
            (c * getattr(f, n) for n, c in zip(names, coefs, strict=True)),
            jnp.float32(0.0),
        )
        return out

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        players = jnp.arange(state.n_players)
        strengths = jax.vmap(lambda q: strength(layout, state, q, q == player))(players)
        mine = strengths[player]
        best_other = jnp.max(jnp.where(players == player, -jnp.inf, strengths))
        return mine - best_other

    return value
