"""State value functions: how good is this board for a given player?

A :class:`ValueFunction` scores a concrete board. The search agents only ever
hand it *sampled* worlds (see ``settlrl_search.sample``), so the "hidden" fields
it reads there are belief-consistent samples; it still treats opponents' dev
cards as counts (their composition is a distribution over the known deck) and
the player's own as exact.

The shipped heuristic is a *weighting*: the terms themselves live in
``internal.feature_engineering.board_features``, and :func:`make_heuristic`
just takes their dot product with its weight kwargs.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, BoolScalar, Player
from settlrl_search.value import Value, ValueFunction

from settlrl_agents.internal.feature_engineering import BoardFeatures, board_features

__all__ = [
    "TUNED_WEIGHTS",
    "Value",
    "ValueFunction",
    "heuristic_value",
    "make_heuristic",
    "make_linear",
    "tuned_value",
]


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
    :class:`~settlrl_agents.internal.feature_engineering.BoardFeatures` (the
    terms document themselves there). The value is the player's weighted
    strength minus the best opponent's.
    """

    def strength(
        layout: BoardLayout, state: BoardState, p: Player, exact_dev: BoolScalar
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

    def value(layout: BoardLayout, state: BoardState, player: Player) -> Value:
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
        layout: BoardLayout, state: BoardState, p: Player, exact_dev: BoolScalar
    ) -> Value:
        f = board_features(layout, state, p, exact_dev)
        out: Value = sum(
            (c * getattr(f, n) for n, c in zip(names, coefs, strict=True)),
            jnp.float32(0.0),
        )
        return out

    def value(layout: BoardLayout, state: BoardState, player: Player) -> Value:
        players = jnp.arange(state.n_players)
        strengths = jax.vmap(lambda q: strength(layout, state, q, q == player))(players)
        mine = strengths[player]
        best_other = jnp.max(jnp.where(players == player, -jnp.inf, strengths))
        return mine - best_other

    return value


# Count-conditional tuned weights (experiments/0002_linear_value_fitting).
# 2p: the self-play CEM champion — beat the hand-tuned weights 56.1%
# head-to-head at 2p (n=310, lower 2-sigma 50.5%) but measured parity at 4p, so
# it ships per count, not as the defaults. 4p: the hand-tuned weights until
# a 4p-arena champion passes its gate.
TUNED_WEIGHTS: dict[int, dict[str, float]] = {
    2: {
        "vp": 14.2031, "production": 1.4078, "diversity": 0.7259,
        "hand": 0.5589, "scarce": 0.7476, "over": -0.5124, "n_dev": 1.9031,
        "best_spot": 0.5498, "n_roads": 0.5699, "progress": 1.2971,
        "knights": 0.9661, "wheat_ore": 0.0906, "race": 0.5990,
        "numbers": 0.2066, "held_knights": 1.3367,
    },
    4: {
        "vp": 10.0, "production": 1.0, "diversity": 0.6, "hand": 0.3,
        "scarce": 1.0, "over": -0.4, "n_dev": 1.5, "best_spot": 0.5,
        "n_roads": 0.15, "progress": 2.0, "knights": 0.5, "wheat_ore": 0.25,
        "race": 0.8, "numbers": 0.3, "held_knights": 0.8,
    },
}  # fmt: skip


_TUNED = {n: make_linear(w) for n, w in TUNED_WEIGHTS.items()}


def tuned_value(layout: BoardLayout, state: BoardState, player: Player) -> Value:
    """The count-tuned value: ``TUNED_WEIGHTS`` picked by the (static) seated
    player count — 3p uses the 4p weights. Not the default ``heuristic_value``;
    select it explicitly (e.g. ``"value": "tuned"`` in a bench spec)."""
    return (_TUNED[2] if state.n_players == 2 else _TUNED[4])(layout, state, player)
