"""Shared search infrastructure: the value frame, the priors, and the dice
constants the search tree and its public wrapper both build on."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import (
    VICTORY_POINTS_TO_WIN,
    BoardState,
    BoolScalar,
    IntScalar,
    KeyScalar,
)
from settlrl_engine.env import N_FLAT
from settlrl_engine.mechanics.action import ActionType
from settlrl_engine.mechanics.common import player_total_vp

from settlrl_agents.greedy import TIER_SCORES
from settlrl_agents.internal.rows import ROW_TYPE as _ROW_TYPE
from settlrl_agents.policy import FlatMask

_ILLEGAL = -1e9  # prior logit for illegal moves

_Weights = Float[Array, f"flat={N_FLAT}"]  # one position's improved-policy weights

PolicyWeights = Callable[
    [KeyScalar, BoardLayout, BeliefView, IntScalar, FlatMask], _Weights
]
"""The search's improved-policy weights over the flat actions — a distribution
for ``num_simulations`` > 0, the lookahead logits at 0. The AlphaZero policy
target (see :func:`search.make_search_weights`)."""

# Trade proposals are excluded from the priors outright: under the paranoid
# two-sided frame the in-tree responder prices every offer as rejected-or-
# harmful, so their ~100 near-tied rows would only flood the candidate pool — the
# search answers trades through search but only ever *offers* one at the root.
_NO_PROPOSE = jnp.where(_ROW_TYPE == ActionType.PROPOSE_TRADE, _ILLEGAL, 0.0)
# Interior-node prior: greedy's static tier table, tempered so tier gaps (>= 100)
# land ~2 nats apart — strong enough to order first expansions, weak enough for a
# few backed-up values to override.
_TIER_LOGITS = TIER_SCORES / 50.0 + _NO_PROPOSE

# Two-dice outcomes and their probabilities.
_ROLLS = jnp.arange(2, 13)
_ROLL_P = jnp.asarray([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=jnp.float32) / 36.0


def _terminal(state: BoardState) -> BoolScalar:
    """Whether the game is over: its current player has won (only the turn's
    owner can win — mirrors the engine's ``awards.current_player_won``)."""
    cur = state.current_player.astype(jnp.int32)
    return player_total_vp(state, cur) >= VICTORY_POINTS_TO_WIN


def _winner(state: BoardState) -> IntScalar:
    """The winner of a terminal state: its current player."""
    return state.current_player.astype(jnp.int32)
