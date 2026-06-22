"""Shared search infrastructure: the priors, the dice constants, and the
policy-weights type the search tree and its public wrapper both build on."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import IntScalar, KeyScalar
from settlrl_engine.env import N_FLAT
from settlrl_engine.mechanics.action import ActionType

from settlrl_agents.greedy import TIER_SCORES
from settlrl_agents.internal.rows import ROW_TYPE as _ROW_TYPE
from settlrl_agents.policy import FlatMask
from settlrl_agents.value import Value

_ILLEGAL = -1e9  # prior logit for illegal moves

_Weights = Float[Array, f"flat={N_FLAT}"]  # one position's improved-policy weights

PolicyWeights = Callable[
    [KeyScalar, BoardLayout, BeliefView, IntScalar, FlatMask], _Weights
]
"""The search's improved-policy weights over the flat actions — a distribution
for ``num_simulations`` > 0, the lookahead logits at 0. The AlphaZero policy
target (see :func:`search.make_search_weights`)."""

PolicyWeightsValue = Callable[
    [KeyScalar, BoardLayout, BeliefView, IntScalar, FlatMask], tuple[_Weights, Value]
]
"""As :data:`PolicyWeights`, but also returning the **searched root value** in the
searcher's frame (2·P(win)-1, [-1, 1]) — the AlphaZero value target's ``q`` term
(see :func:`search.make_search_weights_value`)."""

# Trade proposals are excluded from the priors (offered only at the root).
_NO_PROPOSE = jnp.where(_ROW_TYPE == ActionType.PROPOSE_TRADE, _ILLEGAL, 0.0)
# Interior-node prior: greedy's tier table, tempered so tier gaps land ~2 nats apart.
_TIER_LOGITS = TIER_SCORES / 50.0 + _NO_PROPOSE

# Two-dice outcomes and their probabilities.
_ROLLS = jnp.arange(2, 13)
_ROLL_P = jnp.asarray([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=jnp.float32) / 36.0
