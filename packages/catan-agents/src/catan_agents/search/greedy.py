"""Greedy over a value function: pick the action whose successor scores best."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from catan_engine.belief import BeliefView
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import BoardState, IntScalar, KeyScalar
from catan_engine.env import N_FLAT, flat_to_action
from catan_engine.mechanics.action import (
    ActionParams,
    ActionType,
    action_available,
    apply_action,
)
from catan_engine.mechanics.trade import _PARTNER_BITS

from catan_agents.shared.policy import BeliefPolicy, FlatAction, FlatMask
from catan_agents.shared.sample import sample_world
from catan_agents.shared.value import ValueFunction, heuristic_value

# Static decode of every flat row: its action type and (idx, target) params.
_ROW_TYPE, _ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))

# The ProposeTrade rows and their partners (the low bits of the packed
# target — see trade.pack_trade): a proposal's successor is material-neutral
# (cards only move on the partner's accept), so these rows are scored by
# their *accepted* outcome instead (see ``make_greedy``).
_PROPOSE_ROWS = jnp.asarray(
    np.flatnonzero(np.asarray(_ROW_TYPE) == int(ActionType.PROPOSE_TRADE))
)
_PROPOSE_PARTNER = _ROW_PARAMS.target[_PROPOSE_ROWS] & ((1 << _PARTNER_BITS) - 1)
_ACCEPT = jnp.int32(ActionType.ACCEPT_TRADE)
_NO_PARAMS = ActionParams(idx=jnp.int32(0), target=jnp.int32(0))


def _accepted_outcome(layout: BoardLayout, succ: BoardState) -> BoardState:
    """The state after the partner accepts ``succ``'s pending proposal.

    ``succ`` unchanged when accepting is illegal there (no pending trade, or
    the partner lacks the asked-for card in this world).
    """
    avail = action_available(layout, succ, _ACCEPT, _NO_PARAMS)
    accepted, _ = apply_action(layout, succ, _ACCEPT, _NO_PARAMS, avail)
    return accepted


def make_greedy(
    value: ValueFunction,
    *,
    trade_penalty: float = 0.25,
    propose_rate: float = 0.5,
) -> BeliefPolicy:
    """One-step lookahead: apply every legal action and argmax ``value``.

    The view is first made concrete with one
    :func:`~catan_agents.shared.sample.sample_world` draw, so stochastic
    successors — dice, steals, dev-card draws — are the policy's own samples
    over a world consistent with what the player knows.

    Trade proposals are the one action whose successor is material-neutral, so
    they are scored by their accepted outcome instead, gated on a model of the
    partner: the same ``value`` from the partner's seat must prefer accepting
    (in particular, proposals the sampled partner cannot accept score nothing).
    ``trade_penalty`` is then subtracted — the quality bar that keeps marginal
    offers below simply not trading. The model can still be wrong about a real
    opponent, and the engine keeps no memory of rejected offers, so a
    deterministic proposer would re-offer the same trade forever against a
    partner it mispredicts; ``propose_rate`` (the chance proposing is allowed
    at all on a given move) bounds such streaks geometrically.
    """

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        k_world, k_noise, k_gate = jax.random.split(key, 3)
        state = sample_world(k_world, view, player)
        successors, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, _ROW_TYPE, _ROW_PARAMS, mask
        )
        values = jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)

        # Score each proposal by its accepted outcome, if the modeled partner
        # would take the trade and proposing is allowed this move.
        offered = jax.tree.map(lambda x: x[_PROPOSE_ROWS], successors)
        accepted = jax.vmap(_accepted_outcome, in_axes=(None, 0))(layout, offered)
        mine = jax.vmap(value, in_axes=(None, 0, None))(layout, accepted, player)
        partner_reject = jax.vmap(value, in_axes=(None, 0, 0))(
            layout, offered, _PROPOSE_PARTNER
        )
        partner_accept = jax.vmap(value, in_axes=(None, 0, 0))(
            layout, accepted, _PROPOSE_PARTNER
        )
        allowed = jax.random.uniform(k_gate) < propose_rate
        propose_score = jnp.where(
            (partner_accept > partner_reject) & allowed,
            mine - trade_penalty,
            -jnp.inf,
        )
        values = values.at[_PROPOSE_ROWS].set(propose_score)

        # Tiny uniform noise breaks exact-value ties at random.
        noise = jax.random.uniform(k_noise, (N_FLAT,)) * 1e-4
        return jnp.argmax(jnp.where(mask, values + noise, -jnp.inf))

    return policy


lookahead_policy = make_greedy(heuristic_value)
"""The value-greedy agent: one-step lookahead over :func:`heuristic_value`."""
