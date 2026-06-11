"""Stochastic MCTS: dice and dev draws as explicit chance nodes.

The tree alternates decision and chance nodes
(:func:`mctx.stochastic_muzero_policy`). A ``ROLL_DICE`` edge stops at an
afterstate whose 11 children are the real outcomes at their true
probabilities, and a ``BUY_DEVELOPMENT_CARD`` edge at one whose children are
the card types weighted by the deck's current composition — both resolved
exactly via the engine's forced-outcome seams. Chance-node selection visits
outcomes proportionally, so a chance edge's backed-up value is an average
over futures instead of one sample: neither the strategy-fusion bias nor the
one-ply draw peek of the determinized searches can arise. Every other action
resolves fully at the decision step (steal identities stay determinized) and
its chance node is a delta.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import mctx
from catan_engine.belief import BeliefView
from catan_engine.board import Board
from catan_engine.board.dev_cards import N_DEV_CARD_TYPES
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import BoardState, IntScalar, KeyScalar
from catan_engine.env import (
    N_FLAT,
    ActionParams,
    available,
    flat_available,
    flat_to_action,
)
from catan_engine.mechanics.action import ActionType, apply_action
from catan_engine.mechanics.common import agent_selection_single

from catan_agents.search.mcts import (
    _ILLEGAL,
    _QTRANSFORM,
    _ROLL_P,
    _ROW_PARAMS,
    _ROW_TYPE,
    _TIER_LOGITS,
    _terminal,
    _winner,
)
from catan_agents.shared.policy import BeliefPolicy, FlatAction, FlatMask
from catan_agents.shared.sample import sample_world
from catan_agents.shared.value import Value, ValueFunction, heuristic_value

_N_OUTCOMES = 11  # two-dice sums 2..12

_Weights = Any  # one tree's visit probabilities over the flat actions

# Chance logits per lane kind: the true roll distribution, or a delta on
# outcome 0 for actions whose chance node is degenerate.
_ROLL_LOGITS = jnp.log(_ROLL_P)
_DELTA_LOGITS = jnp.concatenate(
    [jnp.zeros((1,)), jnp.full((_N_OUTCOMES - 1,), _ILLEGAL)]
)


class _Afterstate(NamedTuple):
    """The afterstate embedding: a committed action awaiting its chance step.

    ``state`` is the fully-resolved next state for delta lanes and the
    *pre-action* state for ``pending_roll`` / ``pending_buy`` lanes (a roll or
    a dev-card draw awaiting its outcome). ``mover_side`` / ``was_terminal``
    carry the acting side and absorbing flag of the decision node, in whose
    frame the chance step must emit reward and discount.
    """

    state: BoardState
    pending_roll: jax.Array
    pending_buy: jax.Array
    mover_side: jax.Array
    was_terminal: jax.Array


class _Resolved(NamedTuple):
    """A chance outcome applied: the child state plus its backup quantities."""

    state: BoardState
    reward: jax.Array
    discount: jax.Array
    value_sign: jax.Array  # child side sign for framing the leaf value
    terminal: jax.Array


def _resolve(
    layout: BoardLayout,
    after: _Afterstate,
    outcome: jax.Array,
    player: IntScalar,
) -> _Resolved:
    """Apply chance ``outcome`` (a forced roll or dev draw for pending lanes,
    a no-op otherwise) and compute the side-framed reward / discount of the
    full transition."""
    roll = (jnp.clip(outcome, 0, _N_OUTCOMES - 1) + 2).astype(jnp.int32)
    card = (jnp.clip(outcome, 0, N_DEV_CARD_TYPES - 1) + 1).astype(jnp.int32)
    atype = jnp.where(
        after.pending_buy,
        jnp.asarray(ActionType.BUY_DEVELOPMENT_CARD, dtype=_ROW_TYPE.dtype),
        jnp.asarray(ActionType.ROLL_DICE, dtype=_ROW_TYPE.dtype),
    )
    idx = jnp.where(after.pending_buy, card, roll)
    params = ActionParams(idx=idx, target=jnp.zeros_like(idx))
    resolved, _ = jax.vmap(apply_action)(
        layout, after.state, atype, params, after.pending_roll | after.pending_buy
    )
    now_terminal = jax.vmap(_terminal)(resolved)
    won = now_terminal & ~after.was_terminal
    next_side = jax.vmap(agent_selection_single)(resolved) == player
    winner_side = jax.vmap(_winner)(resolved) == player
    return _Resolved(
        state=resolved,
        reward=jnp.where(
            won, jnp.where(winner_side == after.mover_side, 1.0, -1.0), 0.0
        ),
        discount=jnp.where(
            now_terminal, 0.0, jnp.where(next_side == after.mover_side, 1.0, -1.0)
        ),
        value_sign=jnp.where(next_side, 1.0, -1.0),
        terminal=now_terminal,
    )


def make_smcts(
    value: ValueFunction,
    *,
    num_worlds: int = 4,
    num_futures: int = 1,
    num_simulations: int = 64,
    value_scale: float = 20.0,
    prior_scale: float = 5.0,
    pb_c_init: float = 1.25,
    dirichlet_fraction: float = 0.0,
    rescale_q: bool = False,
    dev_chance: bool = True,
) -> BeliefPolicy:
    """Stochastic-MuZero search using the engine itself as the dynamics model.

    Same ensemble semantics as :func:`~catan_agents.search.mcts.make_mcts`
    (``num_worlds`` determinizations x ``num_futures`` chance re-keyings,
    averaged visit weights), but dice are explicit 11-way chance nodes, so a
    simulation costs two tree edges per game ply. ``prior_scale`` softens the
    one-step value sweep into the PUCT prior (PUCT explores by prior mass; the
    raw sweep is near-one-hot).
    """

    def leaf_value(layout: BoardLayout, state: BoardState, p: IntScalar) -> Value:
        return jnp.tanh(value(layout, state, p) / value_scale)

    def search_world(
        key: KeyScalar,
        layout: BoardLayout,
        state: BoardState,
        player: IntScalar,
        mask: FlatMask,
    ) -> _Weights:
        """Visit weights from one stochastic search of a single concrete world."""

        def decision_fn(
            params: None,
            rng: KeyScalar,
            action: jax.Array,
            embedding: Board,
        ) -> tuple[mctx.DecisionRecurrentFnOutput, tuple[BoardLayout, _Afterstate]]:
            layout, state = embedding
            # mctx calls this with chance indices at chance nodes; clip and
            # let the `where` in its wrapper discard the result.
            action = jnp.clip(action, 0, N_FLAT - 1)
            atype, aparams = flat_to_action(action)
            mover = jax.vmap(agent_selection_single)(state)
            was_terminal = jax.vmap(_terminal)(state)
            avail = available((layout, state), atype, aparams) & ~was_terminal
            is_roll = (atype == ActionType.ROLL_DICE) & avail
            is_buy = (atype == ActionType.BUY_DEVELOPMENT_CARD) & avail & dev_chance
            applied, _ = jax.vmap(apply_action)(
                layout, state, atype, aparams, avail & ~is_roll & ~is_buy
            )
            after = _Afterstate(
                state=applied,
                pending_roll=is_roll,
                pending_buy=is_buy,
                mover_side=mover == player,
                was_terminal=was_terminal,
            )
            # Afterstate value, in the decision node's frame: for the delta
            # lanes the (sole) chance child's backup; for roll lanes the leaf
            # value as-is — the chance children refine it with real outcomes.
            sign_m = jnp.where(after.mover_side, 1.0, -1.0)
            res = _resolve(layout, after, jnp.zeros_like(action), player)
            v_child = jnp.where(
                res.terminal,
                0.0,
                res.value_sign
                * jax.vmap(leaf_value, in_axes=(0, 0, None))(layout, res.state, player),
            )
            after_value = jnp.where(
                is_roll | is_buy,
                sign_m
                * jax.vmap(leaf_value, in_axes=(0, 0, None))(layout, state, player),
                res.reward + res.discount * v_child,
            )
            # Dev-draw outcomes at the deck's current composition (the chance
            # probabilities are state-dependent; empty types are unselectable).
            deck = state.dev_deck.astype(jnp.float32)
            deck_logits = jnp.where(
                deck > 0,
                jnp.log(deck / jnp.maximum(deck.sum(1, keepdims=True), 1.0)),
                _ILLEGAL,
            )
            buy_logits = jnp.concatenate(
                [
                    deck_logits,
                    jnp.full((deck.shape[0], _N_OUTCOMES - N_DEV_CARD_TYPES), _ILLEGAL),
                ],
                axis=1,
            )
            chance_logits = jnp.where(
                is_roll[:, None],
                _ROLL_LOGITS[None, :],
                jnp.where(is_buy[:, None], buy_logits, _DELTA_LOGITS[None, :]),
            )
            out = mctx.DecisionRecurrentFnOutput(  # type: ignore[call-arg]
                chance_logits=chance_logits,
                afterstate_value=after_value,
            )
            return out, (layout, after)

        def chance_fn(
            params: None,
            rng: KeyScalar,
            outcome: jax.Array,
            embedding: tuple[BoardLayout, _Afterstate],
        ) -> tuple[mctx.ChanceRecurrentFnOutput, Board]:
            layout, after = embedding
            res = _resolve(layout, after, outcome, player)
            v = jnp.where(
                res.terminal,
                0.0,
                res.value_sign
                * jax.vmap(leaf_value, in_axes=(0, 0, None))(layout, res.state, player),
            )
            priors = jnp.where(
                flat_available((layout, res.state)), _TIER_LOGITS, _ILLEGAL
            )
            out = mctx.ChanceRecurrentFnOutput(  # type: ignore[call-arg]
                action_logits=priors,
                value=v,
                reward=res.reward,
                discount=res.discount,
            )
            return out, (layout, res.state)

        successors, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, _ROW_TYPE, _ROW_PARAMS, mask
        )
        root_vals = jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)
        batched: Any = jax.tree.map(lambda x: x[None], (layout, state))
        root = mctx.RootFnOutput(  # type: ignore[call-arg]
            prior_logits=jnp.where(mask, root_vals / prior_scale, _ILLEGAL)[None],
            value=leaf_value(layout, state, player)[None],
            embedding=batched,
        )
        out = mctx.stochastic_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            decision_recurrent_fn=decision_fn,
            chance_recurrent_fn=chance_fn,
            num_simulations=num_simulations,
            invalid_actions=(~mask)[None],
            qtransform=(
                mctx.qtransform_by_parent_and_siblings if rescale_q else _QTRANSFORM
            ),
            pb_c_init=pb_c_init,
            dirichlet_fraction=dirichlet_fraction,
        )
        return cast(_Weights, out.action_weights[0])

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        k_world, k_future, k_search = jax.random.split(key, 3)
        states = jax.vmap(sample_world, in_axes=(0, None, None))(
            jax.random.split(k_world, num_worlds), view, player
        )
        n_trees = num_worlds * num_futures
        states = jax.tree.map(lambda x: jnp.repeat(x, num_futures, axis=0), states)
        states = states._replace(key=jax.random.split(k_future, n_trees))
        weights = jax.vmap(search_world, in_axes=(0, None, 0, None, None))(
            jax.random.split(k_search, n_trees), layout, states, player, mask
        )
        return jnp.argmax(jnp.where(mask, weights.mean(axis=0), -jnp.inf))

    return policy


smcts_policy = make_smcts(heuristic_value)
"""The stochastic-MCTS agent over :func:`heuristic_value`."""
