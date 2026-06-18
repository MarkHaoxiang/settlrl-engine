"""One principled search over the engine: re-determinizing MCTS on ``mctx``.

A single tree whose every simulation draws a fresh ``sample_world``
determinization and replays the root path under it, so a node's backed-up
value integrates over the belief instead of one frozen world (the ISMCTS fix
for the determinized-searcher strategy fusion that a frozen-world PIMC suffers).
The immediate dice roll is valued by its exact 11-roll expectation at the leaf;
deeper chance is integrated by the per-simulation resampling — so explicit
chance nodes buy nothing here. ``num_simulations=0`` collapses the tree to its
root: a one-step value sweep (the *lookahead* special case), which is also the
only configuration that offers trades.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import mctx
import numpy as np
from jaxtyping import Array, Bool, Float, Int
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import (
    VICTORY_POINTS_TO_WIN,
    BoardState,
    BoolScalar,
    IntScalar,
    KeyScalar,
)
from settlrl_engine.env import N_FLAT, available, flat_available, flat_to_action
from settlrl_engine.mechanics.action import (
    ActionParams,
    ActionType,
    action_available,
    apply_action,
)
from settlrl_engine.mechanics.common import agent_selection_single, player_total_vp
from settlrl_engine.mechanics.dice import distribute_resources
from settlrl_engine.mechanics.trade import _PARTNER_BITS

from settlrl_agents.greedy import TIER_SCORES
from settlrl_agents.internal.rows import ROW_PARAMS as _ROW_PARAMS
from settlrl_agents.internal.rows import ROW_TYPE as _ROW_TYPE
from settlrl_agents.policy import BeliefPolicy, FlatAction, FlatMask, PolicyPrior
from settlrl_agents.sample import sample_world
from settlrl_agents.value import Value, ValueFunction, heuristic_value

__all__ = [
    "PolicyWeights",
    "lookahead_policy",
    "make_search",
    "make_search_weights",
    "search_policy",
]

_ILLEGAL = -1e9  # prior logit for illegal moves

_Weights = Float[Array, f"flat={N_FLAT}"]  # one tree's improved-policy weights

PolicyWeights = Callable[
    [KeyScalar, BoardLayout, BeliefView, IntScalar, FlatMask], _Weights
]
"""The search's improved-policy weights over the flat actions — a distribution
for ``num_simulations`` > 0, the lookahead logits at 0. The AlphaZero policy
target (see :func:`make_search_weights`)."""

# Interior-node prior: greedy's static tier table, tempered so tier gaps
# (>= 100) land ~2 nats apart — strong enough to order first expansions, weak
# enough for a few backed-up values to override. Trade proposals are excluded
# from the interior prior outright: under the paranoid two-sided frame the
# in-tree responder prices every offer as rejected-or-harmful, so their ~100
# near-tied rows would only flood the candidate pool — the search answers trades
# through search but only ever *offers* one at the root (see make_search).
_NO_PROPOSE = jnp.where(_ROW_TYPE == ActionType.PROPOSE_TRADE, _ILLEGAL, 0.0)
_TIER_LOGITS = TIER_SCORES / 50.0 + _NO_PROPOSE

# Two-dice outcomes and their probabilities.
_ROLLS = jnp.arange(2, 13)
_ROLL_P = jnp.asarray([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=jnp.float32) / 36.0

# Absolute completed-Q scaling: without the min-max rescale the search only
# overrides a prior gap in proportion to the backed-up value difference.
_QTRANSFORM = functools.partial(
    mctx.qtransform_completed_by_mix_value, rescale_values=False
)

# Trade-proposal rows and their partners (the low bits of the packed target —
# see trade.pack_trade): a proposal's successor is material-neutral (cards move
# only on the partner's accept), so the root scores them by their *accepted*
# outcome under a partner model.
_PROPOSE_ROWS = jnp.asarray(
    np.flatnonzero(np.asarray(_ROW_TYPE) == int(ActionType.PROPOSE_TRADE))
)
_PROPOSE_PARTNER = _ROW_PARAMS.target[_PROPOSE_ROWS] & ((1 << _PARTNER_BITS) - 1)
_ACCEPT = jnp.int32(ActionType.ACCEPT_TRADE)
_NO_PARAMS = ActionParams(idx=jnp.int32(0), target=jnp.int32(0))


def _terminal(state: BoardState) -> BoolScalar:
    """Whether the game is over: its current player has won (only the turn's
    owner can win — mirrors the engine's ``awards.current_player_won``)."""
    cur = state.current_player.astype(jnp.int32)
    return player_total_vp(state, cur) >= VICTORY_POINTS_TO_WIN


def _winner(state: BoardState) -> IntScalar:
    """The winner of a terminal state: its current player."""
    return state.current_player.astype(jnp.int32)


class _Transition(NamedTuple):
    """One batched tree step, in mctx's frame conventions."""

    next_state: BoardState
    next_mover: Int[Array, "batch"]
    reward: Float[Array, "batch"]  # +/-1 to the acting side on a winning transition
    discount: Float[Array, "batch"]  # -1 crossing the side boundary, 0 into terminals
    prior_logits: Float[Array, "batch flat"]  # tier table over the child's legal moves
    roll_child: Bool[Array, "batch"]  # lanes backing up the roll expectation
    terminal: Bool[Array, "batch"]


def _transition(
    layout: BoardLayout,
    state: BoardState,
    action: Int[Array, "batch"],
    player: IntScalar,
) -> _Transition:
    """Apply one flat action per lane; value-free dynamics shared by all trees.

    Frames are two-sided — the searching ``player`` vs the rest of the table
    (the *paranoid* reduction; exact zero-sum at 2 players, identical there to
    flipping on every mover change). Reward is in the acting side's frame and
    the discount flips only when the move crosses the side boundary, so all
    opponents share one frame and the searcher's own later turns never come
    back negated (at 3-4 players the every-mover-flip rule negates them on odd
    cycles).
    """
    atype, aparams = flat_to_action(action)
    mover = jax.vmap(agent_selection_single)(state)
    was_terminal = jax.vmap(_terminal)(state)
    # Gating with ~terminal makes won states absorbing (INVALID = no-op).
    avail = available((layout, state), atype, aparams) & ~was_terminal
    next_state, _ = jax.vmap(apply_action)(layout, state, atype, aparams, avail)
    now_terminal = jax.vmap(_terminal)(next_state)
    won = now_terminal & ~was_terminal
    next_mover = jax.vmap(agent_selection_single)(next_state)
    mover_side = mover == player
    next_side = next_mover == player
    winner_side = jax.vmap(_winner)(next_state) == player
    return _Transition(
        next_state=next_state,
        next_mover=next_mover,
        reward=jnp.where(won, jnp.where(winner_side == mover_side, 1.0, -1.0), 0.0),
        discount=jnp.where(
            now_terminal, 0.0, jnp.where(next_side == mover_side, 1.0, -1.0)
        ),
        prior_logits=jnp.where(
            flat_available((layout, next_state)), _TIER_LOGITS, _ILLEGAL
        ),
        roll_child=(atype == ActionType.ROLL_DICE) & avail,
        terminal=now_terminal,
    )


class _Path(NamedTuple):
    """The embedding: the action path from the root, replayed each simulation
    under a freshly sampled world. ``depth`` is how many entries of ``actions``
    are live (the rest are padding to the fixed history length)."""

    actions: Int[Array, "batch depth"]
    depth: Int[Array, "batch"]


def make_search_weights(
    value: ValueFunction,
    *,
    prior: PolicyPrior | None = None,
    num_trees: int = 1,
    num_simulations: int = 32,
    max_depth: int = 12,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
    propose_rate: float = 0.0,
    trade_penalty: float = 0.25,
) -> PolicyWeights:
    """Re-determinizing Gumbel-MuZero search, returning the improved-policy
    weights (the AlphaZero policy target; :func:`make_search` argmaxes these).

    Each of ``num_trees`` independent trees (averaged for variance reduction)
    re-determinizes once per simulation: a simulation replays the root path
    under a freshly sampled world (``max_depth`` bounds the replayed history),
    so leaf values integrate over the belief. ``value`` drives the leaf
    (``tanh(value / value_scale)``, commensurate with the +/-1 terminal reward)
    and, when ``prior`` is None, the root and interior priors (one-step sweep
    over ``prior_scale``; interior nodes a static tier table). A ``prior``
    replaces both with learned logits (legality-masked here).

    ``num_simulations=0`` is the *lookahead* special case: no tree, the root
    one-step value sweep over ``num_trees`` sampled worlds. ``propose_rate`` > 0
    (the search never offers trades by default)
    lets the root score trade proposals by their accepted outcome under a
    partner model — gated geometrically per move, ``trade_penalty`` the quality
    bar below which an offer loses to not trading; offers are root-only.
    """

    def leaf_value(layout: BoardLayout, state: BoardState, p: IntScalar) -> Value:
        return jnp.tanh(value(layout, state, p) / value_scale)

    def expected_roll_value(
        layout: BoardLayout, state: BoardState, p: IntScalar
    ) -> Value:
        """E over the 11 rolls of the post-payout value of a pre-roll state.

        The 7 row distributes nothing, so it values the state as-is — the
        pending discard/robber resolution is approximated away.
        """
        vals = jax.vmap(
            lambda r: leaf_value(layout, distribute_resources(layout, state, r), p)
        )(_ROLLS)
        return _ROLL_P @ vals

    def root_logits(
        key: KeyScalar,
        layout: BoardLayout,
        world: BoardState,
        player: IntScalar,
        mask: FlatMask,
    ) -> _Weights:
        """The root prior over one concrete world: the one-step value sweep
        (proposals excluded), optionally with trade offers priced in."""
        if prior is not None:
            return prior(layout, world, player)
        successors, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, world, _ROW_TYPE, _ROW_PARAMS, mask
        )
        values = jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)
        logits = values / prior_scale + _NO_PROPOSE
        if propose_rate <= 0.0:
            return logits
        # Score each proposal by its accepted outcome, if the modeled partner
        # would take it and proposing is allowed this move (see trade_penalty).
        offered = jax.tree.map(lambda x: x[_PROPOSE_ROWS], successors)
        accepted = jax.vmap(_accepted_outcome, in_axes=(None, 0))(layout, offered)
        mine = jax.vmap(value, in_axes=(None, 0, None))(layout, accepted, player)
        partner_reject = jax.vmap(value, in_axes=(None, 0, 0))(
            layout, offered, _PROPOSE_PARTNER
        )
        partner_accept = jax.vmap(value, in_axes=(None, 0, 0))(
            layout, accepted, _PROPOSE_PARTNER
        )
        allowed = jax.random.uniform(key) < propose_rate
        propose_score = jnp.where(
            (partner_accept > partner_reject) & allowed, mine - trade_penalty, _ILLEGAL
        )
        return logits.at[_PROPOSE_ROWS].set(propose_score / prior_scale)

    def search_tree(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> _Weights:
        """Improved-policy weights from one re-determinizing tree."""
        layout_b = jax.tree.map(lambda x: x[None], layout)
        k_root, k_gate, k_search = jax.random.split(key, 3)

        def replay(rng: KeyScalar, path: _Path) -> BoardState:
            """The parent node's state, replaying ``path`` under a fresh world.

            Looped to the path's *own* depth, not ``max_depth``: at ~32 sims the
            tree is a few plies, so replaying the fixed ``max_depth`` history spent
            most steps on masked no-ops (a no-op ``apply_action`` still runs the
            full transition). The body still guards ``i < depth``, so the dynamic
            bound is output-identical -- it just skips the dead tail.
            """
            world = jax.vmap(sample_world, in_axes=(0, None, None))(
                jax.random.split(rng, 1), view, player
            )

            def step(i: Array, state: BoardState) -> BoardState:
                atype, aparams = flat_to_action(path.actions[:, i])
                live = (i < path.depth) & ~jax.vmap(_terminal)(state)
                avail = available((layout_b, state), atype, aparams) & live
                nxt, _ = jax.vmap(apply_action)(layout_b, state, atype, aparams, avail)
                return nxt

            depth = jnp.minimum(path.depth.reshape(()), max_depth)
            return cast(BoardState, jax.lax.fori_loop(0, depth, step, world))

        def recurrent_fn(
            params: None, rng: KeyScalar, action: Array, embedding: _Path
        ) -> tuple[mctx.RecurrentFnOutput, _Path]:
            state = replay(rng, embedding)
            t = _transition(layout_b, state, action, player)
            sign = jnp.where(t.next_mover == player, 1.0, -1.0)
            v = jax.vmap(leaf_value, in_axes=(0, 0, None))(
                layout_b, t.next_state, player
            )
            v = jnp.where(
                t.roll_child,
                jax.vmap(expected_roll_value, in_axes=(0, 0, None))(
                    layout_b, state, player
                ),
                v,
            )
            v = jnp.where(t.terminal, 0.0, sign * v)
            logits = t.prior_logits
            if prior is not None:
                logits = jnp.where(
                    t.prior_logits > _ILLEGAL,
                    jax.vmap(prior, in_axes=(0, 0, None))(
                        layout_b, t.next_state, player
                    ),
                    _ILLEGAL,
                )
            out = mctx.RecurrentFnOutput(  # type: ignore[call-arg]  # chex dataclass
                reward=t.reward,
                discount=t.discount,
                prior_logits=logits,
                value=v,
            )
            idx = jnp.minimum(embedding.depth, max_depth - 1)
            new_actions = embedding.actions.at[jnp.arange(1), idx].set(action)
            new_depth = jnp.minimum(embedding.depth + 1, max_depth)
            return out, _Path(new_actions, new_depth)

        # Root prior + value use one sampled world (the prior only orders the
        # first expansions; per-simulation resampling carries the rest).
        root_world = sample_world(k_root, view, player)
        logits = jnp.where(
            mask, root_logits(k_gate, layout, root_world, player, mask), _ILLEGAL
        )
        root = mctx.RootFnOutput(  # type: ignore[call-arg]  # chex dataclass
            prior_logits=logits[None],
            value=leaf_value(layout, root_world, player)[None],
            embedding=_Path(
                actions=jnp.full((1, max_depth), -1, dtype=jnp.int32),
                depth=jnp.zeros((1,), dtype=jnp.int32),
            ),
        )
        out = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=k_search,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=(~mask)[None],
            max_num_considered_actions=max_num_considered_actions,
            qtransform=_QTRANSFORM,
        )
        return cast(_Weights, out.action_weights[0])

    def lookahead(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> _Weights:
        """The root one-step sweep over one sampled world (num_simulations=0)."""
        k_world, k_gate = jax.random.split(key)
        world = sample_world(k_world, view, player)
        return jnp.where(
            mask, root_logits(k_gate, layout, world, player, mask), _ILLEGAL
        )

    def weights(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> _Weights:
        tree = lookahead if num_simulations == 0 else search_tree
        w = jax.vmap(tree, in_axes=(0, None, None, None, None))(
            jax.random.split(key, num_trees), layout, view, player, mask
        )
        return w.mean(axis=0)

    return weights


def make_search(
    value: ValueFunction,
    *,
    prior: PolicyPrior | None = None,
    num_trees: int = 1,
    num_simulations: int = 32,
    max_depth: int = 12,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
    propose_rate: float = 0.0,
    trade_penalty: float = 0.25,
) -> BeliefPolicy:
    """Re-determinizing search as a :class:`BeliefPolicy`: the masked argmax of
    the improved policy. Parameters are :func:`make_search_weights`'; tiny noise
    breaks the lookahead sweep's exact-value ties."""
    weights = make_search_weights(
        value,
        prior=prior,
        num_trees=num_trees,
        num_simulations=num_simulations,
        max_depth=max_depth,
        max_num_considered_actions=max_num_considered_actions,
        value_scale=value_scale,
        prior_scale=prior_scale,
        propose_rate=propose_rate,
        trade_penalty=trade_penalty,
    )

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        noise = jax.random.uniform(key, (N_FLAT,)) * 1e-4
        w = weights(key, layout, view, player, mask)
        return jnp.argmax(jnp.where(mask, w + noise, -jnp.inf))

    return policy


def _accepted_outcome(layout: BoardLayout, succ: BoardState) -> BoardState:
    """The state after the partner accepts ``succ``'s pending proposal.

    ``succ`` unchanged when accepting is illegal there (no pending trade, or
    the partner lacks the asked-for card in this world).
    """
    avail = action_available(layout, succ, _ACCEPT, _NO_PARAMS)
    accepted, _ = apply_action(layout, succ, _ACCEPT, _NO_PARAMS, avail)
    return accepted


search_policy = make_search(heuristic_value)
"""The shipped search: re-determinizing Gumbel-MuZero over :func:`heuristic_value`."""

lookahead_policy = make_search(heuristic_value, num_simulations=0, propose_rate=0.5)
"""One-step lookahead — the ``num_simulations=0`` special case of the search."""
