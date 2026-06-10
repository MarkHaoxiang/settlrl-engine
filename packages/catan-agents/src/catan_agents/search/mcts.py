"""MCTS over the engine: mctx search with a value function at the leaves."""

from __future__ import annotations

import functools
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import mctx
from catan_engine.belief import BeliefView
from catan_engine.board import Board
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import VICTORY_POINTS_TO_WIN, BoardState, IntScalar
from catan_engine.env import N_FLAT, available, flat_available, flat_to_action
from catan_engine.mechanics.action import ActionType, apply_action
from catan_engine.mechanics.common import agent_selection_single, player_total_vp
from catan_engine.mechanics.dice import distribute_resources

from catan_agents.shared.greedy import _BASE
from catan_agents.shared.policy import BeliefPolicy, FlatAction, FlatMask
from catan_agents.shared.sample import sample_world
from catan_agents.shared.value import ValueFunction, heuristic_value

_ILLEGAL = -1e9  # prior logit for illegal moves

# Static decode of every flat row, for the root's one-step value sweep.
_ROW_TYPE, _ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))

# Interior-node prior: greedy's static tier table, tempered so tier gaps
# (>= 100) land ~2 nats apart — strong enough to order first expansions,
# weak enough for a few backed-up values to override.
_TIER_LOGITS = _BASE / 50.0

# Two-dice outcomes and their probabilities.
_ROLLS = jnp.arange(2, 13)
_ROLL_P = jnp.asarray([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=jnp.float32) / 36.0

# Absolute completed-Q scaling: without the min-max rescale the search only
# overrides a prior gap in proportion to the backed-up value difference.
_QTRANSFORM = functools.partial(
    mctx.qtransform_completed_by_mix_value, rescale_values=False
)


def _terminal(state: BoardState) -> jax.Array:
    """Whether any player has won (single game)."""
    players = jnp.arange(state.n_players)
    totals = jax.vmap(lambda p: player_total_vp(state, p))(players)
    return jnp.any(totals >= VICTORY_POINTS_TO_WIN)


def _winner(state: BoardState) -> jax.Array:
    """The player with the highest VP total (single game)."""
    players = jnp.arange(state.n_players)
    totals = jax.vmap(lambda p: player_total_vp(state, p))(players)
    return jnp.argmax(totals)


class _Transition(NamedTuple):
    """One batched tree step, in mctx's frame conventions."""

    next_state: BoardState
    next_mover: jax.Array
    reward: jax.Array  # +/-1 to the acting player on a winning transition
    discount: jax.Array  # -1 mover switch, +1 same mover, 0 into terminals
    prior_logits: jax.Array  # tier table over the child's legal moves
    roll_child: jax.Array  # lanes whose child should back up the roll expectation
    terminal: jax.Array


def _transition(
    layout: BoardLayout, state: BoardState, action: jax.Array
) -> _Transition:
    """Apply one flat action per lane; value-free dynamics shared by all trees.

    Reward is in the acting player's frame, the child's value belongs to its
    own player-to-move, and the discount flips the frame whenever the mover
    changes — exact zero-sum at 2 players, the *paranoid* reduction beyond.
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
    return _Transition(
        next_state=next_state,
        next_mover=next_mover,
        reward=jnp.where(
            won, jnp.where(jax.vmap(_winner)(next_state) == mover, 1.0, -1.0), 0.0
        ),
        discount=jnp.where(
            now_terminal, 0.0, jnp.where(next_mover == mover, 1.0, -1.0)
        ),
        prior_logits=jnp.where(
            flat_available((layout, next_state)), _TIER_LOGITS, _ILLEGAL
        ),
        roll_child=(atype == ActionType.ROLL_DICE) & avail,
        terminal=now_terminal,
    )


def make_mcts(
    value: ValueFunction,
    *,
    num_worlds: int = 4,
    num_futures: int = 1,
    num_simulations: int = 32,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
    prior_scale: float = 1.0,
) -> BeliefPolicy:
    """Gumbel-MuZero search using the engine itself as the dynamics model.

    Searches ``num_worlds * num_futures`` independent trees per move —
    ``num_worlds`` :func:`~catan_agents.shared.sample.sample_world`
    determinizations of the view, ``num_futures`` chance re-keyings of each —
    and averages their improved-policy weights before the final masked
    argmax. ``value`` drives both the root prior (one-step sweep, divided by
    ``prior_scale``) and the leaf evaluation (``tanh(value / value_scale)``,
    commensurate with the ±1 terminal reward).
    """

    # --- evaluation: everything below the engine step that needs `value` ---

    def leaf_value(layout: BoardLayout, state: BoardState, p: jax.Array) -> jax.Array:
        return jnp.tanh(value(layout, state, p) / value_scale)

    def expected_roll_value(
        layout: BoardLayout, state: BoardState, p: jax.Array
    ) -> jax.Array:
        """E over the 11 rolls of the post-payout value of a pre-roll state.

        The 7 row distributes nothing, so it values the state as-is — the
        pending discard/robber resolution is approximated away.
        """
        vals = jax.vmap(
            lambda r: leaf_value(layout, distribute_resources(layout, state, r), p)
        )(_ROLLS)
        return _ROLL_P @ vals

    def recurrent_fn(
        params: None, rng: jax.Array, action: jax.Array, embedding: Board
    ) -> tuple[mctx.RecurrentFnOutput, Board]:
        layout, state = embedding
        t = _transition(layout, state, action)
        v = jax.vmap(leaf_value)(layout, t.next_state, t.next_mover)
        v = jnp.where(
            t.roll_child,
            jax.vmap(expected_roll_value)(layout, state, t.next_mover),
            v,
        )
        v = jnp.where(t.terminal, 0.0, v)
        out = mctx.RecurrentFnOutput(  # type: ignore[call-arg]  # chex dataclass
            reward=t.reward, discount=t.discount, prior_logits=t.prior_logits, value=v
        )
        return out, (layout, t.next_state)

    # --- search: one tree over one concrete world ---

    def search_world(
        key: jax.Array,
        layout: BoardLayout,
        state: BoardState,
        player: IntScalar,
        mask: FlatMask,
    ) -> jax.Array:
        """Improved-policy weights from one search of a single concrete world."""
        # Heuristic root prior: the one-step value sweep over all legal moves.
        successors, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, _ROW_TYPE, _ROW_PARAMS, mask
        )
        root_vals = jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)
        batched: Any = jax.tree.map(lambda x: x[None], (layout, state))
        root = mctx.RootFnOutput(  # type: ignore[call-arg]  # chex dataclass
            prior_logits=jnp.where(mask, root_vals / prior_scale, _ILLEGAL)[None],
            value=leaf_value(layout, state, player)[None],
            embedding=batched,
        )
        out = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=(~mask)[None],
            max_num_considered_actions=max_num_considered_actions,
            qtransform=_QTRANSFORM,
        )
        return cast(jax.Array, out.action_weights[0])

    # --- ensemble: sample worlds, fan out futures, average the trees ---

    def policy(
        key: jax.Array,
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
        # Re-keying each replica is what makes its in-tree chance draws differ.
        states = states._replace(key=jax.random.split(k_future, n_trees))
        weights = jax.vmap(search_world, in_axes=(0, None, 0, None, None))(
            jax.random.split(k_search, n_trees), layout, states, player, mask
        )
        return jnp.argmax(jnp.where(mask, weights.mean(axis=0), -jnp.inf))

    return policy


mcts_policy = make_mcts(heuristic_value)
"""The MCTS agent: Gumbel-MuZero search over :func:`heuristic_value`."""
