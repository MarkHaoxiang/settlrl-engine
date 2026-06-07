"""MCTS over the engine: mctx search with a value function at the leaves."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import mctx

from catan_engine.belief import PlayerBelief
from catan_engine.board import Board
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import VICTORY_POINTS_TO_WIN, BoardState, IntScalar
from catan_engine.env import available, flat_available, flat_to_action
from catan_engine.mechanics.action import apply_action
from catan_engine.mechanics.common import agent_selection_single, player_total_vp

from catan_agents.shared.policy import BeliefPolicy, FlatAction, FlatMask
from catan_agents.shared.sample import sample_world
from catan_agents.shared.value import ValueFunction, heuristic_value

_ILLEGAL = -1e9  # prior logit for illegal moves


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


def make_mcts(
    value: ValueFunction,
    *,
    num_simulations: int = 32,
    max_num_considered_actions: int = 16,
    value_scale: float = 20.0,
) -> BeliefPolicy:
    """Gumbel-MuZero search using the engine itself as the dynamics model.

    Each simulation expands one node: the chosen flat action is applied with
    :func:`apply_action`, the child's legal moves become its prior, and
    ``tanh(value / value_scale)`` evaluated for the child's player-to-move is
    its leaf value. Transitions discount by -1 when the player-to-move
    switches and a win backs up as a +/-1 reward into an absorbing terminal —
    exact zero-sum framing for two players, the *paranoid* reduction (every
    opponent maximizes against the mover) beyond. The censored root is made
    concrete with one :func:`~catan_agents.shared.sample.sample_world` draw,
    so the search runs in a world consistent with what the seat knows and
    samples its own dice / steals / dev draws.
    """

    def leaf_value(layout: BoardLayout, state: BoardState, p: jax.Array) -> jax.Array:
        return jnp.tanh(value(layout, state, p) / value_scale)

    def recurrent_fn(
        params: None, rng: jax.Array, action: jax.Array, embedding: Board
    ) -> tuple[mctx.RecurrentFnOutput, Board]:
        layout, state = embedding
        atype, aparams = flat_to_action(action)
        mover = jax.vmap(agent_selection_single)(state)
        was_terminal = jax.vmap(_terminal)(state)
        # Gating with ~terminal makes won states absorbing (INVALID = no-op).
        avail = available(embedding, atype, aparams) & ~was_terminal
        next_state, _ = jax.vmap(apply_action)(layout, state, atype, aparams, avail)
        now_terminal = jax.vmap(_terminal)(next_state)
        won = now_terminal & ~was_terminal
        # Reward in the acting player's frame; value in the next mover's frame;
        # the discount flips the frame whenever the player-to-move changes.
        reward = jnp.where(
            won, jnp.where(jax.vmap(_winner)(next_state) == mover, 1.0, -1.0), 0.0
        )
        next_mover = jax.vmap(agent_selection_single)(next_state)
        discount = jnp.where(
            now_terminal, 0.0, jnp.where(next_mover == mover, 1.0, -1.0)
        )
        prior = jnp.where(flat_available((layout, next_state)), 0.0, _ILLEGAL)
        v = jnp.where(
            now_terminal, 0.0, jax.vmap(leaf_value)(layout, next_state, next_mover)
        )
        out = mctx.RecurrentFnOutput(  # type: ignore[call-arg]  # chex dataclass
            reward=reward, discount=discount, prior_logits=prior, value=v
        )
        return out, (layout, next_state)

    def policy(
        key: jax.Array,
        layout: BoardLayout,
        state: BoardState,
        belief: PlayerBelief,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        k_world, k_search = jax.random.split(key)
        state = sample_world(k_world, state, belief, player)
        batched: Any = jax.tree.map(lambda x: x[None], (layout, state))
        root = mctx.RootFnOutput(  # type: ignore[call-arg]  # chex dataclass
            prior_logits=jnp.where(mask, 0.0, _ILLEGAL)[None],
            value=leaf_value(layout, state, player)[None],
            embedding=batched,
        )
        out = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=k_search,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=(~mask)[None],
            max_num_considered_actions=max_num_considered_actions,
        )
        return cast(jax.Array, out.action[0])

    return policy


mcts_policy = make_mcts(heuristic_value)
"""The MCTS agent: Gumbel-MuZero search over :func:`heuristic_value`."""
