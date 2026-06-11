"""MCTS over the engine: mctx search with a value function at the leaves."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import mctx
from catan_engine.belief import BeliefView
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import (
    VICTORY_POINTS_TO_WIN,
    BoardState,
    BoolScalar,
    IntScalar,
    KeyArray,
    KeyScalar,
)
from catan_engine.env import N_FLAT, available, flat_available, flat_to_action
from catan_engine.mechanics.action import ActionType, apply_action
from catan_engine.mechanics.common import agent_selection_single, player_total_vp
from catan_engine.mechanics.dice import distribute_resources
from jaxtyping import Array, Bool, Float, Int, UInt8

from catan_agents.shared.greedy import _BASE
from catan_agents.shared.policy import BeliefPolicy, FlatAction, FlatMask, PolicyPrior
from catan_agents.shared.sample import sample_world
from catan_agents.shared.value import Value, ValueFunction, heuristic_value

_ILLEGAL = -1e9  # prior logit for illegal moves

_Weights = Float[Array, f"flat={N_FLAT}"]  # one tree's improved-policy weights

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


# Everything on BoardState except the PRNG key is uint8, so a state packs into
# one row per game. mctx gathers/scatters every embedding leaf into its
# (batch, nodes, ...) storage on each simulation; two leaves instead of ~25
# keeps that launch-latency-bound traffic off the per-simulation clock.
_U8_FIELDS = tuple(f for f in BoardState._fields if f != "key")


class _Packed(NamedTuple):
    """A ``BoardState`` flattened for mctx node storage."""

    data: UInt8[Array, "batch packed"]
    key: KeyArray


def _codec(
    template: BoardState,
) -> tuple[Callable[[BoardState], _Packed], Callable[[_Packed], BoardState]]:
    """Pack/unpack for batched states whose games are shaped like ``template``
    (one game, no batch axis). Round-trips exactly; a non-uint8 ``BoardState``
    field fails here loudly rather than packing lossily."""
    shapes = [getattr(template, f).shape for f in _U8_FIELDS]
    bad = [f for f in _U8_FIELDS if getattr(template, f).dtype != jnp.uint8]
    if bad:
        raise TypeError(f"non-uint8 BoardState fields cannot pack: {bad}")
    sizes = [math.prod(s) for s in shapes]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]

    def pack(state: BoardState) -> _Packed:
        b = state.phase.shape[0]
        return _Packed(
            jnp.concatenate(
                [getattr(state, f).reshape(b, -1) for f in _U8_FIELDS], axis=1
            ),
            state.key,
        )

    def unpack(packed: _Packed) -> BoardState:
        b = packed.data.shape[0]
        return BoardState(
            key=packed.key,
            **{
                f: packed.data[:, o : o + n].reshape(b, *shp)
                for f, o, n, shp in zip(_U8_FIELDS, offsets, sizes, shapes, strict=True)
            },
        )

    return pack, unpack


def _terminal(state: BoardState) -> BoolScalar:
    """Whether any player has won (single game)."""
    players = jnp.arange(state.n_players)
    totals = jax.vmap(lambda p: player_total_vp(state, p))(players)
    return jnp.any(totals >= VICTORY_POINTS_TO_WIN)


def _winner(state: BoardState) -> IntScalar:
    """The player with the highest VP total (single game)."""
    players = jnp.arange(state.n_players)
    totals = jax.vmap(lambda p: player_total_vp(state, p))(players)
    return jnp.argmax(totals)


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


def make_mcts(
    value: ValueFunction,
    *,
    prior: PolicyPrior | None = None,
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
    argmax. ``value`` drives the leaf evaluation (``tanh(value /
    value_scale)``, commensurate with the ±1 terminal reward) and, when
    ``prior`` is None, the root prior too (one-step sweep divided by
    ``prior_scale``; interior nodes use a static tier table). A ``prior``
    supplies the root's and every interior node's logits instead
    (legality-masked here) — the seam for learned policy heads.
    """

    # --- evaluation: everything below the engine step that needs `value` ---

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

    # --- search: one tree over one concrete world ---

    def search_world(
        key: KeyScalar,
        layout: BoardLayout,
        state: BoardState,
        player: IntScalar,
        mask: FlatMask,
    ) -> _Weights:
        """Improved-policy weights from one search of a single concrete world."""
        # The layout never changes inside a search, so it lives in the closure
        # (batch axis of 1, mctx's in-tree batch) instead of the embedding.
        pack, unpack = _codec(state)
        layout_b = jax.tree.map(lambda x: x[None], layout)

        # Every node's value is the searcher's, signed into the mover's side
        # of the two-sided frame (see _transition).
        def recurrent_fn(
            params: None,
            rng: KeyScalar,
            action: Int[Array, "batch"],
            embedding: _Packed,
        ) -> tuple[mctx.RecurrentFnOutput, _Packed]:
            state = unpack(embedding)
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
            return out, pack(t.next_state)

        if prior is None:
            # Heuristic root prior: the one-step value sweep over legal moves.
            successors, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
                layout, state, _ROW_TYPE, _ROW_PARAMS, mask
            )
            root_logits = (
                jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)
                / prior_scale
            )
        else:
            root_logits = prior(layout, state, player)
        root = mctx.RootFnOutput(  # type: ignore[call-arg]  # chex dataclass
            prior_logits=jnp.where(mask, root_logits, _ILLEGAL)[None],
            value=leaf_value(layout, state, player)[None],
            embedding=pack(jax.tree.map(lambda x: x[None], state)),
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
        return cast(_Weights, out.action_weights[0])

    # --- ensemble: sample worlds, fan out futures, average the trees ---

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
        # Re-keying each replica is what makes its in-tree chance draws differ.
        states = states._replace(key=jax.random.split(k_future, n_trees))
        weights = jax.vmap(search_world, in_axes=(0, None, 0, None, None))(
            jax.random.split(k_search, n_trees), layout, states, player, mask
        )
        return jnp.argmax(jnp.where(mask, weights.mean(axis=0), -jnp.inf))

    return policy


mcts_policy = make_mcts(heuristic_value)
"""The MCTS agent: Gumbel-MuZero search over :func:`heuristic_value`."""
