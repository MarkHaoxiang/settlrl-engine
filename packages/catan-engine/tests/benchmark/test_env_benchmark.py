"""pytest-benchmark throughput benchmarks for the RL envs under random play.

Two rollouts driven by uniformly-random *legal* actions:

- ``BatchedCatanEnv`` -- a batch of games stepped in lockstep (the vectorised
  surface); one random legal action per lane per step.
- ``CatanAECEnv`` -- a single game, turn at a time (the PettingZoo surface).

Legality comes from the engine itself: every candidate action is screened with
the engine's own ``available`` / action mask, so the rollouts always make
progress and exercise every action type (including the forced DISCARD /
MOVE_ROBBER after a 7). JIT compilation is warmed up before the timed region.

These need the ``rl`` extra (``pettingzoo`` / ``gymnasium``) for the flat action
table shared with the AEC wrapper.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.board import Board
from catan_engine.board.layout import BoardLayout
from catan_engine.board.resources import N_RESOURCES
from catan_engine.board.state import BoardState
from catan_engine.env.aec import (
    _ATYPE,
    _DISCARD_FLAT,
    _IDX,
    _N_FLAT,
    _TARGET,
    CatanAECEnv,
    _canonical_discard,
)
from catan_engine.env.batched import BatchedCatanEnv, available
from catan_engine.mechanics.action import ActionParams


def _repeat_board(board: Board, repeats: int) -> Board:
    """Repeat each lane ``repeats`` times along the batch axis (B -> B*repeats)."""
    layout, state = board
    layout_r = BoardLayout(*(jnp.repeat(x, repeats, axis=0) for x in layout))
    fields: dict[str, Any] = {}
    for name in state._fields:
        v = getattr(state, name)
        if name == "key":
            fields[name] = jax.random.wrap_key_data(
                jnp.repeat(jax.random.key_data(v), repeats, axis=0)
            )
        else:
            fields[name] = jnp.repeat(v, repeats, axis=0)
    return layout_r, BoardState(**fields)


def _random_batched_actions(
    env: BatchedCatanEnv, rng: np.random.Generator
) -> tuple[jax.Array, ActionParams]:
    """Pick a uniformly-random legal action per lane via full mask enumeration.

    Evaluates every flat action (the AEC action table) against every lane with
    one batched ``available`` call, then samples a legal one per lane.
    """
    layout, state = env.board
    B = env.batch_size
    sel = np.asarray(env.agent_selection)  # (B,) acting player per lane

    # Flat-action params are state-independent except DISCARD, whose player /
    # amounts depend on the acting lane -- fill that column in per lane.
    idx = np.broadcast_to(_IDX, (B, _N_FLAT)).copy()
    target = np.broadcast_to(_TARGET, (B, _N_FLAT)).copy()
    resources = np.zeros((B, _N_FLAT, N_RESOURCES), dtype=np.int32)
    hands = np.asarray(state.player_resources)  # (B, P, R)
    pending = np.asarray(state.pending_discard)  # (B, P)
    for b in range(B):
        s = int(sel[b])
        idx[b, _DISCARD_FLAT] = s
        resources[b, _DISCARD_FLAT] = _canonical_discard(
            hands[b, s], int(pending[b, s])
        )

    rep = _repeat_board((layout, state), _N_FLAT)
    atype_flat = jnp.asarray(np.broadcast_to(_ATYPE, (B, _N_FLAT)).reshape(-1))
    params_flat = ActionParams(
        idx=jnp.asarray(idx.reshape(-1)),
        target=jnp.asarray(target.reshape(-1)),
        resources=jnp.asarray(resources.reshape(B * _N_FLAT, N_RESOURCES)),
    )
    mask = np.asarray(available(rep, atype_flat, params_flat)).reshape(B, _N_FLAT)

    # Random legal action per lane (illegal -> -1, so argmax never picks one
    # unless the lane has no legal action at all, in which case the resulting
    # action is INVALID and the lane simply stalls until its next auto-reset).
    scores = np.where(mask, rng.random((B, _N_FLAT)), -1.0)
    chosen = scores.argmax(axis=1)
    rows = np.arange(B)
    sel_atype = jnp.asarray(_ATYPE[chosen].astype(np.int32))
    sel_params = ActionParams(
        idx=jnp.asarray(idx[rows, chosen].astype(np.int32)),
        target=jnp.asarray(target[rows, chosen].astype(np.int32)),
        resources=jnp.asarray(resources[rows, chosen]),
    )
    return sel_atype, sel_params


def _batched_rollout(seed: int, batch_size: int, steps: int) -> None:
    env = BatchedCatanEnv(batch_size=batch_size, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        action_type, params = _random_batched_actions(env, rng)
        env.step(action_type, params)
    np.asarray(env.board[1].phase)  # force device->host sync so timing is honest


def _aec_rollout(seed: int, steps: int) -> None:
    e = CatanAECEnv(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        agent = e.agent_selection
        if e.terminations[agent] or e.truncations[agent]:
            e.reset(seed)
            agent = e.agent_selection
        legal = np.where(e.observe(agent)["action_mask"])[0]
        if legal.size == 0:  # only a terminal game has no legal move
            e.reset(seed)
            continue
        e.step(int(rng.choice(legal)))


def test_batched_env_random_rollout(benchmark: Any) -> None:
    """Throughput of a batch of games stepped with random legal actions."""
    batch_size, steps = 8, 40
    _batched_rollout(seed=0, batch_size=batch_size, steps=steps)  # warm up JIT
    benchmark(lambda: _batched_rollout(seed=0, batch_size=batch_size, steps=steps))


def test_aec_env_random_rollout(benchmark: Any) -> None:
    """Throughput of a single PettingZoo-AEC game under random legal play."""
    steps = 80
    _aec_rollout(seed=0, steps=steps)  # warm up JIT
    benchmark(lambda: _aec_rollout(seed=0, steps=steps))
