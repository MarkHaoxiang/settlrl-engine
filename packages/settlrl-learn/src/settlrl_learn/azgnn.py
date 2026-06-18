"""AlphaZero with a GNN trunk (the experiment-0003 ``gn_global`` net).

The GNN reads the board *graph* (``settlrl_learn.graph.board_sample``), not the
flat engineered vector, so it needs its own self-play / training path: this
module mirrors :mod:`settlrl_learn.selfplay` + :mod:`settlrl_learn.alphazero` for
an equinox :class:`~settlrl_learn.graphnet.GraphNet` with a shared trunk feeding
a value head (win-prob logit, ``value_scale=2``) and an ``N_FLAT`` policy head.

Self-play -> a flashbax on-device replay -> train -> periodic arena vs
``lookahead(heuristic)``. The whole :class:`GNNState` is eqx-serialised every
iteration for bit-exact resume (eqx's native serialiser fits the equinox model,
where orbax's pure-array assumption does not).

A training-side module (equinox/jraph/optax/flashbax): not imported by the root.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int
from settlrl_agents import POLICIES, BeliefSpec, evaluate
from settlrl_agents.policy import PolicyPrior
from settlrl_agents.search import make_search, make_search_weights
from settlrl_agents.value import Value, ValueFunction
from settlrl_engine.belief import belief_view
from settlrl_engine.board.layout import EDGE_V, N_VERTICES, TILE_V, BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar
from settlrl_engine.env import N_FLAT, BatchedSettlrlEnv, flat_to_action

from settlrl_learn import action_layout as al
from settlrl_learn.features import FEATURE_DIM
from settlrl_learn.graph import (
    EDGE_DIM,
    GLOBAL_DIM,
    N_DIR_EDGES,
    NODE_DIM,
    Sample,
    board_sample,
)
from settlrl_learn.graphnet import PRESETS, GraphNetConfig, GraphTrunk, readout_dim

# undirected-edge endpoints and tile corner-vertices, for the spatial heads.
_EDGE_U, _EDGE_W = EDGE_V[:, 0], EDGE_V[:, 1]
_TILE_V = jnp.asarray(TILE_V)
_SCATTER = jnp.asarray(al.SCATTER)
_TYPE_ID = jnp.asarray(al.TYPE_ID)


class _FactoredPolicy(eqx.Module):
    """A structure-aware policy head: spatial actions get their logit from the
    matching board embedding (vertex / undirected-edge / tile), the rest from a
    dense head, plus a per-type bias (the class-balance knob). Equivariant under
    board symmetry (shared per-slot heads), invariant under player relabeling."""

    vertex: eqx.nn.Linear  # node emb -> {setup-settle, settle, city}
    edge: eqx.nn.Linear  # symmetric endpoint emb -> {setup-road, road}
    tile: eqx.nn.Linear  # mean corner emb -> {robber, knight} x {no-steal, steal}
    other: eqx.nn.MLP  # global ctx -> the non-spatial actions
    type_bias: eqx.nn.Linear  # global ctx -> per-type bias

    def __init__(self, key: Array, cfg: GraphNetConfig, ctx_dim: int) -> None:
        w = cfg.width
        ks = jax.random.split(key, 5)
        self.vertex = eqx.nn.Linear(w, al.N_VCLASS, key=ks[0])
        self.edge = eqx.nn.Linear(w, al.N_ECLASS, key=ks[1])
        self.tile = eqx.nn.Linear(w, al.N_TCLASS, key=ks[2])
        self.other = eqx.nn.MLP(ctx_dim, al.N_OTHER, w, cfg.head_depth, key=ks[3])
        self.type_bias = eqx.nn.Linear(ctx_dim, al.N_TYPES, key=ks[4])

    def __call__(
        self, h: Float[Array, "v w"], ctx: Float[Array, "ctx"]
    ) -> Float[Array, f"flat={N_FLAT}"]:
        v = jax.vmap(self.vertex)(h)  # (V, N_VCLASS)
        e = jax.vmap(self.edge)(h[_EDGE_U] + h[_EDGE_W])  # (E, N_ECLASS), symmetric
        t = jax.vmap(self.tile)(h[_TILE_V].mean(axis=1))  # (T, N_TCLASS)
        big = jnp.concatenate(
            [v.reshape(-1), e.reshape(-1), t.reshape(-1), self.other(ctx)]
        )
        return big[_SCATTER] + self.type_bias(ctx)[_TYPE_ID]


class AZGraphNet(eqx.Module):
    """The board value+policy net: a shared :class:`GraphTrunk`, then the heads
    **split right after message passing** -- a value head (pooled readout ->
    win-prob logit, ``value_scale=2``) and the structure-factored policy head
    (:class:`_FactoredPolicy`). Returns ``(value_logit, policy_logits)``."""

    trunk: GraphTrunk
    value: eqx.nn.MLP
    policy: _FactoredPolicy

    def __init__(self, key: Array, cfg: GraphNetConfig) -> None:
        kt, kv, kp = jax.random.split(key, 3)
        ctx_dim = readout_dim(cfg) + cfg.width
        self.trunk = GraphTrunk(kt, cfg)
        self.value = eqx.nn.MLP(ctx_dim, 1, cfg.width, cfg.head_depth, key=kv)
        self.policy = _FactoredPolicy(kp, cfg, ctx_dim)

    def __call__(self, s: Sample) -> tuple[Value, Float[Array, f"flat={N_FLAT}"]]:
        h, g, readout = self.trunk(s)
        ctx = jnp.concatenate([readout, g])
        return self.value(ctx)[0], self.policy(h, ctx)


def make_az_gnn(model: AZGraphNet) -> tuple[ValueFunction, PolicyPrior]:
    """Adapt the GNN onto the search seams as ``(value, prior)``; both run the
    board-graph forward. Build the search with ``value_scale=2``."""

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        return model(board_sample(layout, state, player))[0]

    def prior(
        layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Float[Array, f"flat={N_FLAT}"]:
        return model(board_sample(layout, state, player))[1]

    return value, prior


class GNNSamples(NamedTuple):
    """Self-play positions: the board graph (nodes/edges/glob), the search's
    improved policy, and the acting seat's eventual win (1) / loss (0)."""

    nodes: np.ndarray
    edges: np.ndarray
    glob: np.ndarray
    policy: np.ndarray
    value: np.ndarray


def _to_sample(b: GNNSamples) -> Sample:
    """A batched :class:`Sample` for the GNN forward (engineered head unused, so
    fed zeros)."""
    n = b.nodes.shape[0]
    return Sample(
        jnp.asarray(b.nodes),
        jnp.asarray(b.edges),
        jnp.asarray(b.glob),
        jnp.zeros((n, FEATURE_DIM), jnp.float32),
    )


def _sample_moves(key: Array, weights: Array, mask: Array, temperature: float) -> Array:
    if temperature <= 0.0:
        return jnp.argmax(jnp.where(mask, weights, -jnp.inf), axis=-1)
    logits = jnp.where(mask, jnp.log(jnp.clip(weights, 1e-8)) / temperature, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1)


def self_play(
    model: AZGraphNet,
    *,
    n_samples: int,
    n_players: int = 2,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    batch_size: int = 16,
    temperature: float = 1.0,
    seed: int = 0,
) -> GNNSamples:
    """Collect >= ``n_samples`` self-play positions under ``model`` (the GNN
    guiding the re-determinizing search). Positions from finished games are
    credited with the acting seat's outcome; unfinished games are discarded."""
    value_fn, prior_fn = make_az_gnn(model)
    weights_fn = make_search_weights(
        value_fn,
        prior=prior_fn,
        value_scale=2.0,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
    )
    search = jax.jit(jax.vmap(weights_fn, in_axes=(0, 0, 0, 0, 0)))
    view_of = jax.jit(jax.vmap(belief_view, in_axes=(0, 0, 0)))
    sample_of = jax.jit(jax.vmap(board_sample, in_axes=(0, 0, 0)))

    env = BatchedSettlrlEnv(
        batch_size=batch_size, seed=seed, reward="sparse",
        n_players=n_players, track_beliefs=True,
    )  # fmt: skip
    pending: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]] = [
        [] for _ in range(batch_size)
    ]
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in ("nodes", "edges", "glob", "pol")
    }
    vals: list[float] = []
    key = jax.random.key(seed)

    while len(vals) < n_samples:
        layout, state = env.board
        beliefs = env.beliefs
        assert beliefs is not None
        sel = jnp.asarray(env.agent_selection)
        mask = env.flat_mask()
        view = view_of(state, beliefs, sel)
        key, k_search, k_move = jax.random.split(key, 3)
        weights = search(
            jax.random.split(k_search, batch_size), layout, view, sel, mask
        )
        move = _sample_moves(k_move, weights, mask, temperature)

        s = sample_of(layout, state, sel)
        n_np, e_np, g_np = (
            np.asarray(s.nodes), np.asarray(s.edges), np.asarray(s.glob),
        )  # fmt: skip
        w_np, sel_np = np.asarray(weights), np.asarray(sel)
        for lane in range(batch_size):
            pending[lane].append(
                (n_np[lane], e_np[lane], g_np[lane], w_np[lane], int(sel_np[lane]))
            )

        env.step(*flat_to_action(move))
        rewards = np.asarray(env.rewards)
        for lane in np.flatnonzero(np.asarray(env.terminations).any(axis=1)).tolist():
            for n_l, e_l, g_l, w_l, seat in pending[lane]:
                out["nodes"].append(n_l)
                out["edges"].append(e_l)
                out["glob"].append(g_l)
                out["pol"].append(w_l)
                vals.append(float(rewards[lane, seat] > 0))
            pending[lane] = []

    return GNNSamples(
        np.stack(out["nodes"]), np.stack(out["edges"]), np.stack(out["glob"]),
        np.stack(out["pol"]), np.asarray(vals, np.float32),
    )  # fmt: skip


def az_gnn_loss(
    model: AZGraphNet, sample: Sample, policy: Array, value: Array
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Policy cross-entropy (against the search target) + value logistic loss."""
    vs, logits = jax.vmap(model)(sample)
    logp = jax.nn.log_softmax(logits, axis=-1)
    policy_loss = -jnp.mean(jnp.sum(policy * logp, axis=-1))
    value_loss = jnp.mean(jax.nn.softplus(vs) - value * vs)
    return policy_loss + value_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


def arena(
    model: AZGraphNet,
    *,
    opponent: str = "lookahead",
    n_games: int = 40,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    batch_size: int = 16,
    seed: int = 0,
) -> float:
    """The GNN's win rate vs. a ``POLICIES`` ``opponent``, seat-swapped at 2p
    (``lookahead`` = the Stage-1 gate; ``random`` = the lower-bound sanity check)."""
    value_fn, prior_fn = make_az_gnn(model)
    net = make_search(
        value_fn, prior=prior_fn, value_scale=2.0,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
    )  # fmt: skip
    net_spec = BeliefSpec(lambda: net, frozenset((2,)))
    base = POLICIES[opponent]
    half = max(1, n_games // 2)
    r1 = evaluate([net_spec, base], n_episodes=half, batch_size=batch_size, seed=seed)
    r2 = evaluate(
        [base, net_spec], n_episodes=half, batch_size=batch_size, seed=seed + 1
    )
    return float(r1.wins[0] + r2.wins[1]) / max(int(r1.episodes + r2.episodes), 1)


class _Item(NamedTuple):
    """One replay item (the board graph + policy + value), for the flashbax
    on-device buffer."""

    nodes: Array
    edges: Array
    glob: Array
    policy: Array
    value: Array


def _empty_item() -> _Item:
    return _Item(
        jnp.zeros((N_VERTICES, NODE_DIM), jnp.float32),
        jnp.zeros((N_DIR_EDGES, EDGE_DIM), jnp.float32),
        jnp.zeros((GLOBAL_DIM,), jnp.float32),
        jnp.zeros((N_FLAT,), jnp.float32),
        jnp.float32(0.0),
    )


def _add(buffer: Any, state: Any, s: GNNSamples) -> Any:
    item = _Item(
        jnp.asarray(s.nodes, jnp.float32),
        jnp.asarray(s.edges, jnp.float32),
        jnp.asarray(s.glob, jnp.float32),
        jnp.asarray(s.policy, jnp.float32),
        jnp.asarray(s.value, jnp.float32),
    )
    return buffer.add(state, item)


class GNNState(NamedTuple):
    """The whole mutable run state, eqx-serialised for resume (the per-iteration
    RNG is a pure function of ``seed`` and the iteration index, so a resumed run
    continues bit-identically)."""

    model: AZGraphNet
    opt_state: optax.OptState
    buffer_state: Any  # flashbax buffer state pytree
    iteration: Int[Array, ""]
    best: Float[Array, ""]


def save_gnn_state(path: str | Path, state: GNNState) -> None:
    eqx.tree_serialise_leaves(Path(path), state)


def load_gnn_state(path: str | Path, template: GNNState) -> GNNState:
    return cast(GNNState, eqx.tree_deserialise_leaves(Path(path), template))


def learn(
    *,
    cfg: GraphNetConfig,
    n_iterations: int,
    selfplay_samples: int,
    selfplay_batch: int = 16,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    temperature: float = 1.0,
    buffer_max: int = 50_000,
    batch_size: int = 256,
    train_steps: int = 200,
    reuse: float = 0.0,
    eval_frac: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    arena_games: int = 0,
    arena_every: int = 1,
    arena_batch: int = 128,
    arena_sims: int = 48,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 1,
    resume_from: str | Path | None = None,
    on_iter: Callable[[int, dict[str, float], AZGraphNet], None] | None = None,
) -> AZGraphNet:
    """An AlphaZero loop over an :class:`AZGraphNet`: each iteration self-plays,
    fills a flashbax on-device replay, and trains; every ``arena_every`` it scores
    vs. ``lookahead(heuristic)``. The full :class:`GNNState` is checkpointed to
    ``checkpoint_dir`` every ``checkpoint_every`` iterations, and ``resume_from``
    (a prior ``gnnstate.eqx``) continues it bit-exactly.

    ``reuse`` > 0 caps the updates per iteration at ``reuse * fresh / batch_size``
    (the AlphaZero sample-reuse factor) instead of a fixed ``train_steps`` -- the
    fix for value-head overfitting on a small early replay. A held-out
    ``eval_frac`` of each iteration's fresh positions (never trained, not
    checkpointed) gives the ``val_*`` metrics, including ``val_value_acc``."""
    opt = optax.adamw(lr, weight_decay=weight_decay)
    buffer = fbx.make_item_buffer(
        max_length=buffer_max, min_length=batch_size,
        sample_batch_size=batch_size, add_batches=True,
    )  # fmt: skip
    model0 = AZGraphNet(jax.random.key(seed), cfg)
    fresh_state = GNNState(
        model0, opt.init(eqx.filter(model0, eqx.is_inexact_array)),
        buffer.init(_empty_item()), jnp.int32(0), jnp.float32(-1.0),
    )  # fmt: skip
    state = load_gnn_state(resume_from, fresh_state) if resume_from else fresh_state
    model, opt_state, buf_state = state.model, state.opt_state, state.buffer_state
    best = float(state.best)
    ckpt = Path(checkpoint_dir) / "gnnstate.eqx" if checkpoint_dir else None

    @eqx.filter_jit
    def step(m: Any, st: Any, s: Sample, pol: Array, val: Array) -> Any:
        (loss, aux), grads = eqx.filter_value_and_grad(az_gnn_loss, has_aux=True)(
            m, s, pol, val
        )
        updates, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        return (
            eqx.apply_updates(m, updates), st, loss, aux,
            optax.global_norm(grads), optax.global_norm(updates),
        )  # fmt: skip

    @eqx.filter_jit
    def evaluate_net(m: Any, s: Sample, pol: Array, val: Array) -> dict[str, Array]:
        vs, logits = jax.vmap(m)(s)
        logp = jax.nn.log_softmax(logits, axis=-1)
        p = jnp.exp(logp)
        return {
            "val_policy_loss": -jnp.mean(jnp.sum(pol * logp, axis=-1)),
            "val_value_loss": jnp.mean(jax.nn.softplus(vs) - val * vs),
            "val_value_acc": jnp.mean((vs > 0).astype(jnp.float32) == val),
            # policy-head health: entropy (collapse -> ~0) vs uniform log(N_FLAT).
            "policy_entropy": -jnp.mean(jnp.sum(p * logp, axis=-1)),
            "policy_top_prob": jnp.mean(jnp.max(p, axis=-1)),
            # value-head health: logit spread + mean predicted P(win) (~0.5 sane).
            "value_logit_mean": jnp.mean(vs),
            "value_logit_std": jnp.std(vs),
            "pred_winrate": jnp.mean(jax.nn.sigmoid(vs)),
            "value_label_mean": jnp.mean(val),
        }

    @eqx.filter_jit
    def _param_norm(m: Any) -> Array:
        return cast(Array, optax.global_norm(eqx.filter(m, eqx.is_inexact_array)))

    ev: GNNSamples | None = None
    for i in range(int(state.iteration), n_iterations):
        t0 = time.perf_counter()
        fresh = self_play(
            model, n_samples=selfplay_samples, num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            batch_size=selfplay_batch, temperature=temperature, seed=seed + 1 + i,
        )  # fmt: skip
        t_selfplay = time.perf_counter() - t0
        # hold out a never-trained eval slice (reproducible per iteration), buffer
        # the rest into flashbax.
        nf = fresh.value.shape[0]
        perm = np.random.default_rng(seed + 50_000 + i).permutation(nf)
        n_ev = int(nf * eval_frac)
        fr, fe = _index(fresh, perm[n_ev:]), _index(fresh, perm[:n_ev])
        buf_state = _add(buffer, buf_state, fr)
        ev = fe if ev is None else _concat(ev, fe, 8192)
        steps = (
            train_steps
            if reuse <= 0
            else max(1, int(reuse * fr.value.shape[0] / batch_size))
        )
        # entropy of the search policy *targets* (degenerate targets -> the net
        # learns a degenerate policy): the diagnostic alongside the net's own
        # policy_entropy.
        tp = jnp.asarray(fr.policy)
        target_entropy = float(
            -jnp.mean(jnp.sum(tp * jnp.log(jnp.clip(tp, 1e-9, 1.0)), axis=-1))
        )
        metrics: dict[str, float] = {
            "samples": float(nf), "train_steps": float(steps),
            "buffer_size": float(buf_state.current_index) if hasattr(buf_state, "current_index") else float("nan"),
            "lr": float(lr), "target_entropy": target_entropy,
            "param_norm": float(_param_norm(model)), "t_selfplay": t_selfplay,
        }  # fmt: skip
        key = jax.random.key(seed + 10_000 + i)
        t1 = time.perf_counter()
        if bool(buffer.can_sample(buf_state)):
            gnorms, unorms = [], []
            for _ in range(steps):
                key, k = jax.random.split(key)
                b = buffer.sample(buf_state, k).experience
                s = Sample(
                    b.nodes, b.edges, b.glob, jnp.zeros((batch_size, FEATURE_DIM))
                )
                model, opt_state, loss, aux, gnorm, unorm = step(
                    model, opt_state, s, b.policy, b.value
                )
                gnorms.append(float(gnorm))
                unorms.append(float(unorm))
            metrics["loss"] = float(loss)
            metrics["grad_norm"] = float(np.mean(gnorms))
            metrics["grad_norm_max"] = float(np.max(gnorms))
            metrics["update_norm"] = float(np.mean(unorms))
            metrics.update({k2: float(v) for k2, v in aux.items()})
        metrics["t_train"] = time.perf_counter() - t1
        if ev.value.shape[0] >= batch_size:
            vm = evaluate_net(
                model, _to_sample(ev), jnp.asarray(ev.policy), jnp.asarray(ev.value)
            )
            metrics.update({k2: float(v) for k2, v in vm.items()})
        if arena_games and (i + 1) % arena_every == 0:
            t2 = time.perf_counter()
            # Arena decoupled from training: many lanes (parallel, GPU-saturating)
            # at a modest sim budget -- ~an order of magnitude faster, same signal.
            winrate = arena(
                model, opponent="lookahead", n_games=arena_games,
                num_simulations=arena_sims, batch_size=arena_batch,
                max_num_considered_actions=max_num_considered_actions,
                seed=seed + 20_000 + i,
            )  # fmt: skip
            metrics["arena_winrate"] = winrate
            metrics["arena_vs_random"] = arena(
                model, opponent="random", n_games=arena_games,
                num_simulations=arena_sims, batch_size=arena_batch,
                max_num_considered_actions=max_num_considered_actions,
                seed=seed + 30_000 + i,
            )  # fmt: skip
            metrics["t_arena"] = time.perf_counter() - t2
            best = max(best, winrate)
        if ckpt is not None and (i + 1) % checkpoint_every == 0:
            save_gnn_state(
                ckpt,
                GNNState(
                    model, opt_state, buf_state, jnp.int32(i + 1), jnp.float32(best)
                ),
            )
        if on_iter is not None:
            on_iter(i, metrics, model)
    return model


def _concat(a: GNNSamples, b: GNNSamples, cap: int) -> GNNSamples:
    j = jax.tree.map(lambda x, y: np.concatenate([x, y])[-cap:], a, b)
    return cast(GNNSamples, j)


def _index(b: GNNSamples, idx: np.ndarray) -> GNNSamples:
    return cast(GNNSamples, jax.tree.map(lambda x: x[idx], b))


__all__ = [
    "PRESETS",
    "AZGraphNet",
    "GNNSamples",
    "GNNState",
    "arena",
    "learn",
    "load_gnn_state",
    "save_gnn_state",
    "self_play",
]
