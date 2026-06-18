"""AlphaZero with a GNN trunk (the experiment-0003 ``gn_global`` net).

The GNN reads the board *graph* (``settlrl_learn.graph.board_sample``), not the
flat engineered vector, so it needs its own self-play / training path: this
module mirrors :mod:`settlrl_learn.selfplay` + :mod:`settlrl_learn.alphazero` for
an equinox :class:`~settlrl_learn.graphnet.GraphNet` with a shared trunk feeding
a value head (win-prob logit, ``value_scale=2``) and an ``N_FLAT`` policy head.

A small proof-of-concept loop: an in-memory replay (no flashbax/orbax bit-exact
resume yet), self-play -> train -> periodic arena vs ``lookahead(heuristic)``.

A training-side module (equinox/jraph/optax): not imported by the package root.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float
from settlrl_agents import POLICIES, BeliefSpec, evaluate
from settlrl_agents.policy import PolicyPrior
from settlrl_agents.search import make_search, make_search_weights
from settlrl_agents.value import Value, ValueFunction
from settlrl_engine.belief import belief_view
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, IntScalar
from settlrl_engine.env import N_FLAT, BatchedSettlrlEnv, flat_to_action

from settlrl_learn.features import FEATURE_DIM
from settlrl_learn.graph import Sample, board_sample
from settlrl_learn.graphnet import PRESETS, GraphNet, GraphNetConfig


class AZGraphNet(eqx.Module):
    """A GraphNet trunk over the board graph with a value + policy head: the
    forward returns ``(value_logit, policy_logits)`` (``value_scale=2``)."""

    net: GraphNet

    def __init__(self, key: Array, cfg: GraphNetConfig) -> None:
        self.net = GraphNet(key, out_dim=1 + N_FLAT, cfg=cfg)

    def __call__(self, s: Sample) -> tuple[Value, Float[Array, f"flat={N_FLAT}"]]:
        out = self.net(s)
        return out[0], out[1:]


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


def _sample_moves(
    key: Array, weights: Array, mask: Array, temperature: float
) -> Array:
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
    out: dict[str, list[np.ndarray]] = {k: [] for k in ("nodes", "edges", "glob", "pol")}
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
    n_games: int = 40,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    batch_size: int = 16,
    seed: int = 0,
) -> float:
    """The GNN's win rate vs. ``lookahead(heuristic)``, seat-swapped at 2p."""
    value_fn, prior_fn = make_az_gnn(model)
    net = make_search(
        value_fn, prior=prior_fn, value_scale=2.0,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
    )  # fmt: skip
    net_spec = BeliefSpec(lambda: net, frozenset((2,)))
    base = POLICIES["lookahead"]
    half = max(1, n_games // 2)
    r1 = evaluate([net_spec, base], n_episodes=half, batch_size=batch_size, seed=seed)
    r2 = evaluate([base, net_spec], n_episodes=half, batch_size=batch_size, seed=seed + 1)
    return float(r1.wins[0] + r2.wins[1]) / max(int(r1.episodes + r2.episodes), 1)


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
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    arena_games: int = 0,
    arena_every: int = 1,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    on_iter: Callable[[int, dict[str, float], AZGraphNet], None] | None = None,
) -> AZGraphNet:
    """A small AlphaZero loop over an :class:`AZGraphNet`: each iteration
    self-plays, grows an in-memory replay, and trains; every ``arena_every`` it
    scores vs. ``lookahead(heuristic)``. Returns the final net."""
    model = AZGraphNet(jax.random.key(seed), cfg)
    opt = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m: Any, st: Any, s: Sample, pol: Array, val: Array) -> Any:
        (loss, aux), grads = eqx.filter_value_and_grad(az_gnn_loss, has_aux=True)(
            m, s, pol, val
        )
        updates, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), st, loss, aux

    buf: GNNSamples | None = None
    rng = np.random.default_rng(seed)
    for i in range(n_iterations):
        fresh = self_play(
            model, n_samples=selfplay_samples, num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            batch_size=selfplay_batch, temperature=temperature, seed=seed + 1 + i,
        )  # fmt: skip
        buf = fresh if buf is None else _concat(buf, fresh, buffer_max)
        n = buf.value.shape[0]
        metrics: dict[str, float] = {"samples": float(fresh.value.shape[0])}
        for _ in range(train_steps):
            idx = rng.integers(0, n, batch_size)
            mb = _index(buf, idx)
            model, opt_state, loss, aux = step(
                model, opt_state, _to_sample(mb),
                jnp.asarray(mb.policy), jnp.asarray(mb.value),
            )  # fmt: skip
        metrics["loss"] = float(loss)
        metrics.update({k: float(v) for k, v in aux.items()})
        if arena_games and (i + 1) % arena_every == 0:
            metrics["arena_winrate"] = arena(
                model, n_games=arena_games, num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
                seed=seed + 20_000 + i,
            )  # fmt: skip
        if checkpoint_dir is not None:
            eqx.tree_serialise_leaves(Path(checkpoint_dir) / "model.eqx", model)
        if on_iter is not None:
            on_iter(i, metrics, model)
    return model


def _concat(a: GNNSamples, b: GNNSamples, cap: int) -> GNNSamples:
    j = jax.tree.map(lambda x, y: np.concatenate([x, y])[-cap:], a, b)
    return cast(GNNSamples, j)


def _index(b: GNNSamples, idx: np.ndarray) -> GNNSamples:
    return cast(GNNSamples, jax.tree.map(lambda x: x[idx], b))


__all__ = ["PRESETS", "AZGraphNet", "GNNSamples", "arena", "learn", "self_play"]
