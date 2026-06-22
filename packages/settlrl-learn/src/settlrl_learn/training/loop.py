"""The net-agnostic training loop: self-play -> replay -> train -> periodic arena.

Each iteration self-plays under the current net (or a fixed teacher during the
warm-up), buffers the positions into a flashbax on-device replay, trains, and --
once past the warm-up -- scores the net vs. ``lookahead(heuristic)``. The
:class:`~settlrl_learn.training.backend.Backend` supplies everything net-specific;
this loop is shared by the flat-MLP and board-GNN paths.

Per-iteration RNG derives from ``seed`` and the iteration index, so ``resume_from``
(a prior ``runstate.eqx``) continues a run bit-exactly.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from settlrl_agents.search import PolicyWeights, make_search_weights
from settlrl_agents.value import ValueFunction

from settlrl_learn.training.arena import arena
from settlrl_learn.training.backend import (
    Backend,
    RunState,
    load_run_state,
    save_run_state,
)
from settlrl_learn.training.selfplay import Samples, concat, index, self_play


def learn(
    backend: Backend,
    *,
    n_iterations: int,
    selfplay_samples: int,
    selfplay_batch: int = 16,
    num_simulations: int = 64,
    max_num_considered_actions: int = 16,
    temperature: float = 1.0,
    teacher_value: ValueFunction | None = None,
    teacher_iters: int = 0,
    teacher_sims: int = 32,
    buffer_max: int = 50_000,
    buffer_min: int = 256,
    batch_size: int = 256,
    train_steps: int = 200,
    reuse: float = 0.0,
    eval_frac: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    arena_games: int = 0,
    arena_every: int = 1,
    arena_batch: int = 16,
    arena_sims: int = 48,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 1,
    resume_from: str | Path | None = None,
    on_iter: Callable[[int, dict[str, float], Any], None] | None = None,
    progress: bool = False,
) -> Any:
    """One training loop over ``backend``; returns the final net.

    ``reuse`` > 0 caps the updates per iteration at ``reuse * fresh / batch_size``
    (the AlphaZero sample-reuse factor) instead of a fixed ``train_steps`` -- the
    fix for value-head overfitting on a small early replay. ``eval_frac`` > 0 holds
    out that fraction of each iteration's fresh positions (never trained, not
    checkpointed) for the ``val_*`` metrics.

    ``teacher_value`` (with ``teacher_iters`` > 0) warm-starts the loop: the first
    ``teacher_iters`` iterations draw their moves and policy targets from a fixed
    strong search over ``teacher_value`` (``teacher_sims`` simulations) instead of
    the cold net, so the heads learn from strong play before self-play takes over.

    The full :class:`RunState` is checkpointed to ``checkpoint_dir/runstate.eqx``
    every ``checkpoint_every`` iterations; ``resume_from`` continues it bit-exactly.
    ``on_iter(i, metrics, net)`` runs after each iteration. ``progress`` shows a
    tqdm bar over the iterations, its postfix tracking loss / arena win rate."""
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    buffer = fbx.make_item_buffer(
        max_length=buffer_max, min_length=buffer_min,
        sample_batch_size=batch_size, add_batches=True,
    )  # fmt: skip
    net0 = backend.init(jax.random.key(seed))
    fresh_state = RunState(
        net0, backend.init_opt(optimizer, net0),
        buffer.init(backend.empty_item()), jnp.int32(0), jnp.float32(-1.0),
    )  # fmt: skip
    state = load_run_state(resume_from, fresh_state) if resume_from else fresh_state
    net, opt_state, buf_state = state.net, state.opt_state, state.buffer_state
    best = float(state.best)
    ckpt = Path(checkpoint_dir) / "runstate.eqx" if checkpoint_dir else None

    step = backend.make_step(optimizer)
    setup_fn = backend.setup_policy()
    teacher_weights = (
        make_search_weights(
            teacher_value,
            num_simulations=teacher_sims,
            max_num_considered_actions=max_num_considered_actions,
        )
        if teacher_value is not None
        else None
    )

    iters: Iterable[int] = range(int(state.iteration), n_iterations)
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(iters, initial=int(state.iteration), total=n_iterations, unit="iter")
        iters = bar

    ev: Samples | None = None
    for i in iters:
        t0 = time.perf_counter()
        teaching = teacher_weights is not None and i < teacher_iters
        if teaching:
            wfn = cast(PolicyWeights, teacher_weights)
        else:
            v_fn, p_fn = backend.seams(net)
            wfn = make_search_weights(
                v_fn, prior=p_fn, value_scale=2.0,
                num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
            )  # fmt: skip
        fresh = self_play(
            wfn, backend.observe, n_samples=selfplay_samples, setup_fn=setup_fn,
            batch_size=selfplay_batch, temperature=temperature, seed=seed + 1 + i,
        )  # fmt: skip
        t_selfplay = time.perf_counter() - t0
        nf = fresh["value"].shape[0]
        if nf == 0:  # degenerate net dragged every game past the budget; skip
            if on_iter is not None:
                on_iter(i, {"samples": 0.0, "teaching": float(teaching)}, net)
            continue

        # hold out a never-trained eval slice (reproducible per iteration).
        if eval_frac > 0:
            perm = np.random.default_rng(seed + 50_000 + i).permutation(nf)
            n_ev = int(nf * eval_frac)
            fr, fe = index(fresh, perm[n_ev:]), index(fresh, perm[:n_ev])
            ev = fe if ev is None else concat(ev, fe, 8192)
        else:
            fr = fresh
        buf_state = buffer.add(buf_state, backend.to_item(fr))
        steps = (
            train_steps
            if reuse <= 0
            else max(1, int(reuse * fr["value"].shape[0] / batch_size))
        )
        # entropy of the search policy *targets* (degenerate targets -> the net
        # learns a degenerate policy).
        tp = jnp.asarray(fr["policy"])
        target_entropy = float(
            -jnp.mean(jnp.sum(tp * jnp.log(jnp.clip(tp, 1e-9, 1.0)), axis=-1))
        )
        metrics: dict[str, float] = {
            "samples": float(nf), "train_steps": float(steps),
            "lr": lr, "target_entropy": target_entropy,
            "t_selfplay": t_selfplay, "teaching": float(teaching),
        }  # fmt: skip

        key = jax.random.key(seed + 10_000 + i)
        t1 = time.perf_counter()
        if bool(buffer.can_sample(buf_state)):
            sums: dict[str, float] = {}
            for _ in range(steps):
                key, k = jax.random.split(key)
                item = buffer.sample(buf_state, k).experience
                net, opt_state, m = step(net, opt_state, item)
                for kk, vv in m.items():
                    sums[kk] = sums.get(kk, 0.0) + float(vv)
            metrics.update({kk: vv / steps for kk, vv in sums.items()})
        metrics["t_train"] = time.perf_counter() - t1

        if ev is not None and ev["value"].shape[0] >= batch_size:
            vm = backend.eval_metrics(net, backend.to_item(ev))
            metrics.update({kk: float(vv) for kk, vv in vm.items()})

        # Arena only once the net is past the warm-up: a half-trained net drags
        # games out, and the search arena pays full cost per step.
        if arena_games and (i + 1) % arena_every == 0 and (i + 1) >= teacher_iters:
            t2 = time.perf_counter()
            winrate = arena(
                backend, net, opponent="lookahead", n_games=arena_games,
                num_simulations=arena_sims, batch_size=arena_batch,
                max_num_considered_actions=max_num_considered_actions,
                seed=seed + 20_000 + i,
            )  # fmt: skip
            metrics["arena_winrate"] = winrate
            metrics["arena_vs_random"] = arena(
                backend, net, opponent="random", n_games=arena_games,
                num_simulations=arena_sims, batch_size=arena_batch,
                max_num_considered_actions=max_num_considered_actions,
                seed=seed + 30_000 + i,
            )  # fmt: skip
            metrics["t_arena"] = time.perf_counter() - t2
            best = max(best, winrate)

        if ckpt is not None and (i + 1) % checkpoint_every == 0:
            save_run_state(
                ckpt,
                RunState(
                    net, opt_state, buf_state, jnp.int32(i + 1), jnp.float32(best)
                ),
            )
        if bar is not None:
            bar.set_postfix(
                {
                    k: round(metrics[k], 3)
                    for k in ("loss", "arena_winrate")
                    if k in metrics
                }
            )
        if on_iter is not None:
            on_iter(i, metrics, net)
    return net
