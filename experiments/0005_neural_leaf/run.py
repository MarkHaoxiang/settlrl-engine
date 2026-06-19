"""Neural leaf: a GNN value fit to a strong teacher's outcomes, gated as a leaf.

Hypothesis: a value head fit to *seat-0 win/loss under a strong teacher's play*
(`lookahead(heuristic)`) is the value of that policy, so one-step lookahead over
it is one policy-improvement step — `lookahead(gnn)` should beat
`lookahead(heuristic)` at 2p, the leaf the search ladder is stuck against.

    uv run python experiments/0005_neural_leaf/run.py [variant] [key=value ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

from settlrl_learn.experiment import Config, Run, start_run


class NeuralLeafConfig(Config):
    seed: int = 0
    # data: the teacher whose outcomes label the positions
    agent: str = "lookahead"
    n_samples: int = 120_000
    snapshot_every: int = 6
    collect_batch: int = 256
    val_frac: float = 0.15
    # net (gn_global trunk + value head)
    width: int = 96
    layers: int = 4
    head_depth: int = 2
    # train
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # gate: lookahead(gnn) vs lookahead(heuristic) / lookahead(tuned), 2p
    gate_games: int = 600
    gate_batch: int = 128


VARIANTS: dict[str, dict[str, object]] = {
    "default": {},
    "search_teacher": {"agent": "mcts", "collect_batch": 128, "n_samples": 80_000},
    "smoke": {
        "n_samples": 200,
        "snapshot_every": 16,
        "collect_batch": 16,
        "width": 16,
        "layers": 1,
        "head_depth": 1,
        "epochs": 2,
        "batch_size": 64,
        "gate_games": 8,
        "gate_batch": 8,
    },
}


def run_experiment(run: Run, cfg: NeuralLeafConfig) -> None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from data import Dataset, generate, split
    from settlrl_agents import BeliefSpec
    from settlrl_agents.evaluate import evaluate
    from settlrl_agents.search import make_search
    from settlrl_agents.value import heuristic_value, tuned_value
    from settlrl_learn.azgnn import AZGraphNet, make_az_gnn
    from settlrl_learn.graph import Sample
    from settlrl_learn.graphnet import PRESETS

    # --- data ---
    data_cfg = {
        "agent": cfg.agent, "n_samples": cfg.n_samples,
        "snapshot_every": cfg.snapshot_every, "batch_size": cfg.collect_batch,
        "seed": cfg.seed,
    }  # fmt: skip
    ds = generate(data_cfg)
    train_ds, val_ds = split(ds, cfg.val_frac, seed=cfg.seed)
    run.log(
        n_samples=int(ds.win.shape[0]), n_train=int(train_ds.win.shape[0]),
        n_episodes=int(np.unique(ds.episode).size), win_rate=float(ds.win.mean()),
    )  # fmt: skip

    def to_jax(d: Dataset) -> tuple[Sample, Any]:
        return cast(Sample, jax.tree.map(jnp.asarray, d.samples)), jnp.asarray(d.win)

    tr_x, tr_y = to_jax(train_ds)
    va_x, va_y = to_jax(val_ds)

    # --- model + train (AZGraphNet value head, BCE on seat-0 win) ---
    gcfg = PRESETS["gn_global"]._replace(
        width=cfg.width, layers=cfg.layers, head_depth=cfg.head_depth
    )
    model = AZGraphNet(jax.random.key(cfg.seed), gcfg)
    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def value_logits(m: AZGraphNet, xs: Sample) -> Any:
        return jax.vmap(lambda s: m(s)[0])(xs)

    @eqx.filter_jit
    def train_step(m: AZGraphNet, st: Any, xs: Sample, y: Any) -> Any:
        def loss_fn(mm: AZGraphNet) -> Any:
            vs = value_logits(mm, xs)
            return jnp.mean(jax.nn.softplus(vs) - y * vs)  # BCE-with-logits

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)  # type: ignore[no-untyped-call]
        updates, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), st, loss

    @eqx.filter_jit
    def val_metrics(m: AZGraphNet) -> tuple[Any, Any]:
        vs = value_logits(m, va_x)
        bce = jnp.mean(jax.nn.softplus(vs) - va_y * vs)
        # AUC via the rank statistic (Mann-Whitney U), pos vs neg.
        ranks = jnp.argsort(jnp.argsort(vs)).astype(jnp.float32) + 1.0
        n_pos = va_y.sum()
        n_neg = va_y.shape[0] - n_pos
        auc = (jnp.sum(ranks * va_y) - n_pos * (n_pos + 1) / 2) / jnp.maximum(
            n_pos * n_neg, 1.0
        )
        return bce, auc

    n = int(tr_y.shape[0])
    rng = np.random.default_rng(cfg.seed)
    best_auc, best_model = -1.0, model
    for epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        for i in range(0, n - cfg.batch_size + 1, cfg.batch_size):
            idx = jnp.asarray(perm[i : i + cfg.batch_size])
            xb = jax.tree.map(lambda a, idx=idx: a[idx], tr_x)
            model, opt_state, _ = train_step(model, opt_state, xb, tr_y[idx])
        bce, auc = (float(v) for v in val_metrics(model))
        run.log(epoch=epoch, val_bce=bce, val_auc=auc)
        if auc > best_auc:
            best_auc, best_model = auc, model
    run.log(best_val_auc=best_auc)
    eqx.tree_serialise_leaves(str(run.dir / "value.eqx"), best_model)

    # --- gate: lookahead(gnn) vs lookahead(heuristic) / lookahead(tuned), 2p ---
    gnn_value, _ = make_az_gnn(best_model)
    two = frozenset((2,))

    def leaf(value_fn: Any) -> BeliefSpec:
        # num_simulations=0 = one-step lookahead; propose off to isolate the value.
        return BeliefSpec(
            lambda: make_search(value_fn, num_simulations=0, propose_rate=0.0), two
        )

    gnn = leaf(gnn_value)

    def match(name: str, opp: BeliefSpec, seed: int) -> tuple[float, int]:
        half = cfg.gate_games // 2
        r1 = evaluate([gnn, opp], n_episodes=half, batch_size=cfg.gate_batch, seed=seed)
        r2 = evaluate(
            [opp, gnn], n_episodes=half, batch_size=cfg.gate_batch, seed=seed + 1
        )
        eps = int(r1.episodes + r2.episodes)
        wr = float(r1.wins[0] + r2.wins[1]) / max(eps, 1)
        se = (wr * (1 - wr) / max(eps, 1)) ** 0.5
        run.log(match=name, win_rate=wr, se=se, n=eps, lower2s=wr - 2 * se)
        return wr, eps

    wr_heur, n_heur = match(
        "vs_lookahead_heuristic", leaf(heuristic_value), 10_000 + cfg.seed
    )
    wr_tuned, _ = match("vs_lookahead_tuned", leaf(tuned_value), 20_000 + cfg.seed)

    se = (wr_heur * (1 - wr_heur) / max(n_heur, 1)) ** 0.5
    verdict = "pass" if wr_heur - 2 * se > 0.5 else "fail"
    run.finish(
        verdict, best_val_auc=best_auc, vs_heuristic=wr_heur, vs_tuned=wr_tuned,
        lower2s_vs_heuristic=wr_heur - 2 * se, n=n_heur,
    )  # fmt: skip


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "default"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}] [key=value ...]")
    cfg = NeuralLeafConfig.resolve(VARIANTS[variant], overrides=sys.argv[2:])
    run_experiment(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
