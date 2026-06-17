"""Training loop: optax + equinox, wandb logging, best-val checkpointing.

One ``train`` call fits one architecture on one task and returns its held-out
metrics. Inputs are standardized on the train split (the engineered/global terms
span very different scales); the fitted stats are saved beside the checkpoint.
The win task is a win-probability logit (BCE, ranked by AUC); the heuristic task
is regression (RMSE / R^2).
"""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from data import Dataset
from features import Sample
from jaxtyping import Array, Float
from settlrl_agents.experiment import Run
from sklearn.metrics import roc_auc_score


def _standardize(
    train: Dataset, val: Dataset
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    """Zero-mean/unit-var each feature on the train split; apply to both."""
    stats: dict[str, Any] = {}

    def fit_apply(field: str, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        x = getattr(train.samples, field)
        mean = x.mean(axis=axes, keepdims=True)
        std = np.maximum(x.std(axis=axes, keepdims=True), 1e-3)
        stats[field] = {"mean": mean.squeeze(0), "std": std.squeeze(0)}
        tr = (x - mean) / std
        v = (getattr(val.samples, field) - mean) / std
        return tr.astype(np.float32), v.astype(np.float32)

    tr_fields: dict[str, Any] = {}
    v_fields: dict[str, Any] = {}
    for field, axes in (
        ("nodes", (0, 1)), ("edges", (0, 1)), ("glob", (0,)), ("engineered", (0,)),
    ):  # fmt: skip
        tr_fields[field], v_fields[field] = fit_apply(field, axes)
    return (
        train._replace(samples=Sample(**tr_fields)),
        val._replace(samples=Sample(**v_fields)),
        stats,
    )


def _to_jax(samples: Sample) -> Sample:
    return cast(Sample, jax.tree.map(jnp.asarray, samples))


def _metrics(task: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if task == "win":
        prob = 1.0 / (1.0 + np.exp(-pred))
        auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
        return {"auc": auc, "acc": float(np.mean((pred > 0) == (y > 0.5)))}
    sse = float(np.sum((pred - y) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2)) + 1e-9
    return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "r2": 1.0 - sse / sst}


def train(
    run: Run, cfg: dict, model: eqx.Module, train_ds: Dataset, val_ds: Dataset
) -> dict[str, float]:
    """Fit ``model`` and return its best held-out metrics (the selection metric
    is AUC for ``win``, R^2 for ``heuristic``; higher is better)."""
    task = cfg["task"]
    train_ds, val_ds, stats = _standardize(train_ds, val_ds)
    run.save_json(
        "standardizer.json",
        {k: {m: a.tolist() for m, a in v.items()} for k, v in stats.items()},
    )

    x_tr, x_val = _to_jax(train_ds.samples), _to_jax(val_ds.samples)
    y_tr = jnp.asarray(train_ds.win if task == "win" else train_ds.heur)
    y_val = np.asarray(val_ds.win if task == "win" else val_ds.heur)
    n = y_tr.shape[0]

    def loss_fn(m: Any, xs: Sample, y: Float[Array, "b"]) -> Float[Array, ""]:
        pred = jax.vmap(m)(xs)[:, 0]
        if task == "win":
            return cast(Array, optax.sigmoid_binary_cross_entropy(pred, y).mean())
        return jnp.mean((pred - y) ** 2)

    opt = optax.adamw(cfg["lr"], weight_decay=cfg["weight_decay"])
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m: Any, st: Any, xs: Sample, y: Array) -> Any:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, xs, y)
        updates, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), st, loss

    @eqx.filter_jit
    def predict(m: Any, xs: Sample) -> Array:
        return cast(Array, jax.vmap(m)(xs)[:, 0])

    select = "auc" if task == "win" else "r2"
    best = -np.inf
    best_metrics: dict[str, float] = {}
    rng = np.random.default_rng(cfg["seed"])
    ckpt = run.dir / "best.eqx"
    bs = cfg["batch_size"]
    wb = wandb.init(
        project=cfg["wandb_project"], name=f"{cfg['arch']}-{task}",
        mode=cfg["wandb_mode"], config=cfg, reinit=True, dir=str(run.dir),
    )  # fmt: skip
    try:
        for epoch in range(cfg["epochs"]):
            order = rng.permutation(n)
            losses = []
            for i in range(0, n - bs + 1, bs):
                idx = jnp.asarray(order[i : i + bs])
                batch = jax.tree.map(lambda x: x[idx], x_tr)  # noqa: B023
                model, opt_state, loss = step(model, opt_state, batch, y_tr[idx])
                losses.append(float(loss))
            if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1:
                pred = np.asarray(predict(model, x_val))
                m = _metrics(task, pred, y_val)
                train_loss = float(np.mean(losses)) if losses else float("nan")
                run.log(
                    epoch=epoch,
                    train_loss=train_loss,
                    **{f"val_{k}": v for k, v in m.items()},
                )
                wb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        **{f"val/{k}": v for k, v in m.items()},
                    }
                )
                if m[select] > best:
                    best = m[select]
                    best_metrics = m
                    eqx.tree_serialise_leaves(ckpt, model)
    finally:
        wb.finish()
    return {f"best_{k}": v for k, v in best_metrics.items()}
