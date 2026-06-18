"""AlphaZero self-play for 2-player Catan.

Hypothesis: a value+policy net trained by AlphaZero self-play (the search as its
own teacher) beats ``lookahead(heuristic)`` at 2p — the settlrl-learn Stage-1
gate. The loop itself lives in ``settlrl_learn.alphazero``; this only composes it
with a config, per-iteration logging, and the gate verdict.

    uv run python experiments/0004_alphazero/run.py [variant] [key=value ...]
"""

import sys
from pathlib import Path
from typing import Literal

import jax
import wandb
from settlrl_learn import init_az_params, save_az_params
from settlrl_learn.alphazero import arena, learn
from settlrl_learn.experiment import Config, Run, start_run
from settlrl_learn.train_state import TrainState


class AlphaZeroConfig(Config):
    seed: int = 0
    # net
    net: Literal["mlp", "gnn"] = "mlp"  # flat engineered MLP, or the board GNN
    width: int = 64
    depth: int = 2  # trunk hidden layers (GNN: readout-head hidden layers)
    layers: int = 3  # GNN message-passing layers (ignored by mlp)
    gnn_preset: str = "gn_global"  # settlrl_learn.graphnet.PRESETS key
    # search
    num_simulations: int = 64
    max_num_considered_actions: int = 16
    temperature: float = 1.0
    # loop
    n_iterations: int = 20
    selfplay_samples: int = 2048
    selfplay_batch: int = 64
    train_steps: int = 200
    reuse: float = 0.0  # GNN: updates/iter = reuse*fresh/batch (0 -> fixed train_steps)
    eval_frac: float = 0.1  # GNN: held-out fraction for the val_* metrics
    batch_size: int = 256
    buffer_max: int = 50_000
    buffer_min: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    # gate
    arena_games: int = 80
    arena_every: int = 5
    arena_batch: int = 128  # GNN arena: many parallel lanes (fast) ...
    arena_sims: int = 48  # ... at a modest sim budget (decoupled from training)
    gate_winrate: float = 0.55  # pass iff the final net clears this vs lookahead
    # logging / checkpointing
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = "settlrl-0004-alphazero"
    checkpoint_every: int = 5  # iterations between full-state checkpoints
    resume_from: str = ""  # prior run dir to continue bit-exactly (its trainstate)


VARIANTS: dict[str, dict[str, object]] = {
    "default": {},
    # The experiment-0003 GNN (gn_global) as the value+policy trunk, small budget.
    "gnn": {
        "net": "gnn",
        "width": 64,
        "layers": 3,
        "n_iterations": 12,
        "selfplay_samples": 1024,
        "selfplay_batch": 64,
        "reuse": 3.0,  # ~12 updates/iter, not 150 -> the value head can't memorize
        "num_simulations": 32,
        "arena_games": 40,
        "arena_every": 3,
    },
    # A genuinely larger run: more iterations + deeper search (sharper policy
    # targets) to test whether the loop bootstraps past 0 arena. Checkpointed
    # every 2 iters (resume via resume_from=<run dir>).
    "gnn_large": {
        "net": "gnn",
        "width": 64,
        "layers": 3,
        "n_iterations": 24,
        "selfplay_samples": 1536,
        "selfplay_batch": 96,
        "num_simulations": 48,
        "reuse": 2.5,
        "arena_games": 48,
        "arena_every": 4,
        "checkpoint_every": 2,
    },
    "gnn_smoke": {
        "net": "gnn",
        "width": 16,
        "layers": 2,
        "num_simulations": 4,
        "max_num_considered_actions": 4,
        "n_iterations": 1,
        "selfplay_samples": 8,
        "selfplay_batch": 4,
        "train_steps": 2,
        "batch_size": 4,
        "arena_games": 4,
        "arena_every": 1,
        "wandb_mode": "disabled",
    },
    "smoke": {
        "width": 16,
        "depth": 1,
        "num_simulations": 4,
        "max_num_considered_actions": 4,
        "n_iterations": 1,
        "selfplay_samples": 8,
        "selfplay_batch": 4,
        "train_steps": 2,
        "batch_size": 4,
        "buffer_min": 4,
        "arena_games": 4,
        "arena_every": 1,
        "wandb_mode": "disabled",
        "checkpoint_every": 1,
    },
}


def run_gnn_experiment(run: Run, cfg: AlphaZeroConfig) -> None:
    """The board-GNN value+policy net (experiment 0003's recommendation) in a
    small AlphaZero loop (in-memory replay; the flat-MLP path keeps the bit-exact
    flashbax/orbax infra)."""
    import equinox as eqx
    import numpy as np
    from settlrl_learn import azgnn
    from settlrl_learn.graphnet import PRESETS

    base = PRESETS.get(cfg.gnn_preset, PRESETS["gn_global"])
    netcfg = base._replace(width=cfg.width, layers=cfg.layers, head_depth=cfg.depth)
    resume = None
    if cfg.resume_from:
        prior = Path(cfg.resume_from) / "gnnstate.eqx"
        resume = prior if prior.exists() else None
    wb = wandb.init(
        project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.dump(),
        reinit=True, dir=str(run.dir),
    )  # fmt: skip
    best = -1.0

    def on_iter(i: int, metrics: dict[str, float], model: azgnn.AZGraphNet) -> None:
        nonlocal best
        run.log(iteration=i, **metrics)  # scalars -> metrics.jsonl
        log: dict[str, object] = {"iteration": i, **metrics}
        # param distributions as wandb histograms (whole net + each head, where a
        # collapse shows first).
        for name, tree in (
            ("params/all", model),
            ("params/policy", model.policy),
            ("params/value", model.value),
        ):
            arrs = [
                np.asarray(x).ravel()
                for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))
            ]
            if arrs:
                log[name] = wandb.Histogram(np.concatenate(arrs))  # type: ignore[arg-type]
        wb.log(log, step=i)
        winrate = metrics.get("arena_winrate")
        if winrate is not None and winrate > best:
            best = winrate
            eqx.tree_serialise_leaves(run.dir / "best.eqx", model)

    try:
        model = azgnn.learn(
            cfg=netcfg,
            n_iterations=cfg.n_iterations,
            selfplay_samples=cfg.selfplay_samples,
            selfplay_batch=cfg.selfplay_batch,
            num_simulations=cfg.num_simulations,
            max_num_considered_actions=cfg.max_num_considered_actions,
            temperature=cfg.temperature,
            buffer_max=cfg.buffer_max,
            batch_size=cfg.batch_size,
            train_steps=cfg.train_steps,
            reuse=cfg.reuse,
            eval_frac=cfg.eval_frac,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            arena_games=cfg.arena_games,
            arena_every=cfg.arena_every,
            arena_batch=cfg.arena_batch,
            arena_sims=cfg.arena_sims,
            seed=cfg.seed,
            checkpoint_dir=run.dir,
            checkpoint_every=cfg.checkpoint_every,
            resume_from=resume,
            on_iter=on_iter,
        )
    finally:
        wb.finish()

    winrate = azgnn.arena(
        model, n_games=cfg.arena_games, num_simulations=cfg.arena_sims,
        batch_size=cfg.arena_batch,
        max_num_considered_actions=cfg.max_num_considered_actions, seed=cfg.seed + 99,
    )  # fmt: skip
    verdict = "pass" if winrate >= cfg.gate_winrate else "fail"
    run.finish(
        verdict, arena_winrate=winrate, best_arena_winrate=best, gate=cfg.gate_winrate
    )


def run_experiment(run: Run, cfg: AlphaZeroConfig) -> None:
    if cfg.net == "gnn":
        run_gnn_experiment(run, cfg)
        return
    params = init_az_params(jax.random.key(cfg.seed), (cfg.width,) * cfg.depth)

    # Resume: restore the prior run's TrainState and continue its wandb run so
    # the dashboard is one unbroken curve.
    resume_dir = Path(cfg.resume_from) if cfg.resume_from else None
    resume_from = None
    wandb_id = None
    if resume_dir is not None:
        trainstate = resume_dir / "trainstate"
        resume_from = trainstate if trainstate.exists() else None
        id_file = resume_dir / "wandb_id.txt"
        wandb_id = id_file.read_text().strip() if id_file.exists() else None

    wb = wandb.init(
        project=cfg.wandb_project,
        mode=cfg.wandb_mode,
        config=cfg.dump(),
        reinit=True,
        dir=str(run.dir),
        id=wandb_id,
        resume="allow" if wandb_id else None,
    )
    (run.dir / "wandb_id.txt").write_text(str(wb.id))  # so a later run can resume it

    best = -1.0  # best arena win rate seen -> best.npz (the shippable net)

    def on_iter(i: int, metrics: dict[str, float], state: TrainState) -> None:
        nonlocal best
        run.log(iteration=i, **metrics)
        wb.log({"iteration": i, **metrics}, step=i)  # explicit step: resume-safe
        winrate = metrics.get("arena_winrate")
        if winrate is not None and winrate > best:
            best = winrate
            save_az_params(run.dir / "best.npz", state.params)  # strongest net so far

    try:
        # learn writes the full-state checkpoint (run.dir/trainstate) for
        # bit-exact resume; resume_from continues a prior run's checkpoint.
        final = learn(
            params,
            n_iterations=cfg.n_iterations,
            selfplay_samples=cfg.selfplay_samples,
            selfplay_batch=cfg.selfplay_batch,
            num_simulations=cfg.num_simulations,
            max_num_considered_actions=cfg.max_num_considered_actions,
            temperature=cfg.temperature,
            buffer_max=cfg.buffer_max,
            buffer_min=cfg.buffer_min,
            batch_size=cfg.batch_size,
            train_steps=cfg.train_steps,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            value_weight=cfg.value_weight,
            arena_games=cfg.arena_games,
            arena_every=cfg.arena_every,
            seed=cfg.seed,
            checkpoint_dir=run.dir,
            checkpoint_every=cfg.checkpoint_every,
            resume_from=resume_from,
            on_iter=on_iter,
        )
    finally:
        wb.finish()

    save_az_params(run.dir / "params.npz", final.params)  # final net
    winrate = arena(
        final.params,
        n_games=cfg.arena_games,
        num_simulations=cfg.num_simulations,
        max_num_considered_actions=cfg.max_num_considered_actions,
        seed=cfg.seed + 99,
    )
    verdict = "pass" if winrate >= cfg.gate_winrate else "fail"
    run.finish(
        verdict, arena_winrate=winrate, best_arena_winrate=best, gate=cfg.gate_winrate
    )


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "default"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}] [key=value ...]")
    cfg = AlphaZeroConfig.resolve(VARIANTS[variant], overrides=sys.argv[2:])
    run_experiment(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
