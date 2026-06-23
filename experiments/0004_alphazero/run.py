"""AlphaZero self-play for 2-player Catan.

Hypothesis: a value+policy net trained by AlphaZero self-play (the search as its
own teacher) beats ``lookahead(heuristic)`` at 2p — the settlrl-learn Stage-1
gate. The loop itself lives in ``settlrl_learn.training``; this only composes it
with a config, per-iteration logging, and the gate verdict.

    uv run python experiments/0004_alphazero/run.py [variant] [key=value ...]
"""

import sys
from pathlib import Path
from typing import Literal

import jax
import wandb
from settlrl_learn import save_az_params
from settlrl_learn.experiment import Config, Run, start_run
from settlrl_learn.training import GNNBackend, MLPBackend, arena, learn


class AlphaZeroConfig(Config):
    seed: int = 0
    # net
    net: Literal["mlp", "gnn"] = "mlp"  # flat engineered MLP, or the board GNN
    width: int = 64
    depth: int = 2  # trunk hidden layers (GNN: readout-head hidden layers)
    layers: int = 3  # GNN message-passing layers (ignored by mlp)
    gnn_preset: str = "gn_global"  # settlrl_learn.nn.graphnet.PRESETS key
    # search
    num_simulations: int = 64
    max_num_considered_actions: int = 16
    temperature: float = 1.0
    # warm-up: the first `teacher_iters` iterations draw data from a fixed strong
    # heuristic search (GNN path only), the cold-start fix.
    teacher: bool = False
    teacher_iters: int = 0
    teacher_sims: int = 32
    # the setup phase is played by a fixed policy (GNN path); the net trains/acts
    # only on the main loop. setup_depth<=1 = lookahead opener (default, as strong
    # as deeper at 2p); >=2 = the probabilistic-expectimax search (>=3p / better
    # value), `setup_temperature` its opponent suboptimality.
    setup_depth: int = 1
    setup_temperature: float = 2.0
    setup_beam: int = 4
    # loop
    n_iterations: int = 20
    selfplay_samples: int = 2048
    selfplay_batch: int = 64
    train_steps: int = 200
    reuse: float = 0.0  # GNN: updates/iter = reuse*fresh/batch (0 -> fixed train_steps)
    eval_frac: float = 0.1  # GNN: held-out fraction for the val_* metrics
    # value-target blend (1-a)z + a*q (Canopy): a ramps 0 -> q_weight_max over
    # q_weight_ramp iters; q = the searched root value. 0 -> pure outcome z.
    q_weight_max: float = 0.0
    q_weight_ramp: int = 10
    # roll leaf = the exact 11-roll expectation (expected EV); else a single
    # sampled roll. Mutually exclusive with chance_nodes (which resolves rolls in-tree).
    expected_rolls: bool = True
    # explicit chance nodes in the search (dice always; dev-card buys when
    # dev_chance) — nature's move resolved in-tree, both in self-play and the arena.
    chance_nodes: bool = False
    dev_chance: bool = True
    # action-ordering lock-out (canonical order over a turn's builds/buys/trades)
    # to cut search-space transpositions; applied in self-play (env + search) and
    # the arena agent.
    ordered: bool = False
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
    resume_from: str = ""  # prior run dir to continue bit-exactly (its runstate.eqx)


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
    # The real run: shallow-but-wide self-play. The search is the per-move
    # *latency* (sequential sim expansions) and the batch is *throughput*
    # (vmapped lanes), so we cut sims + considered actions and push the batch
    # far up to saturate the GPU -- noisier per-position targets, but many more
    # diverse games per iteration to average over.
    "gnn_run": {
        "net": "gnn",
        "width": 96,
        "layers": 4,
        "n_iterations": 40,
        "selfplay_samples": 4096,
        "selfplay_batch": 256,
        "num_simulations": 24,
        "max_num_considered_actions": 8,
        "reuse": 2.0,
        "batch_size": 512,
        "arena_games": 64,
        "arena_every": 4,
        "arena_batch": 256,
        "arena_sims": 24,
        "checkpoint_every": 2,
    },
    # Warm-started run: the first `teacher_iters` iterations learn from a fixed
    # strong heuristic search (the cold-start fix), then self-play takes over.
    "gnn_warm": {
        "net": "gnn",
        "width": 96,
        "layers": 4,
        "n_iterations": 30,
        "teacher": True,
        "teacher_iters": 8,
        "teacher_sims": 32,
        "selfplay_samples": 4096,
        "selfplay_batch": 256,
        "num_simulations": 32,
        "max_num_considered_actions": 16,
        "reuse": 2.0,
        "batch_size": 512,
        # Cheap arena (GNN search is costly), gated to fire first at the end of the
        # 8-iter warm-up (i=7) -- the capacity probe -- then every 4 iters.
        "arena_games": 40,
        "arena_every": 4,
        "arena_batch": 64,
        "arena_sims": 24,
        "checkpoint_every": 2,
    },
    # gnn_warm + Canopy value-target blend: train value on (1-a)z + a*q, a ramping
    # to 0.5 over 12 iters once past the warm-up -- the audit's #1 lever for the
    # dice-variance-starved value head (the search now exposes its root q).
    "gnn_warm_qblend": {
        "net": "gnn",
        "width": 96,
        "layers": 4,
        "n_iterations": 30,
        "teacher": True,
        "teacher_iters": 8,
        "teacher_sims": 32,
        "selfplay_samples": 4096,
        "selfplay_batch": 256,
        "num_simulations": 32,
        "max_num_considered_actions": 16,
        "reuse": 2.0,
        "batch_size": 512,
        "q_weight_max": 0.5,
        "q_weight_ramp": 12,
        "arena_games": 40,
        "arena_every": 4,
        "arena_batch": 64,
        "arena_sims": 24,
        "checkpoint_every": 2,
    },
    # The full stack: warm-up + q-blend + explicit dice/dev chance nodes (search
    # plans past rolls, both in self-play and the arena). The combined bet that
    # the learned value finally converts what the stationary heuristic couldn't.
    "gnn_warm_qblend_chance": {
        "net": "gnn",
        "width": 96,
        "layers": 4,
        "n_iterations": 30,
        "teacher": True,
        "teacher_iters": 8,
        "teacher_sims": 32,
        "selfplay_samples": 4096,
        "selfplay_batch": 256,
        "num_simulations": 32,
        "max_num_considered_actions": 16,
        "reuse": 2.0,
        "batch_size": 512,
        "q_weight_max": 0.5,
        "q_weight_ramp": 12,
        "chance_nodes": True,
        "dev_chance": True,
        "arena_games": 40,
        "arena_every": 4,
        "arena_batch": 64,
        "arena_sims": 24,
        "checkpoint_every": 2,
    },
    # ~10h overnight run: no chance nodes / no expected-EV leaf (single sampled
    # roll), self-play batch at the measured throughput sweet spot (256), q-blend +
    # lr from Canopy, larger replay + samples/iter to amortize the per-iter recompile.
    # n_iterations is sized at launch from a quick calibration of the net-phase rate.
    "gnn_overnight": {
        "net": "gnn",
        "width": 96,
        "layers": 4,
        "n_iterations": 240,
        "teacher": True,
        "teacher_iters": 8,
        "teacher_sims": 32,
        "selfplay_samples": 16384,
        "selfplay_batch": 256,
        "num_simulations": 64,
        "max_num_considered_actions": 16,
        "expected_rolls": False,  # no expected EV (single sampled roll)
        "chance_nodes": False,  # no chance nodes
        "reuse": 2.0,
        "batch_size": 1024,
        "buffer_max": 200_000,
        "lr": 5e-4,
        "q_weight_max": 0.85,  # Canopy q-blend
        "q_weight_ramp": 60,  # Canopy ramp
        "arena_games": 40,
        "arena_every": 5,
        "arena_batch": 128,
        "arena_sims": 24,
        "checkpoint_every": 10,
    },
    "gnn_smoke": {
        "net": "gnn",
        "width": 16,
        "layers": 2,
        "num_simulations": 4,
        "max_num_considered_actions": 4,
        "n_iterations": 2,
        "teacher": True,
        "teacher_iters": 1,
        "teacher_sims": 4,
        "setup_depth": 2,
        "setup_beam": 4,
        "selfplay_samples": 8,
        "selfplay_batch": 4,
        "train_steps": 2,
        "batch_size": 4,
        "q_weight_max": 0.5,  # exercise the value-blend path in the smoke
        "q_weight_ramp": 1,
        "chance_nodes": True,  # exercise the dice+dev chance-node path in the smoke
        "ordered": True,  # exercise the action-ordering lock-out in the smoke
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
    """The board-GNN value+policy net (experiment 0003's recommendation) in the
    training loop, with the setup phase delegated to a fixed policy."""
    import equinox as eqx
    import numpy as np
    from settlrl_agents.value import heuristic_value
    from settlrl_learn.nn.board_gnn import BoardGNN
    from settlrl_learn.nn.graphnet import PRESETS

    base = PRESETS.get(cfg.gnn_preset, PRESETS["gn_global"])
    netcfg = base._replace(width=cfg.width, layers=cfg.layers, head_depth=cfg.depth)
    backend = GNNBackend(
        netcfg, setup_depth=cfg.setup_depth,
        setup_temperature=cfg.setup_temperature, setup_beam=cfg.setup_beam,
        chance_nodes=cfg.chance_nodes, dev_chance=cfg.dev_chance, ordered=cfg.ordered,
    )  # fmt: skip
    resume = None
    if cfg.resume_from:
        prior = Path(cfg.resume_from) / "runstate.eqx"
        resume = prior if prior.exists() else None
    wb = wandb.init(
        project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.dump(),
        reinit=True, dir=str(run.dir),
    )  # fmt: skip
    best = -1.0

    def on_iter(i: int, metrics: dict[str, float], model: BoardGNN) -> None:
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
        model = learn(
            backend,
            n_iterations=cfg.n_iterations,
            selfplay_samples=cfg.selfplay_samples,
            selfplay_batch=cfg.selfplay_batch,
            num_simulations=cfg.num_simulations,
            max_num_considered_actions=cfg.max_num_considered_actions,
            temperature=cfg.temperature,
            teacher_value=heuristic_value if cfg.teacher else None,
            teacher_iters=cfg.teacher_iters,
            teacher_sims=cfg.teacher_sims,
            buffer_max=cfg.buffer_max,
            buffer_min=cfg.batch_size,
            batch_size=cfg.batch_size,
            train_steps=cfg.train_steps,
            reuse=cfg.reuse,
            eval_frac=cfg.eval_frac,
            value_blend_max=cfg.q_weight_max,
            value_blend_ramp=cfg.q_weight_ramp,
            expected_rolls=cfg.expected_rolls,
            chance_nodes=cfg.chance_nodes,
            dev_chance=cfg.dev_chance,
            ordered=cfg.ordered,
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
            progress=True,
        )
    finally:
        wb.finish()

    winrate = arena(
        backend, model, n_games=cfg.arena_games, num_simulations=cfg.arena_sims,
        batch_size=cfg.arena_batch,
        max_num_considered_actions=cfg.max_num_considered_actions,
        seed=cfg.seed + 99,
    )  # fmt: skip
    verdict = "pass" if winrate >= cfg.gate_winrate else "fail"
    run.finish(
        verdict, arena_winrate=winrate, best_arena_winrate=best, gate=cfg.gate_winrate
    )


def run_experiment(run: Run, cfg: AlphaZeroConfig) -> None:
    if cfg.net == "gnn":
        run_gnn_experiment(run, cfg)
        return
    backend = MLPBackend(
        (cfg.width,) * cfg.depth, value_weight=cfg.value_weight,
        chance_nodes=cfg.chance_nodes, dev_chance=cfg.dev_chance, ordered=cfg.ordered,
    )  # fmt: skip

    # Resume: restore the prior run's RunState and continue its wandb run so the
    # dashboard is one unbroken curve.
    resume_dir = Path(cfg.resume_from) if cfg.resume_from else None
    resume_from = None
    wandb_id = None
    if resume_dir is not None:
        runstate = resume_dir / "runstate.eqx"
        resume_from = runstate if runstate.exists() else None
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

    def on_iter(i: int, metrics: dict[str, float], net: object) -> None:
        nonlocal best
        run.log(iteration=i, **metrics)
        wb.log({"iteration": i, **metrics}, step=i)  # explicit step: resume-safe
        winrate = metrics.get("arena_winrate")
        if winrate is not None and winrate > best:
            best = winrate
            save_az_params(run.dir / "best.npz", net)  # type: ignore[arg-type]

    try:
        # learn writes the full-state checkpoint (run.dir/runstate.eqx) for
        # bit-exact resume; resume_from continues a prior run's checkpoint.
        final = learn(
            backend,
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
            value_blend_max=cfg.q_weight_max,
            value_blend_ramp=cfg.q_weight_ramp,
            expected_rolls=cfg.expected_rolls,
            chance_nodes=cfg.chance_nodes,
            dev_chance=cfg.dev_chance,
            ordered=cfg.ordered,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            arena_games=cfg.arena_games,
            arena_every=cfg.arena_every,
            seed=cfg.seed,
            checkpoint_dir=run.dir,
            checkpoint_every=cfg.checkpoint_every,
            resume_from=resume_from,
            on_iter=on_iter,
            progress=True,
        )
    finally:
        wb.finish()

    save_az_params(run.dir / "params.npz", final)  # final net
    winrate = arena(
        backend,
        final,
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
