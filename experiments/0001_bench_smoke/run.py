"""Bench smoke: greedy must beat random.

Hypothesis: the scripted greedy agent beats uniform-random play decisively
(2-player, seat-swapped). Known true from the strength ladder — this
experiment exists as the worked example of the contract: a manifest-pinned
run, streamed metrics, a saved bench verdict, and a gate asserted in code.

    uv run python experiments/0001_bench_smoke/run.py [key=value ...]
"""

import sys
from pathlib import Path

from settlrl_agents.cli import bench
from settlrl_learn.experiment import Config, Run, start_run


class BenchSmokeConfig(Config):
    a: str = "greedy"
    b: str = "random"
    players: int = 2
    games: int = 60
    batch_size: int = 32
    seed: int = 0
    gate: float = 0.70  # pass iff rate - 2*se >= gate


def run_bench(run: Run, cfg: BenchSmokeConfig) -> str:
    """Bench ``a`` vs ``b``, log the per-seat split, gate on the lower 2-sigma
    bound. Returns the verdict."""
    result = bench(
        cfg.a,
        cfg.b,
        n_games=cfg.games,
        players=cfg.players,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    for seat, (wins, episodes) in enumerate(result.by_position):
        run.log(seat=seat, wins=wins, episodes=episodes)
    run.save_json("bench.json", result._asdict())
    lower = result.rate - 2 * result.se
    verdict = "pass" if lower >= cfg.gate else "fail"
    run.finish(verdict, rate=result.rate, se=result.se, lower_2se=lower)
    return verdict


def main() -> None:
    cfg = BenchSmokeConfig.resolve({}, overrides=sys.argv[1:])
    run_bench(start_run(Path(__file__).parent, cfg.dump()), cfg)


if __name__ == "__main__":
    main()
