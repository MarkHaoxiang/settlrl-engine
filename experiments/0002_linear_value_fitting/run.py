"""Linear value fitting over the hand-engineered features.

A framework for a class of experiments: optimize linear weights over
``BoardFeatures``, deploy them in one-step lookahead, and gate against the
hand-tuned weights. Pick a variant::

    uv run python experiments/0002_linear_value_fitting/run.py [variant]

- ``predict``   — fit weights to predict game outcomes vs a known opponent
- ``maximise``  — CEM search over weights for match win rate vs a known
  opponent
- ``self_play`` — iterated maximise: each round's opponent is the current
  champion (round 0: the hand weights), challengers accepted only by
  beating it; rounds are configurable
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import start_run
from value_fitting import HAND_WEIGHTS, run_experiment

BASE = {
    "seed": 0,
    "features": list(HAND_WEIGHTS),  # BoardFeatures names to optimize
    "collect": {"steps": 12_000, "batch_size": 64, "snapshot_every": 4},
    "maximise": {
        "iterations": 3,
        "population": 6,
        "elites": 3,
        "eval_games": 60,
        "sigma": 0.3,
    },
    "probe_games": 120,
    "bench_opponent": "greedy",
    "bench_games": 200,
    "gate_games": 300,
}

VARIANTS = {
    "predict": {"target": "predict", "opponent": "greedy", "rounds": 1},
    "maximise": {"target": "maximise", "opponent": "greedy", "rounds": 1},
    "self_play": {"target": "maximise", "opponent": "self", "rounds": 3},
}


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}]")
    config = {**BASE, **VARIANTS[variant], "variant": variant}
    run_experiment(start_run(Path(__file__).parent, config), config)


if __name__ == "__main__":
    main()
