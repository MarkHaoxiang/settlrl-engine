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
- ``self_play_4p`` — the same ladder in a four-player arena (one challenger
  seat rotating against a champion table; acceptance and the gate clear the
  1/players chance line)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import start_run
from value_fitting import HAND_WEIGHTS, run_experiment

BASE = {
    "seed": 0,
    "players": 2,  # the optimization arena's player count
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
    "eval_players": [2, 4],  # deployment numbers per count (gate stays 2p)
    "games_multi": 240,  # 3-4p matches need n>=240 (seed-batch variance)
    "bench_opponent": "greedy",
    "bench_games": 200,
    "gate_games": 300,
}

VARIANTS = {
    "predict": {"target": "predict", "opponent": "greedy", "rounds": 1},
    "maximise": {"target": "maximise", "opponent": "greedy", "rounds": 1},
    "self_play": {"target": "maximise", "opponent": "self", "rounds": 3},
    "self_play_4p": {
        "target": "maximise",
        "opponent": "self",
        "rounds": 2,
        "players": 4,
        "maximise": {
            "iterations": 2,
            "population": 5,
            "elites": 2,
            "eval_games": 80,
            "sigma": 0.3,
        },
        "probe_games": 160,
        "eval_players": [4],
        "games_multi": 200,
    },
}


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if variant not in VARIANTS:
        raise SystemExit(f"usage: run.py [{'|'.join(VARIANTS)}]")
    config = {**BASE, **VARIANTS[variant], "variant": variant}
    run_experiment(start_run(Path(__file__).parent, config), config)


if __name__ == "__main__":
    main()
