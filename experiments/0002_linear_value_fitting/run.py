"""Linear value fitting over the hand-engineered features, vs a known opponent.

Hypothesis: weights optimized against a known opponent (greedy) — fit to
predict outcomes, or searched to maximise the match win rate — recover or
beat the hand-tuned heuristic weights when deployed in one-step lookahead.

The optimisation target is a config option::

    uv run python experiments/0002_linear_value_fitting/run.py            # predict
    uv run python experiments/0002_linear_value_fitting/run.py maximise   # search
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import start_run
from _value_fitting import HAND_WEIGHTS, run_experiment

CONFIG = {
    "seed": 0,
    "opponent": "greedy",  # the known opponent (data and objective)
    "features": list(HAND_WEIGHTS),  # BoardFeatures names to fit
    "target": "predict",  # or "maximise"; argv overrides
    "collect": {"steps": 12_000, "batch_size": 64, "snapshot_every": 4},
    "maximise": {
        "iterations": 3,
        "population": 6,
        "elites": 3,
        "eval_games": 60,
        "sigma": 0.3,
    },
    "probe_games": 120,
    "bench_games": 200,
    "gate_games": 300,
}


def main() -> None:
    if len(sys.argv) > 1:
        if sys.argv[1] not in ("predict", "maximise"):
            raise SystemExit("usage: run.py [predict|maximise]")
        CONFIG["target"] = sys.argv[1]
    run_experiment(start_run(Path(__file__).parent, CONFIG), CONFIG)


if __name__ == "__main__":
    main()
