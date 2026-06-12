"""Linear value fitting, maximise target: CEM weight search vs greedy.

Hypothesis: cross-entropy search over the hand-tuned terms' weights, with
measured match win rate vs greedy as the objective, finds weights at least
as strong as the hand-tuned ones (the predict target plateaued just below
them — exp 0002).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import start_run
from _value_fitting import HAND_WEIGHTS, run_experiment

CONFIG = {
    "seed": 0,
    "opponent": "greedy",
    "features": list(HAND_WEIGHTS),
    "target": "maximise",
    "collect": {},  # unused under "maximise"
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
    run_experiment(start_run(Path(__file__).parent, CONFIG), CONFIG)


if __name__ == "__main__":
    main()
