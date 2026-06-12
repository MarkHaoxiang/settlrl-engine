"""Linear value fitting, predict target: outcome-fit weights vs greedy.

Hypothesis: logistic/NNLS weights fit on game outcomes against a known
opponent recover or beat the hand-tuned heuristic weights when deployed in
one-step lookahead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import start_run
from _value_fitting import HAND_WEIGHTS, run_experiment

CONFIG = {
    "seed": 0,
    "opponent": "greedy",  # the known opponent (data and target)
    "features": list(HAND_WEIGHTS),  # BoardFeatures names to fit
    "target": "predict",  # fit to predict outcomes ("maximise" searches)
    "collect": {"steps": 12_000, "batch_size": 64, "snapshot_every": 4},
    "maximise": {},  # unused under "predict"
    "probe_games": 120,
    "bench_games": 200,
    "gate_games": 300,
}


def main() -> None:
    run_experiment(start_run(Path(__file__).parent, CONFIG), CONFIG)


if __name__ == "__main__":
    main()
