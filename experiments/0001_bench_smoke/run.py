"""Bench smoke: greedy must beat random.

Hypothesis: the scripted greedy agent beats uniform-random play decisively
(2-player, seat-swapped). Known true from the strength ladder — this
experiment exists as the worked example of the contract: a manifest-pinned
run, streamed metrics, a saved bench verdict, and a gate asserted in code.
"""

import sys
from pathlib import Path

from catan_agents.cli import bench

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import start_run

CONFIG = {
    "a": "greedy",
    "b": "random",
    "players": 2,
    "games": 60,
    "batch_size": 32,
    "seed": 0,
    "gate": 0.70,  # pass iff rate - 2*se >= gate
}


def main() -> None:
    run = start_run(Path(__file__).parent, CONFIG)
    result = bench(
        CONFIG["a"],
        CONFIG["b"],
        n_games=CONFIG["games"],
        players=CONFIG["players"],
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
    )
    for seat, (wins, episodes) in enumerate(result.by_position):
        run.log(seat=seat, wins=wins, episodes=episodes)
    run.save_json("bench.json", result._asdict())
    lower = result.rate - 2 * result.se
    verdict = "pass" if lower >= CONFIG["gate"] else "fail"
    run.finish(verdict, rate=result.rate, se=result.se, lower_2se=lower)


if __name__ == "__main__":
    main()
