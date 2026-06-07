"""Command-line tools over the agent registry (``compare``; tournaments later)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import NamedTuple

from catan_agents import POLICIES
from catan_agents.shared.evaluate import evaluate


class CompareResult(NamedTuple):
    """Outcome of a :func:`compare` run (totals are per agent, in name order)."""

    names: tuple[str, str]
    wins: tuple[int, int]
    first_seat_wins: tuple[int, int]
    """Each agent's wins from the games where it was seated first."""

    first_seat_episodes: tuple[int, int]
    """How many games each agent was seated first in."""

    episodes: int


def compare(
    a: str,
    b: str,
    *,
    n_games: int = 100,
    batch_size: int = 32,
    seed: int = 0,
) -> CompareResult:
    """Head-to-head between two registered agents over ``n_games`` two-player
    games, seats swapped halfway to cancel the first-mover advantage.

    ``a`` / ``b`` are ``POLICIES`` names. Lanes finishing on the same step can
    overshoot ``n_games`` slightly; every finished game is counted.
    """
    for name in (a, b):
        if name not in POLICIES:
            raise ValueError(f"unknown agent {name!r} (choose from {sorted(POLICIES)})")
    half = n_games // 2
    r_ab = evaluate(
        [POLICIES[a], POLICIES[b]],
        n_episodes=half,
        batch_size=batch_size,
        seed=seed,
    )
    r_ba = evaluate(
        [POLICIES[b], POLICIES[a]],
        n_episodes=n_games - half,
        batch_size=batch_size,
        seed=seed + 1,
    )
    return CompareResult(
        names=(a, b),
        wins=(
            int(r_ab.wins[0]) + int(r_ba.wins[1]),
            int(r_ab.wins[1]) + int(r_ba.wins[0]),
        ),
        first_seat_wins=(int(r_ab.wins[0]), int(r_ba.wins[0])),
        first_seat_episodes=(r_ab.episodes, r_ba.episodes),
        episodes=r_ab.episodes + r_ba.episodes,
    )


def _format(result: CompareResult) -> str:
    (a, b), (wa, wb) = result.names, result.wins
    (fa, fb), (ea, eb) = result.first_seat_wins, result.first_seat_episodes
    rate = 100.0 * wa / result.episodes if result.episodes else 0.0
    return (
        f"{a} vs {b}: {wa}-{wb} over {result.episodes} games ({a} {rate:.1f}%)\n"
        f"  seated first: {a} {fa}/{ea}, {b} {fb}/{eb}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="catan-agents", description="Tools over the shipped agent registry."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_p = sub.add_parser(
        "compare", help="head-to-head between two agents (seat-swapped)"
    )
    cmp_p.add_argument("agent_a", choices=sorted(POLICIES))
    cmp_p.add_argument("agent_b", choices=sorted(POLICIES))
    cmp_p.add_argument("--games", type=int, default=100)
    cmp_p.add_argument("--batch-size", type=int, default=32)
    cmp_p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = compare(
        args.agent_a,
        args.agent_b,
        n_games=args.games,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(_format(result))
