"""Command-line tools over the agent registry (``compare``, ``bench``)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Sequence
from typing import NamedTuple

from catan_agents import POLICIES
from catan_agents.evaluate import evaluate
from catan_agents.policy import BeliefSpec, ObservationSpec, StatefulSpec
from catan_agents.value import make_heuristic


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


def build_spec(text: str) -> ObservationSpec | BeliefSpec | StatefulSpec:
    """An agent spec from a registry name or a JSON configuration.

    JSON shape: ``{"kind": <name>, "params": {<make kwargs>},
    "value": {<make_heuristic weights>}}`` — ``params`` are knob overrides for
    the family builder; ``value`` builds a reweighted heuristic for the
    families that take one.
    """
    if text in POLICIES:
        return POLICIES[text]
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"unknown agent {text!r} (a registry name or JSON spec)"
        ) from exc
    kind = doc.get("kind")
    if kind not in POLICIES:
        raise ValueError(
            f"unknown agent kind {kind!r} (choose from {sorted(POLICIES)})"
        )
    base = POLICIES[kind]
    overrides = dict(doc.get("params") or {})
    if doc.get("value") is not None:
        overrides["value"] = make_heuristic(**doc["value"])
    if not overrides:
        return base
    return dataclasses.replace(base, defaults={**base.defaults, **overrides})


class BenchResult(NamedTuple):
    """Outcome of a :func:`bench` run, from agent ``a``'s side."""

    wins_a: int
    wins_b: int
    episodes: int
    rate: float
    """``a``'s win share of all finished games (chance: 1 / players)."""

    se: float
    """Binomial standard error of ``rate``."""

    by_position: tuple[tuple[int, int], ...]
    """Per seating of ``a`` (seat 0 first): (a's wins, episodes)."""


def bench(
    spec_a: str,
    spec_b: str,
    *,
    n_games: int = 200,
    players: int = 2,
    batch_size: int = 32,
    seed: int = 0,
) -> BenchResult:
    """Head-to-head between two (possibly configured) agents.

    At 2 players the seats swap halfway; at 3-4, ``a`` rotates through every
    seat with ``b`` filling the rest. Specs are :func:`build_spec` strings.
    """
    a, b = build_spec(spec_a), build_spec(spec_b)
    per = n_games // players
    wins_a = wins_b = episodes = 0
    by_position: list[tuple[int, int]] = []
    for pos in range(players):
        agents: list[ObservationSpec | BeliefSpec | StatefulSpec] = [b] * players
        agents[pos] = a
        r = evaluate(
            agents,
            n_episodes=per if pos < players - 1 else n_games - per * (players - 1),
            batch_size=batch_size,
            seed=seed + pos,
        )
        wa = int(r.wins[pos])
        by_position.append((wa, r.episodes))
        wins_a += wa
        wins_b += r.episodes - wa
        episodes += r.episodes
    rate = wins_a / episodes if episodes else 0.0
    se = math.sqrt(rate * (1.0 - rate) / episodes) if episodes else 0.0
    return BenchResult(wins_a, wins_b, episodes, rate, se, tuple(by_position))


def _format_bench(result: BenchResult, players: int) -> str:
    pos = ", ".join(f"seat {i}: {w}/{n}" for i, (w, n) in enumerate(result.by_position))
    return (
        f"a={result.wins_a} b={result.wins_b} n={result.episodes} "
        f"rate={100 * result.rate:.1f}% se={100 * result.se:.1f}% "
        f"(chance {100 / players:.1f}%)\n  a seated at {pos}"
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
    bench_p = sub.add_parser(
        "bench",
        help="head-to-head between configured agents (names or JSON specs; "
        "seat-swapped at 2 players, seat-rotated at 3-4)",
    )
    bench_p.add_argument("spec_a")
    bench_p.add_argument("spec_b")
    bench_p.add_argument("--games", type=int, default=200)
    bench_p.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    bench_p.add_argument("--batch-size", type=int, default=32)
    bench_p.add_argument("--seed", type=int, default=0)
    bench_p.add_argument(
        "--json",
        action="store_true",
        help="emit the result as one JSON object (for experiment gates)",
    )
    args = parser.parse_args(argv)
    if args.command == "bench":
        result = bench(
            args.spec_a,
            args.spec_b,
            n_games=args.games,
            players=args.players,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if args.json:
            print(json.dumps({**result._asdict(), "players": args.players}))
        else:
            print(_format_bench(result, args.players))
        return
    result_c = compare(
        args.agent_a,
        args.agent_b,
        n_games=args.games,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(_format(result_c))
