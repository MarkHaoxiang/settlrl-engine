# 0001 — bench smoke

Status: concluded (pass)

## Hypothesis

The scripted greedy agent beats uniform-random play decisively (2-player,
seat-swapped). Known true from the strength ladder — the experiment exists to
prove the contract end to end: manifest-pinned run, streamed metrics, a saved
bench verdict, a gate asserted in code.

## Setup

`uv run python experiments/0001_bench_smoke/run.py` — config at the top of
run.py: greedy vs random, 60 games, seats swapped halfway, seed 0; gate
pass iff (rate − 2·se) ≥ 0.70.

## Results

From `runs/0001_bench_smoke/2026-06-12T154132Z` (RTX 5090, jax 0.10.1):
greedy won 54/65 (83.1%, se 4.7%), lower 2σ bound 73.8% — gate passed.
Seat-balanced: 27/32 seated first, 27/33 seated second.

## Decision

Infrastructure adopted. The pattern for every following experiment:
`_lib.start_run` for the manifest, `metrics.jsonl` for anything stepwise,
`catan_agents.cli.bench` (saved as `bench.json`) for strength claims, and the
verdict computed by the script.
