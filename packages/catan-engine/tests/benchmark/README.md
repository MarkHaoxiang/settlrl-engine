# Benchmarks

Throughput benchmarks for the RL environments under random legal play, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/). Two rollouts are
measured:

- **`test_batched_env_random_rollout`** — a batch of games stepped in lockstep
  through `BatchedCatanEnv` (the vectorised surface), one random legal action per
  lane per step.
- **`test_aec_env_random_rollout`** — a single game played turn-at-a-time through
  the PettingZoo `CatanAECEnv`.

Both pick only *legal* moves (screened by the engine's own action mask), so every
action type is exercised, including the forced discard / move-robber after a 7.

## Running

From the repo root (`uv sync` first if you haven't), use the wrapper script:

```bash
./run_benchmarks.sh
```

It runs `pytest packages/catan-engine/tests/benchmark` with:

- `--benchmark-only` — skips the regular test suite and runs just the benchmarks.
- `--no-cov` — turns off coverage (on by default via `addopts`), which otherwise
  instruments the hot loop and distorts the timings.

Any extra arguments are passed straight through to pytest-benchmark (see below).
JIT compilation is warmed up before each timed region, so the reported numbers
are steady-state throughput, not first-call latency.

## Useful options

```bash
# Save this run as a named baseline ...
./run_benchmarks.sh --benchmark-save=baseline

# ... then compare a later run against it (fails if it regresses past 5%).
./run_benchmarks.sh --benchmark-compare --benchmark-compare-fail=mean:5%

# Run a single benchmark.
./run_benchmarks.sh -k batched
```

See `pytest-benchmark --help` for the full set of `--benchmark-*` flags.
