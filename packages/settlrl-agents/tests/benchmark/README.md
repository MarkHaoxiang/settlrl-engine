# Benchmarks

Latency/throughput benchmarks for the shipped agents, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/). Three measures,
each over every agent in the `POLICIES` registry at its shipped (default)
parameters:

- **`test_agent_move[name-BN-dev]`** — steady-state latency of one jitted
  decision on a fixed mid-game position, swept over batch sizes `N` ∈ {1, 32}
  (grouped per batch size, so agents compare directly).
- **`test_sample_world[BN-dev]`** — the determinization primitive: filling a
  `BeliefView` per lane into a playable world.
- **`test_selfplay_window[name-dev]`** — throughput of one fused self-play
  window through the engine's `rollout(actor=...)` seam, the hot loop of
  `evaluate` / the CLI match tools.

All run 2-player games and are swept over devices: `cpu` always, plus `cuda`
when an NVIDIA GPU is usable (the workspace installs the CUDA jaxlib by
default on Linux) — otherwise the CUDA variants skip. Each variant pins its device explicitly, and JIT is
warmed up before every timed region.

These tests carry the `benchmark` marker and are **deselected from the default
`pytest` run**. Run them from the repo root via the shared wrapper (extra
arguments pass through to pytest-benchmark):

```bash
./run_benchmarks.sh                 # engine + agents benchmarks
./run_benchmarks.sh -k cuda         # GPU variants only
./run_benchmarks.sh -k "mcts and cuda"
./run_benchmarks.sh --benchmark-save=baseline
./run_benchmarks.sh --benchmark-compare --benchmark-compare-fail=mean:5%
```

or directly:

```bash
uv run --package settlrl-agents pytest packages/settlrl-agents/tests/benchmark \
    -m benchmark --benchmark-only -n 0
```
