# catan-engine

## Rules

At the start of every session, read the official Catan rulebook:
www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf

## Documentation

When you change code, check whether the relevant module-level docs (per-package `CLAUDE.md`) and READMEs still describe it accurately, and update them in the same change. Remove references that have gone stale.

**Code and types are documentation.** Never repeat in prose what a signature, type annotation, or name already states — if something is clearly understandable by reading the code, don't document it. Docs record only what code cannot express: invariants, design rationale, cross-module contracts, perf evidence, gotchas.

Array parameters and returns carry jaxtyping annotations. Reuse the shared alias — defined beside the constants that pin its dimensions — instead of bare `jax.Array` or a local redefinition; bare `jax.Array` is for the rare genuinely shape/dtype-polymorphic case. The test conftests turn these annotations into enforced runtime checks for the hooked modules, so they must be exact, not aspirational.

Keep docs concise. User-facing docs (READMEs) should describe what something does and how to use it — no implementation details — and keep abstractions clear; leave internal/technical notes to `CLAUDE.md`.

Comments should be concise. Doc comments (docstrings) describe only the contract to callers — behavior not evident from the signature; no implementation detail, design motivation, or perf notes (those belong in the per-package `CLAUDE.md`, or are simply omitted).

## Parallel development

The main checkout belongs to the user — his editor saves land there
mid-session. For any multi-step change, work in your own worktree
(`./wt.sh <branch>`, see the README) and land it on `main` via PR once CI is
green. The GPU is the one shared resource between worktrees: at most one
bench/eval session uses it at a time; everything else runs
`JAX_PLATFORMS=cpu`.

## Checks

Pre-commit hooks (ruff check/format, mypy over every package, the engine test
suite) run on each commit — `uv run pre-commit install` after a fresh clone.
CI (`.github/workflows/ci.yml`) runs the full gate on push/PR: lint, format
check, mypy, and every package's test suite (including catan-agents, whose
suite is too slow for a commit hook).

Before finishing any session, ensure the mypy checker passes:

```bash
uv run --package catan-engine mypy packages/catan-engine/src packages/catan-engine/tests
uv run --package catan-agents mypy packages/catan-agents/src packages/catan-agents/tests
```

When CUDA is available (check `jax.devices("cuda")` or `nvidia-smi`), always run benchmarks directly on the GPU (`-k cuda`) — skip the CPU benchmark runs. Without CUDA, run CPU-only (`JAX_PLATFORMS=cpu`, or `-k cpu` for the benchmarks).
