# catan-engine

## Rules

At the start of every session, read the official Catan rulebook:
www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf

## Documentation

When you change code, check whether the relevant module-level docs (per-package `CLAUDE.md`) and READMEs still describe it accurately, and update them in the same change. Remove references that have gone stale.

Keep docs concise. User-facing docs (READMEs) should describe what something does and how to use it — no implementation details — and keep abstractions clear; leave internal/technical notes to `CLAUDE.md`.

## Checks

Before finishing any session, ensure the mypy checker passes:

```bash
uv run --package catan-engine mypy packages/catan-engine/src packages/catan-engine/tests
```
