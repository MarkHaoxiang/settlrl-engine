# catan-engine

## Rules

At the start of every session, read the official Catan rulebook:
www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf

## Checks

Before finishing any session, ensure the mypy checker passes:

```bash
uv run --package catan-engine mypy packages/catan-engine/src packages/catan-engine/tests
```
