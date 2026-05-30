# catan

A fast, batched implementation of the board game [Catan](https://www.catan.com/), built for reinforcement learning and large-scale simulation.

## Packages

| Package | Description |
|---|---|
| [`catan-engine`](packages/catan-engine/) | The game engine — board generation, rules, and actions. |
| [`catan-render`](packages/catan-render/) | A web app for viewing a board in the browser. |

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18 (only for `catan-render`)

## Getting started

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/). Install everything from the repo root:

```bash
uv sync
```

Then see each package's README for usage.
