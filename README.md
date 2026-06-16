# settlrl

A fast, batched implementation of Settlrl, a hex-tile trading board game, built for reinforcement learning and large-scale simulation.

## Packages

| Package | Description |
|---|---|
| [`settlrl-engine`](packages/settlrl-engine/) | The game engine — board generation, rules, and actions. |
| [`settlrl-reference`](packages/settlrl-reference/) | A plain-Python, gold-standard reference implementation of the rules, used as the differential test oracle for `settlrl-engine`. |
| [`settlrl-agents`](packages/settlrl-agents/) | Agents that play the game — heuristics and search (greedy, lookahead, MCTS) — plus a CLI for matches and benchmarks. |
| [`settlrl-learn`](packages/settlrl-learn/) | Learned value and policy functions that plug into the agents. |
| [`settlrl-render`](packages/settlrl-render/) | A web app for viewing a board in the browser. |

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18 (only for `settlrl-render`)

## Getting started

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/). Install everything from the repo root:

```bash
uv sync
```

Then see each package's README for usage.
