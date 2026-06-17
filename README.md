# settlrl

A fast, batched implementation of Settlrl, a hex-tile trading board game, built for reinforcement learning and large-scale simulation.

## Packages

| Package | Description |
|---|---|
| [`settlrl-engine`](packages/settlrl-engine/) | The game engine — board generation, rules, and actions, vectorised in JAX. |
| [`settlrl-game`](packages/settlrl-game/) | The shared game model: the plain-Python reference rules (the differential test oracle for `settlrl-engine`) plus the serialization / replay layer the app and bot service build on. Engine-free. |
| [`settlrl-agents`](packages/settlrl-agents/) | Agents that play the game — heuristics and search (greedy, lookahead, MCTS) — plus a CLI for matches and benchmarks, and (`[service]`) the bot service that serves their moves over HTTP. |
| [`settlrl-learn`](packages/settlrl-learn/) | Learned value and policy functions that plug into the agents. |
| [`settlrl-app`](packages/settlrl-app/) | The web game server: REST + SSE API, async game runtime, storage, auth, and the browser frontend. |

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18 (only for `settlrl-app`)

## Getting started

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/). Install everything from the repo root:

```bash
uv sync
```

Then see each package's README for usage.
