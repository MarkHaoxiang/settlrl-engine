# catan

A fast, batched implementation of the board game [Catan](https://www.catan.com/), built for reinforcement learning and large-scale simulation.

## Packages

| Package | Description |
|---|---|
| [`catan-engine`](packages/catan-engine/) | The game engine — board generation, rules, and actions. |
| [`catan-reference`](packages/catan-reference/) | A plain-Python, gold-standard reference implementation of the rules, used as the differential test oracle for `catan-engine`. |
| [`catan-agents`](packages/catan-agents/) | Agents that play the game — heuristics and search (greedy, lookahead, MCTS) — plus a CLI for matches and benchmarks. |
| [`catan-learn`](packages/catan-learn/) | Learned value and policy functions that plug into the agents. |
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

## Parallel development

Each session (or agent) works in its own [git worktree](https://git-scm.com/docs/git-worktree) — a separate checkout of the same repository — so several branches can move at once without sharing a working tree:

```bash
./wt.sh learn-stage1     # checkout at ../catan-engine.wt/learn-stage1, new branch off main, synced
./wt.sh ls               # list worktrees
./wt.sh rm learn-stage1  # remove the checkout once merged (the branch is kept)
```

Every worktree gets its own venv; the JAX compilation cache is shared, so warm compiles carry across. Branches land on `main` via PR once CI is green.
