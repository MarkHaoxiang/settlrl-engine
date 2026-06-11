# catan-engine

The game engine for Catan: it generates boards, knows the rules, and applies player actions to advance a game.

It is designed to run many games at once and to plug into reinforcement-learning training loops, so it favours a simple, uniform interface over many specialised entry points.

## Installation

From the repo root:

```bash
uv sync
```

The engine is device-agnostic; in the workspace a Linux `uv sync` installs the
CUDA jaxlib by default, so JAX picks up an NVIDIA GPU automatically (and falls
back to CPU without one). Standalone installs get the same via the `cuda`
extra: `pip install 'catan-engine[cuda]'`.

## What it provides

- **Board setup** — randomly generated boards with terrain, number tokens, and ports. Number tokens can be placed fully at random (default) or in the rulebook's alphabetical spiral (`number_placement="spiral"`), the balanced setup common in tournament play.
- **Game state** — the full state of a game in progress: placements, resources, development cards, turn order, and the Longest Road / Largest Army awards.
- **Actions** — every move a player can make, from the opening placements through building, trading, playing development cards, and ending a turn.
- **Legality** — given a state, which actions are currently allowed.
- **Environment** — a multi-agent environment that follows the PettingZoo Agent-Environment-Cycle conventions, with each player taking turns. It reports observations, rewards, legal-action masks, and when a game is over, and it runs a batch of games in parallel for reinforcement-learning rollouts (finished games restart automatically).
- **2–4 players** — games seat 4 players by default; pass `n_players` (2, 3, or 4) when creating a board or environment to play with fewer, under the unchanged base rules.
- **Beliefs** — optional card counting (`track_beliefs=True`): the environment tracks, for every player, provable bounds on what each opponent holds, using only information that player could see. Agents can read this honest view (`belief_view`) instead of the true state, so they never act on hidden cards.
- **Game records** — `catan_engine.record` saves a complete game as readable JSON (the configuration plus every move played, with dice outcomes) and replays it deterministically: `record_game` plays and records a game, `GameRecord.to_json` / `from_json` serialise it, and `replay` re-steps it move by move, verifying it against the engine as it goes.

A game is driven by repeatedly asking which actions are available and applying one of them. Many games can be advanced together in a single step.

## Testing

```bash
uv run --package catan-engine pytest
```

Performance benchmarks for the environments live in `tests/benchmark/` — see
[its README](tests/benchmark/README.md) for how to run and compare them.
