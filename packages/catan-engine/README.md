# catan-engine

The game engine for Catan: it generates boards, knows the rules, and applies player actions to advance a game.

It is designed to run many games at once and to plug into reinforcement-learning training loops, so it favours a simple, uniform interface over many specialised entry points.

## Installation

From the repo root:

```bash
uv sync
```

## What it provides

- **Board setup** — randomly generated boards with terrain, number tokens, and ports.
- **Game state** — the full state of a game in progress: placements, resources, development cards, turn order, and the Longest Road / Largest Army awards.
- **Actions** — every move a player can make, from the opening placements through building, trading, playing development cards, and ending a turn.
- **Legality** — given a state, which actions are currently allowed.

A game is driven by repeatedly asking which actions are available and applying one of them. Many games can be advanced together in a single step.

## Testing

```bash
uv run --package catan-engine pytest
```
