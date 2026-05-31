# catan-reference

A plain-Python, gold-standard reference implementation of the
[Catan](https://www.catan.com) base-game rules.

It models a single game with ordinary Python objects and straightforward control
flow — no vectorisation, no performance tricks. The goal is to be an obviously
correct, readable statement of the rules that can serve as an independent oracle
for the optimised, JAX-native `catan-engine`.

## Usage

```python
from catan_reference import Game, Layout, Resource, Roll, BuildRoad

game = Game.new(layout, robber=desert_tile)   # a fresh game in setup
for action in game.legal_actions():           # every legal move right now
    ...
game.apply(SetupSettlement(vertex=12))        # mutate the game in place
print(game.total_vp(player=0))
```

Each action is a small dataclass. `game.legal_actions()` returns every legal
action; `game.is_legal(action)` checks one; `game.apply(action)` performs it,
advancing the turn, recomputing the Longest Road / Largest Army awards, and
ending the game when someone reaches 10 victory points.

### Injected randomness

The rules are deterministic *given* the random outcomes, so the reference does
not roll dice or draw cards itself — those outcomes are passed in on the action:

- `Roll(value=8)` — the dice total.
- `BuyDevelopmentCard(card=DevCard.KNIGHT)` — the card drawn.
- `MoveRobber(tile, victim, stolen=Resource.ORE)` / `PlayKnight(...)` — the card
  stolen from the victim.

This lets a test feed `catan-engine`'s realised outcomes into the reference and
compare the two engines step for step.

## Board coordinates

The board geometry is generated from cube coordinates `(q, r, s)`. Tile centres
sum to `0`; vertices sum to `±1`; an edge joins an adjacent `+1`/`-1` vertex
pair. This is the canonical hex-grid convention and matches `catan-engine`'s, so
a position can be translated between the two libraries by cube coordinate.
