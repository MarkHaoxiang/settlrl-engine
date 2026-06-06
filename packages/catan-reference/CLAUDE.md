# catan-reference library

A plain-Python gold-standard implementation of the Catan base game, written from
the official rulebook. It is the independent oracle for `catan-engine`: the two
are driven with the same action stream and their states compared (see
`packages/catan-engine/tests/conversion.py` and `test_reference_equivalence.py`).
Optimise for correctness and clarity, never speed; no jax/numpy.

## Source files

- `types.py` — independent enums (`Resource`, `DevCard`, `PortType`, `Building`,
  `Phase`) and every rulebook constant (build costs, bank size, dev-deck counts,
  piece caps, win/award thresholds).
- `board.py` — board geometry generated from cube coordinates, independently of
  the engine. Tiles, vertices, edges and all adjacency tables are built once at
  import with this module's own indexing; cube lookups (`vertex_cube`,
  `cube_to_vertex`, `tile_cube`, `edge_vertices`, `edge_between`, …) are the
  bridge the engine-side conversion uses. `Layout` carries the variable board
  (per-tile resource + number token, and the harbours as `Port`s).
- `game.py` — the engine. `Player` and `Game` hold the state; one frozen
  dataclass per action; `Game.legal_actions()` / `is_legal()` / `apply()` drive
  play, with the rule logic (placement, `longest_road_length`, `production`,
  `port_ratio`, award recomputation, turn flow) written straight from the
  rulebook. Games seat `n_players` (2..4, default 4; `Game.new(layout, robber,
  n_players)`): `len(players) == n_players`, and the setup snake
  (`setup_order(n_players)`), turn rotation, discard, production, monopoly and
  award loops all run over it. Stochastic actions carry their realised outcome (`Roll.value`,
  `BuyDevelopmentCard.card`, the robber's `stolen` card) — see README.

## Deliberate rulebook choices worth noting

- **Longest Road ties** (`recompute_longest_road`): the holder keeps the card
  while still tied for the longest road; if the holder is beaten and 2+ players
  tie for the new longest, the card is set aside (no holder). This follows the
  rulebook and is *not* the same as awarding ties to a fixed player — a point of
  difference worth checking against the engine.
- **Largest Army** is taken only by a *strictly* larger army; the holder keeps it
  on a tie.
- The robber: when any opponent with cards borders the target tile you must steal
  from one of them; only when there is no such victim do you steal nothing.
- **Discarding is one card per action** (`Discard(player, resource)`): the
  rulebook's simultaneous half-hand discard is serialized into single-card
  steps, which cannot change any outcome (each player's choice is independent).
  `is_legal` accepts any owing player; `legal_actions()` enumerates only the
  lowest-indexed owing player, matching the engine's fixed order so the
  differential driver exercises identical action streams.

## Checks

```bash
uv run --package catan-reference mypy packages/catan-reference/src
```
