# settlrl-game library

The Settlrl game model shared by `settlrl-app` and the bot service. It is
engine-free (depends only on the reference rules below + pydantic), so both the
web app and the agents' bot service can build on it without pulling in the JAX
engine, and `settlrl-engine`'s tests can use the reference as their oracle
without a dependency cycle.

Two layers: the reference rules (`settlrl_game.reference`, below) and the
serialization/replay layer over them — `session.py` (`GameSession`: a live game
driven by a stable flat action index), `actions.py` (the flat space + decode,
plus the `move_for_flat` / `flat_for_move` / `legal_moves` translation to the
structured `MoveModel`), `convert.py` (reference `Game` → `BoardModel`),
`models.py` (the pydantic wire models), `record.py` (replayable records), and
`botproto.py` (the bot-service wire protocol: `BotInfo`, the structured
`MoveModel` in board coordinates, and the incremental `ActRequest`/`ActResponse`).
The flat action indexing and the record format are the contract the app and bot
service agree on; keep them stable. Flat indices stay internal — the bot wire
speaks `MoveModel` (cube/axial coordinates), so it survives engine reindexing.

## `settlrl_game.reference` — the reference rules

A plain-Python gold-standard implementation of the Settlrl base game, written from
the official rulebook. It is the independent oracle for `settlrl-engine`: the two
are driven with the same action stream and their states compared (see
`packages/settlrl-engine/tests/conversion.py` and `test_reference_equivalence.py`).
Optimise for correctness and clarity, never speed; no jax/numpy. The source files
below live under `src/settlrl_game/reference/`.

## Source files

- `types.py` — independent enums (`Resource`, `DevCard`, `PortType`, `Building`,
  `Phase`) and every rulebook constant (build costs, bank size, dev-deck counts,
  piece caps, win/award thresholds).
- `board.py` — board geometry generated from cube coordinates, independently of
  the engine. Tiles, vertices, edges and all adjacency tables are built once at
  import with this module's own indexing; cube lookups (`vertex_cube`,
  `cube_to_vertex`, `tile_cube`, `edge_vertices`, `edge_between`, …) are the
  bridge the engine-side conversion uses. `Layout` carries the variable board
  (per-tile resource + number token, and the harbours as `Port`s);
  `random_layout(rng, number_placement)` shuffles the standard allotment over
  the fixed geometry — `"random"` shuffles the number tokens, `"spiral"` lays
  them along the rulebook spiral (`SPIRAL_NUMBERS`); terrain and ports depend
  only on the rng, so a seed's map is identical across both modes. The nine
  harbour positions are baked in by cube coordinate (physical board data, shared
  with the engine like the cube convention itself — not an import).
- `game.py` — the engine. `Player` and `Game` hold the state; one frozen
  dataclass per action; `Game.legal_actions()` / `is_legal()` / `apply()` drive
  play, with the rule logic (placement, `longest_road_length`, `production`,
  `port_ratio`, award recomputation, turn flow) written straight from the
  rulebook. Games seat `n_players` (2..4, default 4; `Game.new(layout, robber,
  n_players)`): `len(players) == n_players`, and the setup snake
  (`setup_order(n_players)`), turn rotation, discard, production, monopoly and
  award loops all run over it. Stochastic actions carry their realised outcome (`Roll.value`,
  `BuyDevelopmentCard.card`, the robber's `stolen` card) — see README.
- `belief.py` — card counting: per-observer lower/upper bounds on every
  player's per-resource hand, derivable from public information alone, plus the
  public played-dev tally. `Belief.update(before, after, action)` advances one
  transition. Hidden info is only a robber steal's card *type* (the rest is
  public), so the bounds open only on steals a third party didn't witness and
  stay exact with two players. This is the readable oracle for the engine's
  `belief.py` (`settlrl-engine`'s `test_reference_equivalence` checks the two
  agree bound-for-bound) — it knows nothing about the engine's representation.
- `chance.py` — sample the stochastic outcomes the game's actions take
  (`roll_dice`, `draw_dev_card`, `steal`) for a live driver; the differential
  test injects the engine's realised outcomes instead.

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
- **Development cards play any time during the turn** (`_dev_play_window`):
  knight *and* the progress cards are legal in ROLL (pre-roll) as well as
  MAIN, per the rulebook's "you can play the card at any time". One
  liberality: Road Building's free roads are placeable immediately (even
  pre-roll) but may also be *deferred* past the roll — the rulebook says
  place immediately, but forcing that could stall a game whose owner has no
  placeable edge, so rolling stays legal while free roads are owed.
- **You can only win during your own turn** (rulebook p.5, `_check_win`): the
  win check reads the *current* player only, and `_apply_end_turn` re-checks
  after rotating — a player who reached 10 VP out of turn (a settlement break
  handing them Longest Road) claims victory at the start of their next turn,
  and play continues until then.
- **Domestic trade carries arbitrary bundles**
  (`ProposeTrade(partner, give, receive)` with per-resource count tuples →
  `AcceptTrade` / `RejectTrade`, through `Phase.TRADE_RESPONSE`), matching
  the engine's packed-params encoding. Both sides must give something and no
  resource may appear on both sides. Proposing is gated on *public*
  information only (the proposer holds the give bundle, the partner's hand
  covers the receive total); whether the partner holds the asked-for cards
  is settled by Accept (illegal without them) / Reject (always legal).
  `legal_actions()` enumerates only the 1:1 subset (the differential driver's
  choice set); `is_legal` / `apply` take any bundle. Disabled in 2-player
  games.

## Checks

```bash
uv run --package settlrl-game mypy packages/settlrl-game/src
```
