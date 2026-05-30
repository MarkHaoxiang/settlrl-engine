# catan-engine library

## Source files

Keep this section up to date when adding, removing, or significantly changing a file.

- `tile.py` — `Tile` enum: the six resource types (sheep, wheat, wood, brick, ore, desert).
- `port.py` — `Port` enum: the six port types (one per resource plus the 3:1 general port).
- `layout.py` — Static board representation. Defines `BoardLayout` (tile resources, number tokens, port allocation), the module-level jaxtyping aliases for its arrays, and the static geometry maps: tile-vertex, port-vertex, edge-vertex, and the padded vertex→edge / vertex→neighbour / vertex→tile / vertex→port incidence maps (sentinel `NO_INDEX`). `make_layout()` randomly generates a batched `BoardLayout`.
- `state.py` — Mutable game state. Defines `GamePhase`, building-stock/win constants, `BoardState` (board occupancy, dev-card deck/hand/knights, turn-flow flags such as phase/current_player/has_rolled/free_roads, and the Longest Road / Largest Army holders plus a PRNG key), `make_board_state()` which initialises a fresh game in the setup phase, and `tree_select(mask, a, b)` — the per-leaf `where` over two single-game states that the action layer uses for branchless application. Players are 0-indexed except in `vertex_owner`/`edge_road` (player + 1, 0 = empty).
- `board.py` — Top-level board type `Board = tuple[BoardLayout, BoardState]` (static layout paired with mutable state) plus action-agnostic construction/shaping helpers used by tests and callers (`make_board`, `replicate`, `set_phase`, `to_main`, `give`, `place_settlement`/`place_road`/`place_city`, `give_dev_card`, `set_robber`).
- **Rule modules** — the single-game, **traceable/vmappable** rule helpers, split by domain (all written without `int()`/NumPy/Python branching on traced values; `NO_INDEX` sentinels handled by masking + clipped gathers; batch by wrapping callers in `jax.vmap`):
  - `geometry.py` — the static incidence maps as `int32` `jnp` arrays for traceable gather/index (`EDGE_V`, `V_EDGES`, `V_NBR`, `V_TILES`, `V_PORT`, `TILE_V`) plus the `NO_IDX` sentinel.
  - `economy.py` — build-cost vectors (`ROAD_COST_ARR` etc.), building counts, `roads_left`, `can_afford`, `pay`, `bank_stock`, and `player_total_vp`.
  - `placement.py` — settlement / road placement legality (`distance_rule_ok`, `settlement_connected`, `road_placeable`).
  - `awards.py` — Longest Road (the explicit-stack iterative DFS `longest_road_length` via `lax.while_loop`, edge-seeded) and Largest Army, plus the `recompute_*` award reassignment.
  - `trade.py` — maritime `port_ratio` (lives here, not `port.py`, to avoid the `layout`→`port` import cycle).
  - `dice.py` — `roll_dice` and the `distribute_resources` production payout (with the bank-cap rule).
  - `robber.py` — `robber_victim_mask` and the random `steal`.
  - `setup.py` — setup turn order (`SETUP_ORDER_ARR`, `N_SETUP`) and the 2nd-settlement `grant_setup_resources`.
  - `development.py` — dev-card rules `playable_dev` / `draw_dev_card` (separate from `dev_cards.py` to avoid the `state`→`dev_cards` import cycle).
- `action.py` — The engine's action layer. Defines `ActionResult` (SUCCESS/INVALID/GAME_COMPLETE), the `VecAction[Params]` base class, and **all 15** vectorized JAX-native actions (`SetupSettlement`, `SetupRoad`, `RollDice`, `Discard`, `MoveRobber`, `BuildRoad`, `BuildSettlement`, `BuildCity`, `BuyDevelopmentCard`, `PlayKnight`, `PlayRoadBuilding`, `PlayYearOfPlenty`, `PlayMonopoly`, `MaritimeTrade`, `EndTurn`; DomesticTrade is intentionally deferred). Each action's core is a single-game traceable transition; the public `is_available` / `__call__` are module-level `jit(vmap(...))`, so they run a whole batch at once (params are batched arrays — scalar params are `(batch,)`, multi-field params are tuples of `(batch,)` arrays — and the outcome is a `(batch,)` array of `ActionResult` codes). Application is branchless: the candidate state is always computed, then selected with `state.tree_select` against the `is_available` mask. Per-action tests live in `tests/actions/` (pytest fixtures in `tests/actions/conftest.py`; the shared ASCII `BoardRenderer` lives in `tests/render.py`).

**Engine path:** the engine is fully JAX-native (the rule modules above + `action.py`); the old NumPy single-game path has been removed. The trusted NumPy reference implementation lives in `tests/reference.py` as a differential test oracle (validated against the rule modules in `tests/test_rules.py`); it is not part of the shipped package.
- `resources.py` — Resource constants (`N_PLAYERS`, `N_RESOURCES`, `BANK_INITIAL`), build-cost tuples (`ROAD_COST`, `SETTLEMENT_COST`, `CITY_COST`), jaxtyping aliases for resource arrays, and `compute_bank_resources()` which derives remaining bank stock from player holdings.
- `dev_cards.py` — `DevCard` enum, deck counts (`DEV_CARD_COUNTS`), and `DEV_CARD_COST`. The dev-card *arrays* live on `BoardState` (see state.py).
- `env.py` — RL environment entry point. Exposes the batched `step(board, action_type, params)` (and `available(...)` legality mask) — a single `(ActionType, ActionParams)` interface over all 15 actions, `jit(vmap(...))` over the `lax.switch` dispatchers in `action.py`. Re-exports `ActionType`, `ActionParams`, `ActionResult`, `N_ACTION_TYPES`.
- `__init__.py` — Package entry point (`main`).
