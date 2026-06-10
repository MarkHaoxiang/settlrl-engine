# catan-engine library

## Layout

Three subpackages, layered so imports only ever point "down"
(`env` → `mechanics` → `board`; no cycles):

- `board/` — **data**: primitives (enums + constant tables) and the
  static/dynamic representation (`BoardLayout` / `BoardState`).
- `mechanics/` — **logic**: traceable single-game rule helpers and the action
  layer on top (`action.py` dispatch, `flat.py` flat action space).
- `env/` — **RL surface**: the batched env and the single-game PettingZoo-AEC
  wrapper.

Top-level: `belief.py` (card counting, beside `mechanics`; only `env` imports
it) and `record.py` (game records, above `env`).

Keep this file up to date when adding, removing, or significantly changing a
file — but record only what the code cannot show (invariants, rationale,
contracts, gotchas); code and types are the documentation for the rest.

## Conventions

- jaxtyping aliases live beside the constants that pin their dimensions —
  batched `*Array` forms and single-game `*Vec` / `*Scalar` counterparts (the
  batch axis is stripped under `vmap`). Check for an existing alias before
  defining one.
- Players are 0-indexed except in `vertex_owner` / `edge_road` (player + 1,
  0 = empty).
- Per-player arrays are sized to the seated count (`n_players` 2..4): a
  2-player game carries no padding rows. `BoardState.n_players` reads the
  count off the player axis as a static Python int; every game in a batch
  seats the same number.
- State arithmetic is committed through the saturating cast `state.to_u8`
  (uint8 wraps past 255 otherwise).
- Rule code is traceable: no `int()` / NumPy / Python branching on traced
  values. Vertex incidence is derived by scattering per-edge/-tile values over
  the dense `EDGE_V` / `TILE_V` / `PORT_V` maps (PyG-style message passing) —
  no ragged reverse maps or padding sentinels exist anywhere.
- `tests/conftest.py` installs the jaxtyping/beartype import hook for the rule
  modules and `belief.py`, turning their single-game annotations into runtime
  shape/dtype checks (per call and per jit trace). `common` / `action` /
  `flat` / `env` are excluded: their batched annotations describe arrays that
  run unbatched under `vmap`.

## board/

- `layout.py` — the "spiral" number placement consumes the same key stream as
  "random", so a seed's terrain/ports are identical across modes; the fixed
  spiral start corner is distributionally irrelevant (the terrain shuffle is
  rotation/reflection-invariant). The cube ↔ index lookups are host-only (not
  traceable); `tests/render.py` reuses them rather than re-deriving geometry.
- `dev_cards.py` — the dev-card *arrays* live on `BoardState`, not here.
- `__init__.py` — `Board = tuple[BoardLayout, BoardState]` plus construction
  helpers for tests and callers; `make_board` seeds the robber on the desert
  (rulebook start).

## mechanics/

**Core convention.** Each topical rule module (`placement`, `longest_road`,
`awards`, `dice`, `robber`, `setup`, `trade`, `development`, `turn`) holds its
rule helpers and the action cores built on them. A core exposes the batched
public `<name>_available` / `<name>_step` plus the private single-game
`_<name>_avail` / `_<name>_apply` used by the dispatch. `_<name>_apply` takes
the precomputed `available` legality instead of computing it: under `vmap`
every `lax.switch` branch runs, so an internal avail call would be paid ~15×
per lane. A core applies only its own state change; award reassignment and the
win check are **stage 2**, run once per step by `awards.resolve_step`. The
`*_step` wrappers that can change an award (BuildRoad / BuildSettlement /
BuildCity / BuyDevelopmentCard / PlayKnight) route through `resolve_step_b` so
the standalone API stays fully resolved.

- `longest_road.py` — iterative DFS under one `lax.while_loop`. Perf design:
  each frame packs the used-edge bitset (ranked over the ≤ 15 owned edges) and
  the tip vertex into a single int32; seeding is endpoint-only over owned
  edges plus a forward fallback that covers closed trails; pops happen in
  blocks of `_POP_K` — exact because the DFS is a pure worklist (order
  irrelevant, `best` is a max over popped frames). `STACK_CAP` (445) is proven
  tight in the abstract stack model: `docs/longest-road-stack-bound.html`
  (regenerate with `tools/gen_stack_bound_doc.py`; pre-commit guards
  freshness). **Safety:** JAX silently *drops* out-of-bounds scatter updates,
  so the real overflow guard is the fuzz test
  `test_rules.py::test_dfs_peak_sp_stays_below_dump`; it relies on the rule
  invariant `n_owned <= MAX_ROADS`, which `tests/mechanics/_occupancy.py`
  enforces when generating random occupancies. A lane gated off via `needed`
  seeds empty and adds zero iterations under `vmap`.
- `awards.py` — the stage-2 resolver, factored out of the cores so the DFS
  runs once per step, not once per switch branch. The DFS is vmapped over the
  player axis (one fused loop paying a single max trip count) and gated per
  lane: only a *successful* BuildRoad can extend a road length and only a
  *successful* BuildSettlement can break one (setup placements stay under the
  5-road threshold; edges never disappear; a city keeps the vertex owner),
  tightened by `road_build_gate` (builder owns ≥ 5 roads) and
  `settlement_break_gate` (a single opponent owns ≥ 2 incident edges — one
  edge only makes the vertex a trail *endpoint*). Gated-off lanes keep their
  stored holder/length at zero DFS cost.
- `robber.py` — Discard is **one card per action**, repeated until the owed
  count reaches zero: keeps the choice space flat instead of enumerating
  combinatorial whole-hand splits.
- `setup.py` — the snake order is computed arithmetically because `n_players`
  is per-game state; the host-side `setup_order` restates it plainly for
  tests.
- `trade.py` / `development.py` — placed to break import cycles:
  `port_ratio` would cycle `layout`→`port`, `playable_dev`/`draw_dev_card`
  would cycle `state`→`dev_cards`.
- `common.py` — the shared vocabulary, so rule modules stay leaves importing
  only `board.*` + `common` (no cycle through `action`).
- `action.py` — the `lax.switch` dispatch over the cores. `apply_action`
  takes precomputed `available` and applies branchlessly (candidate always
  computed, then `tree_select`-ed), followed by the stage-2 resolve.
  DomesticTrade is intentionally deferred.
- `flat.py` — the flat action space, the public seam (imports `action.py`,
  never the reverse). The table is fully static. The legality sweep calls each
  core directly over its own static slice of the table — vmapping the switch
  over the table would evaluate all 15 branches per entry. `flat_legality`
  reads one chosen action's bit out of a cached sweep, so the env never
  recomputes avail for the action it applies.

**Testing.** Per-action tests live in `tests/mechanics/actions/`. The trusted
differential oracle is the plain-Python `catan-reference` package, bridged by
`tests/conversion.py`; `tests/test_reference_equivalence.py` drives both
engines with the same action stream and asserts full-state agreement. Gotcha:
seed expecttest inline expectations that hold board renders with a **raw**
literal (`r""""""`) — `EXPECTTEST_ACCEPT=1` preserves the seed's rawness,
while a non-raw seed gets every `\` doubled into unreadable hex art.

## env/

- `batched.py` — the functional `step` / `available` (self-validating for
  arbitrary params: legality computed internally via the switch) and
  `BatchedCatanEnv`, a batched PettingZoo-AEC env (batch axis = parallel
  games; the acting agent per lane is its `current_player`, or the next owing
  player during DISCARD). Design points:
  - **One cached legality source, `self._avail`** (the flat sweep), computed
    at `reset` and refreshed by every `step`; the step gate, `random_actions`,
    and `action_mask` all read it. `_vps` / `_agent_sel` are cached the same
    way — tests that poke `_state` directly must refresh the caches (see
    `flat_available_b` uses in tests).
  - The whole step is **one fused jit dispatch** (`_env_step_core`); small
    batches are dispatch-bound, so collapsing ~5 kernels into 1 is the
    speedup. `rollout(key, n_steps)` replays the `random_actions` + `step`
    driver inside one `lax.scan`, bit-for-bit identical for the same key
    (~2.2x at B=1; `benchmark/test_env_benchmark.py`).
  - Auto-reset is a device-side `lax.cond` on `any(done)` — no per-step
    device→host sync, and board generation is only paid when a lane finished.
    `auto_reset=False` freezes finished lanes instead (used by `aec.py`).
  - The flat table keeps the full 4-player victim domain at every `n_players`
    (rows naming unseated victims are simply never legal), so the flat action
    space is constant across player counts.
  - `Observation` / `Infos` are TypedDicts; `Observation`'s leading `*batch`
    axis fits both the env's batched form and the single-game slice the
    catan-agents policies consume. A test pins `observation_space()` and
    `infos` to their key sets (mypy ties `observe()` to the TypedDict).
  - **Public flat-action seam** consumed by catan-agents: `N_FLAT`,
    `flat_to_action`, `BatchedCatanEnv.flat_mask()`, and `flat_available(board)`
    (the pure-function sweep MCTS uses for in-tree masks).
  - Optional belief tracking (`track_beliefs=True`): a batched `BeliefState`
    advanced once per step by diffing the pre/post states; auto-reset lanes
    restart from the empty-board belief, and a frozen lane's INVALID steps are
    belief no-ops (zero diff).
- `aec.py` — single-game PettingZoo wrapper over `BatchedCatanEnv(batch_size=1,
  auto_reset=False)`. Flat `Discrete` action space; legality exposed as
  `observation["action_mask"]`. Needs the optional `rl` extra;
  `tests/env/test_aec.py` includes PettingZoo's `api_test`.

## Top level

- `belief.py` — card counting: per-observer proven bounds on hidden hands,
  derivable entirely from public information (handing it to an agent never
  leaks). Information model: a robber steal's card *type* is hidden from third
  parties (thief and victim see it) and held dev-card identities are hidden;
  everything else — production, costs, discards, Monopoly surrenders,
  hand/dev counts, the bank — is public, so per-type resource totals across
  hands stay public too. `update_belief` is diff-based; every tightening rule
  is individually sound, and INVALID transitions are no-ops by construction
  (zero diff). **Derived theorem, tested:** with 2 players the bounds stay
  exact (`lo == hi`), recovering "2p is perfect-info up to dev identities"
  from the tracker rather than assuming it. `BeliefView` is deliberately *not*
  a `BoardState`: hidden state is unrepresentable, not placeholdered, and the
  only road back to a playable position is catan-agents' `sample_world`.
  `own_bought` is zeroed off-turn — sampled opponents may therefore play a
  just-bought card one turn early (documented approximation). The
  public/hidden classification is total and forced:
  `test_every_board_field_is_classified` asserts `PublicState` ∪ hidden ==
  `BoardState._fields`, and `sample_world` rebuilds a `BoardState` by explicit
  keyword, so a new `BoardState` field breaks loudly until classified.
- `record.py` — serialisable game records. A game is fully determined by
  `(seed, n_players, number_placement)` plus the flat action trace (all engine
  randomness derives from the seed), so that is exactly what `GameRecord`
  stores; the human-readable JSON move annotations are derived on save and
  **ignored on load** (`flat` is authoritative). `replay` re-steps the record
  and raises on any divergence — doubling as a regression check that engine
  semantics haven't drifted under recorded games. `record_game` drives
  `BatchedCatanEnv` directly (not `CatanAECEnv`) so the `rl` extra isn't
  needed.
