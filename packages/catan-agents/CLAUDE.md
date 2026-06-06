# catan-agents — internal notes

Pure-JAX Catan policies over `catan-engine`'s public flat-action seam
(`catan_engine.env.N_FLAT` / `flat_to_action` / `BatchedCatanEnv.flat_mask`).

- `policy.py` — the `Policy` protocol: single-game `(key, obs, mask) -> flat
  action` (callers `vmap` for batches), plus the `FlatMask` / `FlatAction`
  jaxtyping aliases. Policies are masked-argmax style: with no legal move the
  index is arbitrary and the engine rejects it as `INVALID` (the lane stalls
  until auto-reset), matching `BatchedCatanEnv.random_actions`.
- `baselines.py` — `random_policy`: uniform noise over `N_FLAT`, masked argmax
  (the same trick as the engine's `_random_action_single`).
- `greedy.py` — `greedy_policy`: a static `(N_FLAT,)` base score from an
  action-type priority table (`_TIER`), plus an observation-dependent bonus per
  row group (settlement/city/setup-settlement: adjacent-tile pips via `TILE_V`;
  robber/knight: target-tile pips + 1 for a steal; discard: held count), plus
  uniform `[0,1)` tie-break noise. Tier gaps (>= 100) exceed every bonus range
  (pips <= 15, held <= 19), so bonuses only reorder within a tier; types
  sharing a tier are phase-disjoint. The flat table is decoded once at import
  (`flat_to_action(arange(N_FLAT))`). Deliberately simple: no resource
  targeting, never trades (`MARITIME_TRADE` scores below `END_TURN`), ignores
  whose production the robber blocks.
- `evaluate.py` — Python-loop driver over `BatchedCatanEnv` (sparse reward,
  auto-reset): every seat's vmapped policy picks a move in every lane each step
  and the acting seat's pick is kept (`picks[agent_selection, lanes]`) — n_seats
  policy evals per step, fine for <= 4 seats. Wins accumulate from the sparse
  terminal rewards (exactly one +1 per completed game, so
  `episodes = wins.sum()`). Not a fused rollout; a `lax.scan` version is the
  obvious next step if evaluation throughput starts to matter.

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every shipped policy via the `POLICIES` dict — legality (a pick must be legal
whenever the lane has a legal move, checked through 150 self-play steps),
seeding reproducibility (same seed -> identical self-play action trajectory),
and short self-play rollouts that must complete games. No per-policy
internal-logic tests; a new policy just registers in `POLICIES`.
`tests/conftest.py` installs the jaxtyping/beartype import hook for all
`catan_agents` modules, same pattern as catan-engine.
