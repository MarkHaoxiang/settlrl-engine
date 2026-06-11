# catan-agents — internal notes

Pure-JAX Catan agents over `catan-engine`'s public flat-action seam.

**No agent assumes full observability.** Model-based agents consume the
engine's honest `BeliefView` (see the engine's `belief.py` notes); hidden
state is unrepresentable there, so the only road back to a playable position
is `sample_world`. There is no 2p/4p module split: with two players the
tracked belief is exact on resources (tested in the engine), so "2p is
perfect-info" is a property of the data, not an API boundary — the same
agents run at 2–4 players with beliefs of varying sharpness.

## shared/

- `policy.py` — the seat protocols and `AgentSpec`: a policy *family*
  (`make` + `defaults`, with `policy` the cached shipped build) plus optional
  `for_testing` parameter overrides — `spec.for_tests` is the cheap family
  member the protocol tests run (the tested properties are
  parameter-independent). `AgentSpec` is generic over its protocol and the
  subclass is the tag (`ObservationSpec` / `BeliefSpec`), so consumers
  dispatch with `isinstance` and `spec.policy` is precisely typed — no casts.
  The generic cannot type `defaults` itself: `make(**mapping)` is uncheckable
  (ParamSpec doesn't apply to dynamic unpacking). Policies are
  masked-argmax style: with no legal
  move the returned index is arbitrary and the engine rejects it as
  `INVALID` (the lane stalls until auto-reset), matching
  `BatchedCatanEnv.random_actions`.
- `sample.py` — `sample_world` fills every hidden field with a posterior
  sample. Guaranteed (tested in `tests/test_sample.py`): public fields
  untouched; hand sizes, dev counts, per-type totals, and the observer's own
  rows all match the public record. The resource deal's
  proportional-headroom weighting is a *surrogate* for the exact posterior,
  not the posterior (`hi` is relaxed if jointly infeasible). The deal's
  `while_loop` stops at the owed count rather than the worst-case 95: same
  sampling law (each draw is a fresh key), 13x on `sample_world`, 7x on a
  lookahead move at B=1 (RTX 5090; the sequential chain was launch-bound).
  The closing
  `BoardState(...)` is built by explicit keyword on purpose: a new
  `BoardState` field fails to compile here until classified public or hidden.
- `value.py` — heuristic strength function; value = own strength − best
  opponent's. On a *sampled* world the "hidden" fields it reads are
  belief-consistent samples, so it stays honest. Tuning evidence (2p
  seat-swapped CLI matches, 200–700 games): the expansion + progress terms
  took lookahead-vs-greedy from 34% to 86.5% (without the
  best-buildable-spot term, lookahead never expanded); the hand-diversity
  term is worth ~55% head-to-head over without; `w_spot` 1.0 vs 0.5 and a
  port-count term measured neutral. June 11 overnight sweep: the wheat/ore
  production premium (`w_wheat_ore=0.25`; 0.4 overshoots, 44.5% vs 0.25) plus
  the closing-urgency term (`w_race=0.8`; 1.2 measured equal) beat the prior
  weights 57.3% (n=600) on the lookahead rung and 57.5% (n=200) on the mcts
  rung, no greedy regression (90.5%); a production-matched 2:1-port synergy
  term measured negative (47.3%) and stays at `w_port=0`. Value-as-win-prob
  calibration (183k self-play positions): P(win) = σ(0.053·v), phase-stable —
  but the calibrated `value_scale≈38` *lost* to the sharper hand-picked 20 in
  mcts (44.5%, n=200): honest calibration is not the best search temperature.
- `greedy.py` — scripted policy: a static per-row tier score plus small
  observation bonuses. Invariant: tier gaps (≥ 100) exceed every bonus range
  (pips ≤ 15, held ≤ 19), so bonuses only reorder within a tier; types
  sharing a tier are phase-disjoint. Deliberately simple: no resource
  targeting, never trades, ignores whose production the robber blocks.
- `evaluate.py` — fused driver over the engine's `rollout(actor=...)` seam:
  every seat's vmapped agent picks in every lane each step inside the scan and
  the acting seat's pick is kept — n_seats policy evals per step, fine for
  ≤ 4 seats. Steps run in `_SYNC_WINDOW`-sized scans; the win count syncs only
  between windows, so `n_episodes` may overshoot by up to a window of lanes.
  Measured June 11 (B=32, RTX 5090): 1.7× over the per-step loop on
  lookahead-vs-greedy, ~1.3× steady-state on mcts matches (42 vs ~57 ms/step).
  Caveat: the scan retraces per `evaluate` call (the actor closure is fresh
  each time) — ~12 s per call for mcts-sized bodies, amortised over 200-game
  matches, noticeable on ≤ 20-game probes.

## search/

All search agents determinize at the root: `sample_world`, then search in
the sample (PIMC, not ISMCTS — the simulated opponent shares the sampled
world; lookahead uses one draw, mcts/smcts an ensemble of `num_worlds`).
Residual approximations in lookahead/mcts: a sampled in-tree draw's identity
is visible one ply ahead (committed per node, not a chance node), and the
in-tree opponent sees the sampled world (strategy fusion) — count-only value
terms blunt what it can exploit. smcts removes the first for dice and dev
draws (true chance nodes); the second is inherent to PIMC.

- `greedy.py` — one-step lookahead: all 560 successors in one
  `vmap(apply_action)`, valued and masked-argmaxed.
- `mcts.py` — `mctx.gumbel_muzero_policy` with the engine as `recurrent_fn`;
  after `jit` the search runs (games × trees)-wide, trees = `num_worlds`
  (belief width) × `num_futures` (chance width: re-keyed replicas of one
  draw). Width is near-free vs `num_simulations` (a sequential scan): 16
  trees cost +66% wall-clock, 8× sims cost ~8× (RTX 5090, B=32). Frames are
  two-sided (searcher vs the table): every node holds the searcher's value
  signed into the mover's side and the discount flips only across the side
  boundary — the true *paranoid* reduction (scalar backups can't express
  max^n). At 2p this is provably identical to flipping on every mover change
  (632/640 same picks, 49.5% n=200 self-match); at 4p the every-mover-flip
  rule negates the searcher's own next turn ((-1)^3 per round) and measured
  *below chance* vs 3× lookahead (20%, n=80) — the side frame took the same
  seeds to 32.3% (n=161, chance 25%; 36.1% pooled n=241) and reads 62.2% vs
  2× lookahead at 3p (n=90, chance 33%).
  Deviations from mctx defaults, each fixing a measured ply-2 bias: the
  root prior is the one-step value sweep (uniform priors made Gumbel's 16
  candidates a random subset of 560 — 6% vs lookahead); interior priors are
  greedy's tempered tier table (uniform + mctx's deterministic interior
  argmax made every first expansion the lowest-index legal action);
  `ROLL_DICE` children back up the 11-roll expectation, not their one
  sampled outcome; `rescale_values=False` (the min-max rescale amplified
  any Q ranking to ~8 nats no matter how noisy — why `value_scale` once
  measured flat). History of the month-long "search subtracts value" bug
  (34–43% vs lookahead, flat across sims / candidates / scale / root
  ensembling): at 32 sims the trees are ~2 plies; decision-level
  decomposition (2.5k positions, picks vs the prior argmax priced by the
  sweep) showed depth-1 selection near-transparent (2% flips) while full
  depth flipped ~9% of decisions, 92% losing 1-ply value, concentrated
  END_TURN → BUY_DEV/TRADE — turn-keeping actions back up a max over noisy
  follow-ups (optimizer's curse), END_TURN a sign flip plus one sampled
  opponent roll. The fixes above took flips 12% → 7% and mcts vs lookahead
  37% → **57%** (114–86, n=200). Ensemble evidence (2p): worlds=4 beats
  worlds=1 head-to-head 54% but didn't move the lookahead number — at 2p
  `sample_world` only varies dev-card identities, so belief width is
  degenerate there; its payoff should be 3–4p (no multi-seat protocol yet).
  Tuning gotchas: diagnose decision-level rather than by ~20-game matches
  (SE ±11%), and at absolute Q scale a large σ flip usually means an
  in-tree terminal that the 1-ply regret referee misprices as a loss.
  June 11 parameter sweep (each vs defaults, n=200+): the defaults are the
  local optimum — sims 64 *loses* (44.5%; depth still can't pay through
  chance fusion, 16 ≈ 32), considered peaks at 16 (8: 42.5%, 32: 47%),
  prior_scale 5 loses (41%), value_scale 12/38 tie or lose to 20. Width (16
  trees) wins self-play 54.5% (n=400) but doesn't widen the lookahead gap at
  2p and is even at 4p (36.3% vs 36.1% pooled, n≈240/side) — so it stays at
  4×1. Depth's unlock would be explicit dice chance nodes;
  `mctx.stochastic_muzero_policy` is PUCT-based (no Gumbel/absolute-Q),
  an architecture change, not a knob. 4p evals: seed-batch variance at n=80
  is huge (30.9% vs 43.8% same config) — matched seeds or n ≥ 240.
  Perf (RTX 5090, B=1, 2p): a shipped move is ~15 ms ≈ 0.70 ms × sims; the
  per-sim cost is mctx's descent/backup over the (nodes × 560) stats tables,
  *not* the embedding or our recurrent_fn (engine step + leaf + roll-EV
  measure 0.22 ms fused). Packing the embedding (`_codec`: BoardState → one
  uint8 row + key, layout in the closure; bit-identical search, round-trip
  pinned by `test_mcts_codec.py`) only cut the per-search fixed cost — tree
  storage init — so it pays at small budgets (sims=8: −32%) and ~4% shipped.
  Further wall-clock wants a narrower in-tree action axis (mctx surgery) or
  more lanes per dispatch (B=64: 0.97 ms/move-lane).
- `smcts.py` — experimental, deliberately **not** in `POLICIES`:
  stochastic-MuZero search (PUCT) with dice and dev draws as true chance
  nodes over the engine's forced-outcome seams (`ROLL_DICE idx=2..12`,
  `BUY_DEV idx=1..5`); the two-sided frame shared with mcts. June 11
  verdict (2p): tuned — `prior_scale=10` (2 degenerates PUCT, 45.6%),
  `qtransform_by_parent_and_siblings` (PUCT wants normalized Q, the
  *opposite* of Gumbel's absolute-Q fix) — it ties mcts: 56.7% vs lookahead
  (pooled n=319) and 49.3% head-to-head (n=215), at ~2× wall-clock (a game
  ply is two tree edges). The motivating hypothesis was **falsified**:
  depth still doesn't pay with chance handled exactly (64→128 sims:
  53.3%→49.5%), and the dev-draw chance node (which also removes the
  one-ply draw peek) is strength-neutral (50.5% A/B, n=210) — the binding
  constraint is the stationary heuristic leaf plus the optimizer's curse
  over decision layers, not chance fusion. Deep lines through k rolls span
  11^k outcomes, so unbiased depth is also variance-starved at 10²-sim
  budgets. Becomes interesting only with a learned value function whose
  error shrinks under search; kept as the working scaffold for that.

## cli.py

`compare` is a seat-swapped head-to-head: two `n_episodes` evaluate runs
(`seed`, `seed + 1`) with the agents' seats exchanged. Tournaments etc. slot
in as new subparsers.

## Registry and tests

`__init__.py` exports the `POLICIES` registry — the single list of shipped
agents, consumed by both the protocol tests and catan-render's bot seam
(which dispatches on the spec's class and filters seat counts).

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every agent in `POLICIES` at its `for_testing` parameters: legality through
self-play, seed reproducibility, and episode-budgeted rollouts that must
complete games. No per-agent internal-logic
tests — a new agent just registers in `POLICIES`. `sample_world` is
infrastructure, not a policy, so it gets unit tests (`tests/test_sample.py`).
`tests/conftest.py` installs the jaxtyping/beartype hook for all
`catan_agents` modules.
