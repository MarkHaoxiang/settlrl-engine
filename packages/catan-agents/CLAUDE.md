# catan-agents — internal notes

Catan agents over `catan-engine`'s public flat-action seam: pure-JAX
policies (`shared/`, `search/`) plus stateful plain-Python planners
(`planner/`).

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
  subclass is the tag (`ObservationSpec` / `BeliefSpec` / `StatefulSpec`), so
  consumers dispatch with `isinstance` and `spec.policy` is precisely typed —
  no casts. A `StatefulSpec`'s `policy` is a *factory* (`seed -> GameAgent`):
  the agent object holds per-game state, so drivers build one per (game,
  seat) and replace it when the lane auto-resets. `GameAgent.act` takes
  *host* data (`HostObservation` / `HostFlatMask`, numpy): handing device
  arrays to host-side logic cost ~10 ms per decision in thirty per-field
  syncs; one `jax.device_get` of the batched observation costs ~0.1 ms/lane.
  The generic cannot type `defaults` itself: `make(**mapping)` is uncheckable
  (ParamSpec doesn't apply to dynamic unpacking). `PolicyPrior` is the
  learned-policy-head seam: `make_mcts` / `make_smcts` take one in place of
  their built-in priors (root sweep + tier table), applying legality
  masking themselves. Policies are masked-argmax style: with no legal
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
  term measured negative (47.3%) and stays at `w_port=0`. June 12: the
  production-scarcity hand term (`w_scarce`, a sqrt-hand copy weighted by
  1/(1+production)) prices cards the player cannot produce, so port/domestic
  conversions toward them read as gains — 57.3% (n=205) over `w_scarce=0` on
  the lookahead rung at the adopted 1.0 (0.5 measured 52.2%; 1.5 lost to 1.0
  head-to-head 43.1%). Value-as-win-prob
  calibration (183k self-play positions): P(win) = σ(0.053·v), phase-stable —
  but the calibrated `value_scale≈38` *lost* to the sharper hand-picked 20 in
  mcts (44.5%, n=200): honest calibration is not the best search temperature.
- `greedy.py` — scripted policy: a static per-row tier score plus small
  observation bonuses. Invariant: tier gaps (≥ 100) exceed every bonus range
  (|bonus| < 50), so bonuses only reorder within a tier; types sharing a tier
  are phase-disjoint, dominated, or deliberately bonus-decided. Two sanctioned
  exceptions, both trade-shaped (June 12, new greedy beat the old 79.0%
  n=305 2p, 67.5% vs chance 33% at 3p; vs random 85%→98.7%): a *productive*
  MARITIME row carries a +150 gate lifting it over END_TURN — productive
  means the bought card is needed for the target build (city if a settlement
  stands, else settlement if a spot is buildable now, else dev card) and the
  sale comes entirely out of surplus, so conversions can't ping-pong; and
  ACCEPT vs REJECT is bonus-decided — accept iff paid purely from surplus
  and (a need advances or it consolidates toward scarcity). The discard
  prefers surplus before most-held. Still deliberately simple: never offers
  a trade (an obs-only policy has no rejected-offer memory), ignores whose
  production the robber blocks. `_BASE` is also mcts's root-prior tier table
  — the maritime gate lives in the bonus channel, so priors are unchanged.
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
  A `StatefulSpec` seat switches `evaluate` to `_evaluate_stepwise`: the same
  seating/budget semantics through a per-step Python loop (stateful seats act
  lane by lane on host-fetched observations; pure seats keep their `_picker`,
  but jitted — eagerly the vmapped greedy alone is 46 ms/step vs 0.1 jitted).
  ~18 ms/step at B=16 on GPU, dominated by the per-step env dispatch; win
  counts sync every step, so the `n_episodes` overshoot is at most a batch.

## search/

All search agents determinize at the root: `sample_world`, then search in
the sample (PIMC, not ISMCTS — the simulated opponent shares the sampled
world; lookahead uses one draw, mcts/smcts an ensemble of `num_worlds`).
Residual approximations in lookahead/mcts: a sampled in-tree draw's identity
is visible one ply ahead (committed per node, not a chance node), and the
in-tree opponent sees the sampled world (strategy fusion) — count-only value
terms blunt what it can exploit. smcts removes the first for dice and dev
draws (true chance nodes); the second is inherent to PIMC.

- `greedy.py` — one-step lookahead: all 662 successors in one
  `vmap(apply_action)`, valued and masked-argmaxed. Trade proposals are the
  one material-neutral successor, so they're scored by their *accepted*
  outcome instead, gated on a partner model (the same value from the
  partner's seat must prefer accepting) minus `trade_penalty` (default 0.25
  — the quality bar that keeps marginal offers below not trading).
  `propose_rate` (default 0.5) randomly withholds proposing each move: the
  engine keeps no memory of rejected offers, so a deterministic proposer
  facing a mispredicted partner (e.g. one playing a different family) would
  re-offer the same trade forever; the gate bounds such streaks
  geometrically. Responding needs none of this — accept vs reject is an
  ordinary value comparison. Measured 3p vs the
  `propose_rate=0` member (seat-rotated, chance 33.3%): 38.7% (n=186) and
  36.9% (n=187) across two seed batches — 37.8% pooled (n=373), a modest but
  consistent edge for offering.
- `mcts.py` — `mctx.gumbel_muzero_policy` with the engine as `recurrent_fn`;
  trade proposals are excluded from both priors (`_NO_PROPOSE`): under the
  paranoid frame the in-tree responder prices every offer as
  rejected-or-harmful, so their ~100 near-tied rows only flood the candidate
  pool — mcts answers trades through search (accept vs reject backs up like
  any move) but never offers one.
  After `jit` the search runs (games × trees)-wide, trees = `num_worlds`
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

## planner/

The stateful decision-tree class: per-game plain-Python agents whose
*strategy* is code (plans, saving, award races, trade memory) and whose
*tactics* consult a one-step lookahead (hybrid sanctioned June 12; it was
value-free before). `pov.py` is the host-side toolkit — one `Pov` per
decision wrapping the host-fetched observation, the static board graph
re-stated as numpy/python tables (`VERTEX_*`, `EDGE_ENDPOINTS`,
`TILE_CORNERS`), and the flat table's host decode (`flat_row`,
`ROWS_OF_TYPE`). `tree.py` is the framework (`Node` / `Selector` / `Plan` /
`Blackboard`); `tactic.py` the lookahead seam; `agent.py` the shipped
`planner` family.

`tactic.py`: `reconstruct` rebuilds a single-game engine board from the
observation — public fields exactly; hidden ones neutrally (opponent hands
spread evenly, their dev cards as knights since only the count enters the
opponent strength term, the dev deck scaled to the unseen remainder,
`free_roads` inferred from an unaffordable-but-legal BUILD_ROAD). The env's
own mask is passed through as the availability `apply_action` trusts, so
reconstruction gaps can't make an illegal action look applied. `Tactic`
caches one 662-wide successor-value sweep per decision (`values`), and
`combo_best` runs the own-turn second ply: apply an enabler, re-sweep
legality (`flat_available_for`), and value the follow-ups — the structural
edge over `lookahead`, which is one-ply and provably weak exactly there
(the June 11 END_TURN→BUY_DEV/TRADE flip diagnosis). `best_paranoid` is the
opponent reply: my candidates within 1.0 of the top are tie-broken by the
opponents' best answer — apply my row, fabricate their MAIN turn on the
result, max over their *grounded* options (builds/bank trades; their dev
plays are masked out because the reconstructed dev hand is fiction).
Measured neutral on the lookahead rung; kept as the principled tie-break
(robber/knight targets, surplus builds). Expectimax over the engine's two
chance seams: `roll_expectation` (exact 11-outcome forced rolls — decides
the pre-roll knight: play iff the relocation raises the expectation of our
own pending roll by > 0.3) and a deck-weighted exact dev-draw expectation
replacing the sweep's single-sample BUY_DEV value. Both measured neutral —
the same lesson as smcts: with the shared stationary heuristic leaf, exact
chance handling doesn't convert into strength; the leaf is the ceiling.

Design invariants:

- **Legality only ever comes from the mask.** Leaves pick among legal rows
  (or build a row with `flat_row` and check it); no rule is re-implemented,
  so engine rule changes can't silently desync the agent. If the whole tree
  declines, a fixed-priority fallback picks some legal row (PROPOSE_TRADE
  deliberately absent there — an unmanaged offer could re-propose forever).
- **Plan steps are declarative targets, not queued actions.** Every tick the
  plan reports its first step missing from the board and the agent re-checks
  the rest; a step gone impossible (spot taken, path edge claimed) triggers a
  replan. That re-validation is what makes state safe across opponents'
  moves *and* auto-reset (a stale plan against a fresh board just invalidates).
- **Memory covers what the engine forgets.** The engine keeps no record of
  rejected trade offers, so the blackboard does: a proposal is remembered
  with the hand that made it, marked rejected if the next MAIN tick shows no
  trace of it, and never re-offered that turn (`max_proposals_per_turn` caps
  the rest). This is the stall-guard the completing-games test exercises.
- **Goals are scored in economic time, switched on clear margins.** A goal's
  score is quality-weighted pips (`_RES_WEIGHT`: wheat/ore premium) minus
  0.35 × the bottleneck rounds production needs to afford it, plus
  closing-urgency near 10 VP (a goal that wins outright dominates at
  25/builds). Settlement goals carry the spot-race model (+3.4 vs lookahead,
  the largest opponent-integration win): an opponent road already touching
  the spot beats any multi-road path of ours (−2·len−1), one edge away is a
  reason to hurry (+0.5), and each path edge adjacent to their network risks
  being cut (−0.4).
  Candidates include the two award races: a Longest Road grab when within two
  trail extensions of taking the card (`my_longest_trail` DFS), and dev buys
  boosted while Largest Army is live. A healthy plan is re-scored against
  fresh rivals each turn and replaced only on a > 2.5 margin (≫ the 0.3
  tie-break noise, so no thrash); `_PLAN_PATIENCE` (8) remains only as the
  starvation backstop for invisibly-starved goals (no dev-deck size or
  `free_roads` in the observation).
- Turn shape after the plan step: `OpportunisticBuild` (any build/dev buy
  fundable from pure surplus, argmaxed by successor value against END_TURN's
  as the do-nothing baseline), `EnablerCombo` (a maritime trade or YOP whose
  *pair* value with an immediate build beats ending the turn; the follow-up
  is committed via `Blackboard.forced_row` and played next tick),
  `Acquire` (YOP/monopoly toward need — monopoly also fires on a ≥ 4-card
  mass grab — maritime from surplus, capped 1:1 proposals), `SpendDown`
  (ending a turn above seven cards banks the excess in a dev card,
  reservations or not), `DenialKnight` (value-timed end-of-turn knight).
  Setup spots, robber/knight targets, discards, and trade responses all pick
  by successor value (setup keeps scripted new-resource/expansion bonuses the
  one-ply value can't see).
- Gotchas: A dev-buy step's "realized"
  check is a baseline comparison on the public dev count (the hand count
  drops again when cards are played, but the plan completes on the next tick,
  before that can happen). The jaxtyping hook enforces `pov.py`/`agent.py`
  annotations at test time — `sum()` of an empty generator is `int`, not
  `float`.

Strength (seat-swapped, June 12 night, hybrid + opponent model): 85.0% vs
greedy (n=200), **~55% vs lookahead** (54.7% and 55.7% on n=300 runs) and
**49.6% vs mcts pooled** (546/1100 over the final configs) — tied with mcts
at the top, clearly over lookahead; from 42% / 10% / — that morning. The
arc: scripted push 75/45/40 (SpendDown, goal-switching margin,
time-to-afford), tactic hybrid to 52/50 (value-arbitrated
`OpportunisticBuild` +4.5, own-turn combos +2-3, leaf-pick delegation
neutral), opponent integration to 55 vs lookahead (spot-race model +3.4;
best-reply tie-break neutral). Gate planner tweaks at n ≥ 200 — n=100
probes swung ±6 points around the n=300 truth. The one measured trap:
letting plain maritime sales dip into plan-reserved cards (47% → 29% vs
lookahead, reverted; combos may, because the pair value prices the whole
exchange). Move latency: the tactic sweep adds ~1 jit dispatch per decision
on top of the ~0.1 ms/lane scripted tick; matches run ~1.5-2x slower than
the pure-scripted planner. Next levers for outright #1: value-informed plan
choice (multi-apply goal valuation), expectimax over the opponent's roll,
or catan-learn's Stage 1 value dropped into tactic.py.

## cli.py

`compare` is a seat-swapped head-to-head: two `n_episodes` evaluate runs
(`seed`, `seed + 1`) with the agents' seats exchanged. `bench` is the
experiment harness: agents are `build_spec` strings (a registry name, or
JSON `{"kind", "params", "value"}` building a configured family member /
reweighted heuristic), seat-swapped at 2 players and seat-rotated at 3-4
(one evaluate per seating, `seed + position`); output includes the binomial
SE and the per-seating split. Tournaments etc. slot in as new subparsers.

## Registry and tests

`__init__.py` exports the `POLICIES` registry — the single list of shipped
agents, consumed by both the protocol tests and catan-render's bot seam
(which dispatches on the spec's class and filters seat counts).

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every agent in `POLICIES` at its `for_testing` parameters: legality through
self-play, seed reproducibility, and episode-budgeted rollouts that must
complete games (`_self_play` drives stateful specs lane by lane, mirroring
`_evaluate_stepwise`). No per-agent internal-logic
tests — a new agent just registers in `POLICIES`. catan-render's bot seam
skips `StatefulSpec` families (its `bot_act` is per-move and stateless; a
stateful seat there needs a per-session agent cache that doesn't exist yet). `sample_world` is
infrastructure, not a policy, so it gets unit tests (`tests/test_sample.py`).
`tests/conftest.py` installs the jaxtyping/beartype hook for all
`catan_agents` modules.

`tests/benchmark/` holds the pytest-benchmark suite (move latency,
`sample_world`, the fused self-play window), parametrized over `POLICIES` at
*shipped* defaults — it measures what ships, unlike the tests' cheap
`for_testing` members. `benchmark`-marked and deselected from the default
run; the repo-root `run_benchmarks.sh` re-selects it (engine + agents). The
self-play window reuses `evaluate`'s private `_picker`/`_actor` so one actor
identity persists across rounds — calling `evaluate()` per round would
retrace its fresh closure every time.
