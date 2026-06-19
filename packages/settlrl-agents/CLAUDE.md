# settlrl-agents — internal notes

Settlrl agents over `settlrl-engine`'s public flat-action seam: pure-JAX
policies plus stateful plain-Python planners
(`planner/`).

**No agent assumes full observability.** Model-based agents consume the
engine's honest `BeliefView` (see the engine's `belief.py` notes); hidden
state is unrepresentable there, so the only road back to a playable position
is `sample_world`. There is no 2p/4p module split: with two players the
tracked belief is exact on resources (tested in the engine), so "2p is
perfect-info" is a property of the data, not an API boundary — the same
agents run at 2–4 players with beliefs of varying sharpness.

## Layout

The API layer is the top-level modules — `policy.py` (protocols/specs),
`value.py` (the value protocol and the heuristic's *weights*), `evaluate.py`,
`sample.py`, the registry in `__init__.py`, `cli.py` — plus the agents
(`baselines.py`, `greedy.py`, `search/`, `planner/`). `internal/` holds the
helpers behind them: `rows.py` (the flat-table decode) and
`feature_engineering.py` (the weight-free hand-engineered features —
`board_features` for the value terms, `target_build` / `maritime_ratio` for
greedy's trade sense). Weights always live with an agent or in `value.py`;
features never carry them.

The lab harness (`Run`/`start_run` bookkeeping + the pydantic/OmegaConf `Config`
base) moved to `settlrl_learn.experiment` — it is a training-side concern, and
relocating it keeps `settlrl-agents` (the play/serve library) free of
`pydantic`/`omegaconf`. Experiments import it from there now.

`service/` is the **one-bot SDK + service**. A service hosts a single `Bot`
(`sdk.py`: subclass it, implement `act(view) -> MoveModel`; the framework tracks
each game in flight and hands the bot a `GameView` from the acting seat). `app.py`
(`create_app(bot)`) serves it over the shared wire protocol
(`settlrl_game.botproto`): `GET /info` (the bot's `BotInfo`) and `POST /act`,
which applies the moves the bot has not seen yet — the incremental, structured
`MoveModel` tail after `base`, with a `409 {resync, have}` handshake when its
tracked game is behind — then returns the chosen move. `bots.py` is the four
bundled engine bots (`EngineBot` over a `POLICIES` policy): it bridges the
tracked reference position to an engine board (`bridge.py`) and translates the
chosen engine flat back to a `MoveModel`. The CLI `settlrl-bot-service --bot KIND`
serves one. It's the `settlrl-app` game server's only source of bot moves
(delegated over HTTP), kept here because it *is* the agents running. Behind the
`[service]` optional extra (`fastapi` + `settlrl-game`) and never imported by
`__init__`, so `import settlrl_agents` stays free of `fastapi`
(`sdk.py`/`app.py` themselves are JAX-free; only `bots.py` pulls the engine). The
`bridge.py` dtype contract is exact: engine
`BoardLayout`/`BoardState` arrays are `uint8` (jaxtyping-enforced under the test
hook — render's conftest never checked engine types, so an int32 slipped by).

## API layer and agents

- `rows.py` — the flat action table decoded once (device `ROW_TYPE` /
  `ROW_PARAMS` for the vmapped sweeps; host `ROW_IDX` / `ROW_TARGET` /
  `ROWS_OF_TYPE` / `flat_row` for the planner). Every agent imports from
  here — there used to be one decode per module, and they can silently
  diverge.
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
  learned-policy-head seam: `make_search` takes one in place of its built-in
  priors (root sweep + tier table), applying legality masking itself. Policies are masked-argmax style: with no legal
  move the returned index is arbitrary and the engine rejects it as
  `INVALID` (the lane stalls until auto-reset), matching
  `BatchedSettlrlEnv.random_actions`.
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
  head-to-head 43.1%). June 12 night feature round (gated
  lookahead-vs-lookahead ablations, n=200 probes → n=400 confirms): adopted
  `w_numbers=0.3` (distinct dice numbers collected on — income smoothness)
  + `w_kheld=0.8` (held knights toward the army race), 56.1% ± 2.4% (n=415)
  over without; the full triple with `w_spots=0.6` measured *worse* (53.2%
  — spots overlaps the best-spot term) and a summed-completeness `w_fill`
  was outright negative (43.1% at 1.0: it rewards hoarding toward several
  builds, i.e. discard exposure) — both knobs kept at 0. Ladder re-checked
  on the new defaults: relative rungs unchanged (every top agent shares the
  leaf), absolute level up. `make_linear(weights)` deploys any named-
  coefficient fit over `BoardFeatures` as a ValueFunction — the classical-
  fit seam (experiments/0002: outcome-fit logistic reaches hand-tuned level
  vs greedy but loses head-to-head 42.7%; held-out AUC is flat across
  candidates whose match probes span 53–78%, so candidate selection must be
  by match, never by fit metric; `second_spot` / `reach` / `army_lead`
  joined the features at weight 0). Value-as-win-prob
  calibration (183k self-play positions): P(win) = σ(0.053·v), phase-stable —
  but the calibrated `value_scale≈38` *lost* to the sharper hand-picked 20 in
  mcts (44.5%, n=200): honest calibration is not the best search temperature.
- `greedy.py` — the scripted agent: its tier table and bonus coefficients
  over the obs-side features. A static per-row tier score plus small
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
  production the robber blocks. `TIER_SCORES` is also the search's interior
  prior (`_TIER_LOGITS`) — the maritime gate lives in the bonus channel, so
  priors are unchanged.
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

`make_search` / `make_search_weights` (`search/__init__.py`) are the
re-determinizing **Single-Observer ISMCTS**: a custom fixed-capacity tree
(`search/ismcts.py`, `make_tree`) whose every simulation draws a fresh
`sample_world` determinization and descends *forward* under it, filtering
legality per simulation. `make_search` argmaxes the improved policy;
`make_search_weights` returns the distribution (the AlphaZero policy target —
experiment 0004). It replaced a former `mcts`/`smcts`/`ismcts`/`lookahead`
quartet (2026-06-17) and then the `mctx` engine that backed it (2026-06-19,
commit 742b94b — `make_search` *is* the custom tree now); shared frame/prior/
dice constants live in `search/_common.py`, the trade/lookahead/`num_trees`
wrapper in `__init__.py`, and the tree in `ismcts.py`.

**The tree (`ismcts.py`).** One XLA program over a fixed-capacity `_Arena`
(node/edge arrays sized to `num_simulations + 1`), so the whole search stays on
device and `vmap`s over lanes. Each simulation determinizes once and descends a
`while_loop`, stepping the engine so legality at every node comes from
`flat_available_for` on the live determinized state — **true per-simulation
legality, no no-op edges** (the half mctx's fixed action axis could not express;
Cowling 2012 / the Canopy custom tree). The loop stops at the first unexpanded
edge / terminal, so a simulation pays only its own depth of engine steps.
Selection is **mctx's Gumbel-MuZero, ported onto this tree**: Gumbel + Sequential
Halving at the root (considered-visits schedule replicated in numpy — no `mctx`
dependency), deterministic interior selection tracking `softmax(prior +
completed_Q)`, and the mixed-value completed-Q transform (unvisited →
prior-weighted blend of the node value and its children's Q, scaled
`(maxvisit_init + max_visits)·mix_scale`), all over *this* determinization's
legal set. The caller (`make_search`) assembles the **root prior logits** — the
raw one-step value sweep, a learned `prior`, or trade-scored proposals — and
passes them in; the interior prior is the learned `prior` if given, else greedy's
tier table. `action_weights = softmax(root_logits + completed_Q)` is the policy
target; the move is its masked argmax. The immediate dice roll is valued by its
exact 11-roll expectation at the leaf (`ROLL_DICE` children); deeper chance is
integrated by the per-simulation resampling. ~5–6 ms/move (B=1 CPU), faster than
the retired mctx search (5.8 vs 7.4 ms); ≤1 node added per simulation so `size`
never overflows. `num_trees` averages independent such trees.

**Strength: parity with the retired mctx search, ~0.55–0.58 vs lookahead** (2p
seat-swapped, n≥220 GPU; head-to-head vs old mctx held 0.49–0.52 across 16/32/64
sims, ±0.03 — not a single-budget fluke). The decisive bug in the first cut of
the port was the root prior: a tanh+tier *compression* flattened the policy to
near-uniform (0.184 vs lookahead); the raw value sweep (`values / prior_scale`,
±20 spread) restored it (commits 928f370 / 742b94b). Contracts in
`tests/test_ismcts.py` (legal move across setup/mid/late + 4p, legal-supported
distribution, reproducibility, concentration above uniform, no-legal fallback,
game completion).

**Why one search (the merges).** The binding constraint is the stationary
heuristic leaf, not search machinery, so prior agents tied at ~parity and the
*principled* one wins on cleanliness — the lever stays the leaf (experiment 0003
/ settlrl-learn), and win rate vs lookahead does **not** climb with sims:
- `smcts`'s explicit dice/dev chance nodes were **falsified as a strength
  lever** (56.7% vs lookahead pooled n=319, 49.3% h2h, ~2× wall-clock; dev
  chance node strength-neutral 50.5% n=210; 64→128 sims 53.3%→49.5%). Roll-EV
  leaves + per-simulation resampling subsume their purpose — dropped, not ported.
- `mcts` (frozen-world) re-determinizes nothing; per-simulation determinization
  is its principled superset (*parity, not a win* at 3p: 0.352 ± 0.031, n=244,
  32 sims; 64 worlds 0.307 — more worlds doesn't help; ~ties at 2p where the
  belief is ~exact). Cost: one `sample_world` (~0.05 ms) + the leaf's own depth
  of engine steps per simulation.
- the **mctx engine** itself: the re-determinizing Gumbel-MuZero ran on mctx's
  fixed action axis (root-only legality, illegal path-actions no-op). The custom
  tree reaches its strength with true per-simulation legality and *faster*, so
  mctx (and its dependency) were retired.

`num_simulations=0` is the **lookahead** special case: no tree, just the masked
argmax of the root one-step value sweep over `num_trees` sampled worlds. It is
also the *only* configuration that offers trades — `propose_rate` > 0 (default
0 for the search, 0.5 for the `lookahead` registry entry) lets the root score
proposals by their *accepted* outcome under a partner model (the same value
from the partner's seat must prefer accepting) minus `trade_penalty` (0.25, the
bar below which an offer loses to not trading); the gate withholds proposing
some moves so a mispredicted-partner proposer can't re-offer forever (the engine
keeps no rejected-offer memory). Offers are root-only: under the paranoid frame
the in-tree responder prices every offer as rejected-or-harmful, so proposals
are excluded from the in-tree prior (`_NO_PROPOSE`) and the search *answers*
trades through search but never offers one. Lookahead offering measured 37.8%
pooled (n=373) vs the `propose_rate=0` member at 3p — a modest, consistent edge.

Frame and tuning evidence (carried over; the search inherits all of it):
- Frames are two-sided (searcher vs the table): every node holds the searcher's
  value signed into the mover's side, the discount flips only across the side
  boundary — the true *paranoid* reduction (scalar backups can't express
  max^n). At 2p provably identical to flipping on every mover change (632/640
  same picks); at 4p the every-mover-flip rule negates the searcher's own next
  turn and measured below chance vs 3× lookahead (20%, n=80) — the side frame
  took the same seeds to 32.3% (n=161, chance 25%) and 62.2% vs 2× lookahead at
  3p (n=90).
- Deviations from a naive Gumbel search, each fixing a measured ply-2 bias: root
  prior = the one-step value sweep (uniform made Gumbel's candidates a random
  subset — 6% vs lookahead); interior prior = greedy's tempered tier table
  (uniform + deterministic interior argmax expanded the lowest-index legal
  action); `ROLL_DICE` 11-roll expectation; the completed-Q transform does **not**
  min-max rescale (it amplified any Q ranking to ~8 nats regardless of noise).
  The month-long
  "search subtracts value" bug (34–43% vs lookahead, flat across sims/
  candidates/scale): at ~32 sims trees are ~2 plies and full-depth selection
  flipped ~9% of decisions, 92% losing 1-ply value, concentrated
  END_TURN → BUY_DEV/TRADE (optimizer's curse over noisy follow-ups). The fixes
  took flips 12% → 7% and the search 37% → **57%** vs lookahead (n=200).
- June 11 sweep: defaults are the local optimum — sims 64 *loses* (44.5%; depth
  can't pay through the leaf), considered peaks at 16, prior_scale 5 loses,
  value_scale 12/38 tie-or-lose to 20. Diagnose decision-level, not by ~20-game
  matches (SE ±11%); 4p evals need matched seeds or n ≥ 240.

## planner/

The stateful decision-tree class: per-game plain-Python agents whose
*strategy* is code (plans, saving, award races, trade memory) and whose
*tactics* consult a one-step lookahead (hybrid sanctioned June 12; it was
value-free before). `pov.py` is the host-side toolkit — one `Pov` per
decision wrapping the host-fetched observation, the static board graph
re-stated as numpy/python tables (`VERTEX_*`, `EDGE_ENDPOINTS`,
`TILE_CORNERS`), and the flat table's host decode (`flat_row`,
`ROWS_OF_TYPE`, re-exported from `internal.rows`). `tree.py` is the framework
(`Node` / `Selector` / `Plan` / `Blackboard`); `goals.py` the goal economics
(`plan_candidates` / `choose_plan` and their scoring weights); `tactic.py`
the lookahead seam; `agent.py` the tree's nodes and the shipped `planner`
family. The tactic's batched seams (`_after_many`, `_best_replies`) take
fixed-size row blocks padded with a repeated legal row — fixed shapes mean
one jit trace instead of one per block size.

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
or settlrl-learn's Stage 1 value dropped into tactic.py.

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
agents, consumed by both the protocol tests and settlrl-app's bot seam
(which dispatches on the spec's class and filters seat counts).

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every agent in `POLICIES` at its `for_testing` parameters: legality through
self-play, seed reproducibility, and episode-budgeted rollouts that must
complete games (`_self_play` drives stateful specs lane by lane, mirroring
`_evaluate_stepwise`). No per-agent internal-logic
tests — a new agent just registers in `POLICIES`. settlrl-app's bot seam
skips `StatefulSpec` families (its `bot_act` is per-move and stateless; a
stateful seat there needs a per-session agent cache that doesn't exist yet). `sample_world` is
infrastructure, not a policy, so it gets unit tests (`tests/test_sample.py`).
`tests/conftest.py` installs the jaxtyping/beartype hook for all
`settlrl_agents` modules.

`tests/benchmark/` holds the pytest-benchmark suite (move latency,
`sample_world`, the fused self-play window), parametrized over `POLICIES` at
*shipped* defaults — it measures what ships, unlike the tests' cheap
`for_testing` members. `benchmark`-marked and deselected from the default
run; the repo-root `run_benchmarks.sh` re-selects it (engine + agents). The
self-play window reuses `evaluate`'s private `_picker`/`_actor` so one actor
identity persists across rounds — calling `evaluate()` per round would
retrace its fresh closure every time.
