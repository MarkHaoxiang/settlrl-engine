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
  parameter-independent). Policies are masked-argmax style: with no legal
  move the returned index is arbitrary and the engine rejects it as
  `INVALID` (the lane stalls until auto-reset), matching
  `BatchedCatanEnv.random_actions`.
- `sample.py` — `sample_world` fills every hidden field with a posterior
  sample. Guaranteed (tested in `tests/test_sample.py`): public fields
  untouched; hand sizes, dev counts, per-type totals, and the observer's own
  rows all match the public record. The resource deal's
  proportional-headroom weighting is a *surrogate* for the exact posterior,
  not the posterior (`hi` is relaxed if jointly infeasible). The closing
  `BoardState(...)` is built by explicit keyword on purpose: a new
  `BoardState` field fails to compile here until classified public or hidden.
- `value.py` — heuristic strength function; value = own strength − best
  opponent's. On a *sampled* world the "hidden" fields it reads are
  belief-consistent samples, so it stays honest. Tuning evidence (2p
  seat-swapped CLI matches, 200–700 games): the expansion + progress terms
  took lookahead-vs-greedy from 34% to 86.5% (without the
  best-buildable-spot term, lookahead never expanded); the hand-diversity
  term is worth ~55% head-to-head over without; `w_spot` 1.0 vs 0.5 and a
  port-count term measured neutral.
- `greedy.py` — scripted policy: a static per-row tier score plus small
  observation bonuses. Invariant: tier gaps (≥ 100) exceed every bonus range
  (pips ≤ 15, held ≤ 19), so bonuses only reorder within a tier; types
  sharing a tier are phase-disjoint. Deliberately simple: no resource
  targeting, never trades, ignores whose production the robber blocks.
- `evaluate.py` — Python-loop driver: every seat's vmapped agent picks in
  every lane each step and the acting seat's pick is kept — n_seats policy
  evals per step, fine for ≤ 4 seats. Budget is exactly one of `n_steps`
  (sync-free) or `n_episodes` (syncs on the win count each step; may
  overshoot when lanes finish together; `_MAX_STEPS_PER_EPISODE` guards
  non-termination). Not a fused rollout; a `lax.scan` version is the obvious
  next step if evaluation throughput starts to matter.

## search/

Both agents determinize at the root: `sample_world`, then search in the
sample (PIMC, not ISMCTS — the simulated opponent shares the sampled world;
lookahead uses one draw, mcts an ensemble of `num_worlds`). Residual
approximations: a sampled in-tree draw's identity is visible one ply ahead
(committed per node, not a chance node), and the in-tree opponent sees the
sampled world (strategy fusion) — count-only value terms blunt what it can
exploit.

- `greedy.py` — one-step lookahead: all 560 successors in one
  `vmap(apply_action)`, valued and masked-argmaxed.
- `mcts.py` — `mctx.gumbel_muzero_policy` with the engine as `recurrent_fn`;
  after `jit` the search runs (games × trees)-wide, trees = `num_worlds`
  (belief width) × `num_futures` (chance width: re-keyed replicas of one
  draw). Width is near-free vs `num_simulations` (a sequential scan): 16
  trees cost +66% wall-clock, 8× sims cost ~8× (RTX 5090, B=32). The
  discount flips the value frame on mover switch — exact zero-sum at 2p,
  the *paranoid* reduction at 3–4 (scalar backups can't express max^n).
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

## cli.py

`compare` is a seat-swapped head-to-head: two `n_episodes` evaluate runs
(`seed`, `seed + 1`) with the agents' seats exchanged. Tournaments etc. slot
in as new subparsers.

## Registry and tests

`__init__.py` exports the `POLICIES` registry — the single list of shipped
agents, consumed by both the protocol tests and catan-render's bot seam
(which dispatches on `AgentSpec.observes` and filters seat counts).

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every agent in `POLICIES` at its `for_testing` parameters: legality through
self-play, seed reproducibility, and episode-budgeted rollouts that must
complete games. No per-agent internal-logic
tests — a new agent just registers in `POLICIES`. `sample_world` is
infrastructure, not a policy, so it gets unit tests (`tests/test_sample.py`).
`tests/conftest.py` installs the jaxtyping/beartype hook for all
`catan_agents` modules.
