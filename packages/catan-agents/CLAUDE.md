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

- `policy.py` — the seat protocols. Policies are masked-argmax style: with no
  legal move the returned index is arbitrary and the engine rejects it as
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
- `mcts.py` — `mctx.gumbel_muzero_policy` with the engine as `recurrent_fn`.
  Structured as a single-tree core under a wrapper that owns all batching
  (`vmap` over trees, then over games; the mctx batch dim stays 1 inside),
  so after `jit` the search runs (games × trees)-wide. Two ensemble-width
  knobs, trees = `num_worlds * num_futures`: `num_worlds` distinct
  `sample_world` draws (belief width) × `num_futures` chance re-keyings per
  draw (chance width — same hidden state, `state._replace(key=...)`, so only
  in-tree dice/steals/draws differ); `action_weights` averaged over all
  trees. Width is near-free vs `num_simulations` (a sequential scan): on an
  RTX 5090 at B=32, 16 trees cost +66% wall-clock while 8× sims cost ~8×.
  Evidence (2p, 200-game seat-swapped): worlds=4 beats worlds=1 head-to-head
  54% but is unchanged vs lookahead — expected, since at 2p `sample_world`
  only varies dev-card identities (belief width is degenerate; any 2p
  ensemble effect is chance width); belief width's payoff should be 3–4p
  (unmeasured, no multi-seat strength protocol yet). The **root prior is the one-step value
  sweep**: with a uniform prior the 16 Gumbel candidates were a random subset
  of 560 and mcts lost to lookahead 6%; the informed prior took it to 37% vs
  lookahead and 86% vs greedy. In-tree child priors stay uniform-over-legal
  (a per-expansion sweep would cost 560×). Frame convention: priors/values
  belong to the node's player-to-move; `discount` is −1 when the mover
  switches, +1 when the same player continues, 0 into terminals (absorbing);
  leaf values are `tanh(value/value_scale)` so the heuristic's scale is
  commensurate with the ±1 terminal reward. Exact zero-sum at 2 players; at
  3–4 the sign-flip discount is the *paranoid* reduction (scalar backups
  can't express max^n). **Known limitation:** search currently *subtracts*
  value relative to its own root prior (flat across sims/candidates/scale) —
  each in-tree child commits a single sampled dice/steal/draw outcome, so
  deeper search plans against fixed chance samples; fixing it needs
  chance-node / afterstate handling, not more simulations. Root-level chance
  width doesn't rescue it either (w=1/f=16/s=32 measured 40% vs lookahead,
  n=20): averaging trees removes variance, but each tree's *in-tree*
  selection stays fused to its own samples. Next probe: a 2-ply expectimax
  root sweep (top-K candidates × the 11 weighted rolls).

## cli.py

`compare` is a seat-swapped head-to-head: two `n_episodes` evaluate runs
(`seed`, `seed + 1`) with the agents' seats exchanged. Tournaments etc. slot
in as new subparsers.

## Registry and tests

`__init__.py` exports the `POLICIES` registry — the single list of shipped
agents, consumed by both the protocol tests and catan-render's bot seam
(which dispatches on `AgentSpec.observes` and filters seat counts).

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every agent in `POLICIES`: legality through self-play, seed reproducibility,
and short rollouts that must complete games. No per-agent internal-logic
tests — a new agent just registers in `POLICIES`. `sample_world` is
infrastructure, not a policy, so it gets unit tests (`tests/test_sample.py`).
`tests/conftest.py` installs the jaxtyping/beartype hook for all
`catan_agents` modules.
