# catan-agents — internal notes

Pure-JAX Catan agents over `catan-engine`'s public flat-action seam
(`catan_engine.env.N_FLAT` / `flat_to_action` / `BatchedCatanEnv.flat_mask`,
plus `step` / `available` / `flat_available` and `mechanics.action.apply_action`
for the model-based agents).

**No agent assumes full observability.** Model-based agents consume the
engine's honest seam — `BatchedCatanEnv(track_beliefs=True).belief_view(seat)`,
a *censored* `BoardState` (hidden fields removed by `catan_engine.belief.censor`)
plus a `PlayerBelief` (proven `[lo, hi]` bounds from card counting) — and the
only road back to a playable position is `sample_world`. The old 2p/4p module
split is gone: with two players the tracked belief is exact on resources
(tested in the engine), so "2p is perfect-info" is now a property of the data,
not an API boundary; the same agents run at 2-4 players with beliefs of
varying sharpness.

## shared/

- `policy.py` — the seat protocols: `Policy` (single-game `(key, obs, mask) ->
  flat action`; callers `vmap` for batches) and `BeliefPolicy` (`(key, layout,
  censored_state, belief, player, mask)`), the `FlatMask` / `FlatAction`
  jaxtyping aliases, and `AgentSpec` (policy + `observes` kind
  ("observation" | "belief") + supported seat counts) — the registry value
  type. Policies are masked-argmax style: with no legal move the index is
  arbitrary and the engine rejects it as `INVALID` (the lane stalls until
  auto-reset), matching `BatchedCatanEnv.random_actions`.
- `sample.py` — `sample_world(key, censored_state, belief, player) ->
  BoardState`: fills everything `censor` removed with a posterior sample —
  opponents' dev hands dealt uniformly without replacement from the censored
  deck (= the unseen pool) via a static 25-slot card view
  (`_CARD_TYPE`/`_CARD_RANK`, slots noised once, ranked, owners assigned by
  `searchsorted` over the per-opponent counts), opponents' resources dealt one
  card at a time within `[lo, hi]` to their public hand sizes against the
  public per-type pool (`res_total` − placed; weights = headroom capped by
  pool, `hi` relaxed if jointly infeasible — proportional-headroom is a
  surrogate for the exact posterior, not the posterior), and a fresh PRNG key.
  Guaranteed: hand sizes, dev counts, per-type totals, and the observer's own
  rows all match the public record (`tests/test_sample.py`). The per-card
  `fori_loop` is `_MAX_DEAL` = 95 iterations (the bank bound), trivial at the
  once-per-root call rate.
- `value.py` — `ValueFunction` protocol: single-game `(layout, state, player)
  -> scalar`, higher better, arbitrary scale. `heuristic_value` = own strength
  minus best opponent's; strength = 10·VP (buildings + awards + VP cards — own
  exact, opponents' expected via the deck's 5/25 VP share over their count) +
  pips of own buildings (city 2x, robber tile zeroed) + 0.3·Σ√(resource counts)
  (diversity; makes the cheapest discard the most-held resource) + 1.5·dev
  count. On a *sampled* world the "hidden" fields it reads are belief-consistent
  samples, so it stays honest. Also hosts `tile_pips` / `vertex_pips` (used by
  greedy too).
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
  auto-reset; `track_beliefs` switched on iff some seat observes "belief"):
  every seat's vmapped agent picks a move in every lane each step and the
  acting seat's pick is kept (`picks[agent_selection, lanes]`) — n_seats
  policy evals per step, fine for <= 4 seats. `_seat` adapts both kinds (obs
  seats get `env.observe(i)`, belief seats `env.board[0]` +
  `env.belief_view(i)` + their seat index) and rejects an unsupported player
  count. Wins accumulate from the sparse terminal rewards (exactly one +1 per
  completed game, so `episodes = wins.sum()`). Budget is exactly one of
  `n_steps` (sync-free) or `n_episodes` (syncs on the win count each step; may
  overshoot when several lanes finish together; capped at
  `_MAX_STEPS_PER_EPISODE` as a non-termination guard). Not a fused rollout; a
  `lax.scan` version is the obvious next step if evaluation throughput starts
  to matter.

## search/

Both agents determinize once per move: `sample_world` at the root, then search
in the sample (PIMC, not ISMCTS — the simulated opponent shares the sampled
world). Residual approximations: a sampled in-tree draw's identity is visible
one ply ahead (committed per node, not a chance node); the search is a
*single* determinization (ensemble over K root samples — vmap + average
`action_weights` — is the known next upgrade); the in-tree opponent sees the
sampled world (strategy fusion), though count-only value terms blunt what it
can exploit.

- `greedy.py` — `make_greedy(value) -> BeliefPolicy`: one-step lookahead. All
  560 successors in one `vmap(apply_action)` over the static row decode, gated
  by the caller's mask (illegal rows no-op but are masked to -inf anyway);
  `vmap(value)` over successors + 1e-4 tie-break noise, masked argmax.
  `lookahead_policy = make_greedy(heuristic_value)`.
- `mcts.py` — `make_mcts(value, num_simulations=32, max_num_considered_actions
  =16, value_scale=20) -> BeliefPolicy`: `mctx.gumbel_muzero_policy` with the
  engine as `recurrent_fn` and embedding = batched `(layout, state)`. Frame
  convention: priors/values are the node's player-to-move's (root: the seat
  asked to act); `discount` is -1 when the mover switches, +1 when the same
  player continues (multi-move turns), 0 into terminals (absorbing — avail is
  gated with `~terminal` so won states no-op). Reward ±1 in the actor's frame
  on a winning transition; leaf values are `tanh(value/value_scale)` so the
  heuristic's scale is commensurate with the ±1 terminal reward. Exact
  zero-sum framing at 2 players; at 3-4 the sign-flip discount is the
  *paranoid* reduction (every opponent maximizes against the mover) — a known
  approximation, scalar backups can't express max^n. Single-game policy:
  batch-of-1 inside, composes under outer `vmap`/`jit` (verified). The
  `# type: ignore[call-arg]` on the mctx dataclass constructors is for chex
  dataclasses being opaque to mypy.

## cli.py

The `catan-agents` console script (`[project.scripts]`), argparse subcommands —
only `compare` so far; tournaments etc. slot in as new subparsers. `compare`
is a seat-swapped head-to-head: two `n_episodes` evaluate runs (`seed`,
`seed + 1`) with the agents' seats exchanged, combined into `CompareResult`
(totals plus a per-first-seat split). Pure Python over the registry — not in
the jaxtyping hook list.

`__init__.py` exports the `POLICIES` registry (name -> `AgentSpec`); it is the
single list of shipped agents, consumed by both the protocol tests and
catan-render's bot seam (`catan_render/bots.py`, which dispatches on
`AgentSpec.observes` and filters seat counts).

Tests are protocol-level only (`tests/test_policies.py`), parametrized over
every agent in `POLICIES` and run at `max(spec.n_players)` (= 4 for all
shipped agents) through whichever protocol the spec declares — legality (a
pick must be legal whenever the lane has a legal move, checked through 150
self-play steps), seeding reproducibility (same seed -> identical self-play
action trajectory), and short self-play rollouts that must complete games.
`tests/test_sample.py` covers `sample_world`'s public-record invariants
(infrastructure, not a policy, so it gets unit tests). No per-agent
internal-logic tests; a new agent just registers in `POLICIES`.
`tests/conftest.py` installs the jaxtyping/beartype import hook for all
`catan_agents` modules, same pattern as catan-engine.
