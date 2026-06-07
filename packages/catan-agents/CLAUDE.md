# catan-agents — internal notes

Pure-JAX Catan agents over `catan-engine`'s public flat-action seam
(`catan_engine.env.N_FLAT` / `flat_to_action` / `BatchedCatanEnv.flat_mask`,
plus `step` / `available` / `flat_available` and `mechanics.action.apply_action`
for the model-based agents).

Three subpackages, split by information access: with two players the full
board state is publicly inferable (all resource flows are public; dev cards
are a distribution over the known deck composition), so `two_player` agents
may consume `(BoardLayout, BoardState)` directly as a world model. With 3-4
players opponent↔opponent steals and discards are hidden, so partial
observability genuinely binds — `four_player` agents must stay obs-based.
`shared` is the count-agnostic core.

## shared/

- `policy.py` — the seat protocols: `Policy` (single-game `(key, obs, mask) ->
  flat action`; callers `vmap` for batches) and `StatePolicy` (`(key, layout,
  state, player, mask)`, full single-game state), the `FlatMask` / `FlatAction`
  jaxtyping aliases, and `AgentSpec` (policy + `observes` kind + supported seat
  counts) — the registry value type. Policies are masked-argmax style: with no
  legal move the index is arbitrary and the engine rejects it as `INVALID`
  (the lane stalls until auto-reset), matching `BatchedCatanEnv.random_actions`.
- `value.py` — `ValueFunction` protocol: single-game `(layout, state, player)
  -> scalar`, higher better, arbitrary scale. `heuristic_value` = own strength
  minus best opponent's; strength = 10·VP (buildings + awards + VP cards — own
  exact, opponents' expected via the deck's 5/25 VP share over their count) +
  pips of own buildings (city 2x, robber tile zeroed) + 0.3·Σ√(resource counts)
  (diversity; makes the cheapest discard the most-held resource) + 1.5·dev
  count. Reads only 2p-inferable fields: exact resources OK, opponent dev hands
  only as counts. Also hosts `tile_pips` / `vertex_pips` (used by greedy too).
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
  auto-reset): every seat's vmapped agent picks a move in every lane each step
  and the acting seat's pick is kept (`picks[agent_selection, lanes]`) — n_seats
  policy evals per step, fine for <= 4 seats. `_seat` adapts both kinds (obs
  seats get `env.observe(i)`, state seats `env.board` + their seat index) and
  rejects an unsupported player count. Wins accumulate from the sparse
  terminal rewards (exactly one +1 per completed game, so
  `episodes = wins.sum()`). Budget is exactly one of `n_steps` (sync-free) or
  `n_episodes` (syncs on the win count each step; may overshoot when several
  lanes finish together; capped at `_MAX_STEPS_PER_EPISODE` as a
  non-termination guard). Not a fused rollout; a `lax.scan` version is the
  obvious next step if evaluation throughput starts to matter.

## two_player/

Both agents determinize the root state from the policy key: re-key
(`state._replace(key=...)`, so search samples its *own* dice / steals / dev
draws instead of foreseeing the env's actual outcomes) **and**
`belief.redeal_dev_cards` (the opponent's hidden card identities replaced by a
posterior sample). Residual approximations: a sampled draw's identity is
visible one ply ahead (committed per node, not a chance node), and the search
is a *single* determinization — the simulated opponent shares the sampled
world (PIMC, not ISMCTS), though the count-only value function means their
in-tree choices can barely exploit our card identities (the main channel left
is `_terminal`/`_winner` seeing our hidden VP cards).

- `belief.py` — `redeal_dev_cards(key, state, player)`: the unseen pool
  (`dev_deck + opponent hand`, which equals deck composition − own hand −
  publicly played cards by conservation, so it is honest) re-dealt to the
  opponent without replacement via a static 25-slot card view
  (`_CARD_TYPE`/`_CARD_RANK`, in-pool slots noised, top-`n_held` taken);
  the remainder becomes the deck. Hand sizes (public) are unchanged.
  Two-player only (`opponent = 1 - player`). Invariant tests in
  `tests/test_belief.py` (infrastructure, not a policy, so it gets unit tests).
- `greedy.py` — `make_greedy(value) -> StatePolicy`: one-step lookahead. All
  560 successors in one `vmap(apply_action)` over the static row decode, gated
  by the caller's mask (illegal rows no-op but are masked to -inf anyway);
  `vmap(value)` over successors + 1e-4 tie-break noise, masked argmax.
  `lookahead_policy = make_greedy(heuristic_value)`.
- `mcts.py` — `make_mcts(value, num_simulations=32, max_num_considered_actions
  =16, value_scale=20) -> StatePolicy`: `mctx.gumbel_muzero_policy` with the
  engine as `recurrent_fn` and embedding = batched `(layout, state)`. Frame
  convention: priors/values are the node's player-to-move's (root: the seat
  asked to act); `discount` is -1 when the mover switches, +1 when the same
  player continues (multi-move turns), 0 into terminals (absorbing — avail is
  gated with `~terminal` so won states no-op). Reward ±1 in the actor's frame
  on a winning transition; leaf values are `tanh(value/value_scale)` so the
  heuristic's scale is commensurate with the ±1 terminal reward. Child priors
  mask illegal rows at -1e9 via `env.flat_available`; the root mask uses
  mctx's `invalid_actions`. Single-game policy: batch-of-1 inside, composes
  under outer `vmap`/`jit` (verified). The `# type: ignore[call-arg]` on the
  mctx dataclass constructors is for chex dataclasses being opaque to mypy.

## four_player/

Placeholder (docstring only) for belief-state / determinization agents.

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
every agent in `POLICIES` and run at `max(spec.n_players)` through whichever
protocol the spec declares — legality (a pick must be legal whenever the lane
has a legal move, checked through 150 self-play steps), seeding
reproducibility (same seed -> identical self-play action trajectory), and
short self-play rollouts that must complete games. No per-agent internal-logic
tests; a new agent just registers in `POLICIES`. `tests/conftest.py` installs
the jaxtyping/beartype import hook for all `catan_agents` modules, same
pattern as catan-engine.
