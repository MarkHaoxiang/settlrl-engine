# The search vs. the ISMCTS taxonomy — design and audit

The one search (`search/__init__.py`, `make_search`) is a **re-determinizing
Gumbel-MuZero** built on `mctx`: a single tree whose every simulation draws a
fresh `sample_world` determinization and replays the root path under it. This
note places it in the Information Set MCTS taxonomy (Cowling, Powley &
Whitehouse, *Information Set Monte Carlo Tree Search*, IEEE TCIAIG 4(2), 2012;
and the 2014 *Information capture and reuse* follow-up), records what the
consolidation kept and dropped, and audits the frames. Conclusion up front: the
search delivers ISMCTS's per-simulation re-determinization but not its
availability-count UCB; this is measured at **parity, not a win**, because the
binding constraint is the stationary heuristic leaf, not the determinization
pathologies — so the lever is a learned value (experiment 0003), after which the
remaining ISMCTS structure (a per-player opponent model) becomes worth its cost.
No correctness bugs were found.

## Taxonomy: where the search sits

Cowling et al. contrast three families for hidden-information games:

- **Determinized UCT (PIMC).** Sample N full states consistent with the
  information set; build one *independent* UCT tree per determinization, each
  frozen for its whole depth; pick the move maximising visits across trees.
- **SO-ISMCTS.** A *single* tree of information sets from the root player's
  view. Each iteration re-samples a determinization and restricts the descent
  to actions legal in it; statistics are *shared* across determinizations, and
  UCB uses an **availability count** (how often a node was *selectable*) in
  place of the parent visit count — the subset-armed-bandit correction that
  stops rarely-legal actions being over-explored.
- **SO-ISMCTS+POM / MO-ISMCTS.** Add partially-observable opponent moves and
  per-player trees with a proper opponent model (each player searches its *own*
  information sets) — what actually removes opponent-node strategy fusion.

The shipped search re-determinizes **per simulation** into one shared tree
(SO-ISMCTS's defining move), via the action-path embedding: `mctx` calls
`recurrent_fn` once per simulation with a fresh rng, so that call samples a world
and replays the stored path under it before evaluating the new edge. It does
**not** have SO-ISMCTS's availability-count UCB or per-simulation legality
restriction — `mctx`'s bandit is Gumbel/PUCT over a fixed action axis with
ordinary visit counts, and the whole search is one jitted, fixed-shape kernel,
so an opponent path-action illegal under the resampled world simply no-ops
(INVALID) rather than the edge being unavailable that iteration. Full ISMCTS
(availability counts + per-node legal sets) would mean leaving `mctx` for a
custom tree.

## The two determinization pathologies

1. **Strategy fusion** — a determinized searcher implicitly assumes it can make
   information-set-dependent choices it cannot actually make, because each tree
   *knows* the sampled hidden state. The per-simulation re-determinization fixes
   this on **leaf values**: a node's backed-up value integrates over resampled
   worlds rather than committing to one frozen sample. It does *not* fix
   opponent-node fusion under our paranoid reduction (the in-tree opponent still
   sees the world it is simulated in); that needs MO-ISMCTS's per-player model.
2. **Non-locality** — the value backed up can depend on lines a rational
   opponent would never enter, because averaging over determinizations leaks
   information between subtrees. Inherent; not separately addressed, and not the
   dominant error here.

## The consolidation (what merged, what was dropped)

The search replaced a quartet — `mcts` (frozen-world PIMC), `smcts`
(stochastic-MuZero chance nodes), `ismcts` (this re-determinization, formerly
experimental), and `lookahead` — that all measured at ~parity with one another:

- **`smcts`'s explicit dice/dev chance nodes were dropped, not ported.** They
  were falsified as a strength lever (56.7% vs lookahead pooled n=319, 49.3%
  h2h, ~2× wall-clock; the dev-draw chance node strength-neutral 50.5% n=210;
  64→128 sims 53.3%→49.5%). The immediate roll is valued exactly at the leaf
  (11-roll expectation on `ROLL_DICE` children) and deeper chance is integrated
  by the per-simulation resampling (each world carries its own key), so explicit
  chance nodes buy nothing over the leaf here.
- **`mcts`'s frozen-world PIMC was dropped** in favour of its principled
  superset. Re-determinization measured *parity, not a win*, over frozen-world
  at 3p (`0.352 ± 0.031`, n=244, 32 sims; chance 0.333, lower-2σ 0.291), 64
  worlds `0.307 ± 0.030` (n=241, flat — more worlds doesn't help), and ~tie at
  2p (`0.47`, n=17; 2p belief ≈ exact, so resampling is near-degenerate). Cost:
  one `sample_world` + up to `max_depth` engine steps per simulation, markedly
  slower at equal sims. Integrating a *biased* (stationary heuristic) leaf over
  more worlds just estimates the same biased value more precisely — the
  leaf-is-the-ceiling finding made sharp.
- **`lookahead` became the `num_simulations=0` special case** (root one-step
  sweep), the only configuration that offers trades.

## Audit notes (no bugs found)

- The two-sided (paranoid) reward/discount in `_transition` is consistent: leaf
  values are evaluated in the searcher's frame and re-signed into the mover's
  side, discounts flip only across the side boundary, terminals absorbed. The 2p
  equivalence to per-mover flipping is tested (632/640 same picks).
- `ROLL_DICE` children back up the **11-roll expectation** from the pre-roll
  state, in the searcher's frame, then re-signed — correct. The subtree below is
  still conditioned on one sampled roll while the node value is the expectation
  (a documented one-ply approximation).
- `expected_roll_value` approximates a 7 as "no payout, state as-is", dropping
  the robber/discard resolution. Documented; the most defensible place to add
  fidelity once a learned leaf makes the search sensitive to it.
- The root mask is taken once from the true view (the root player's own legality
  is fully observable); interior legality is recomputed per replayed world.

## The remaining lever

The search's `value` is the seam: the same per-simulation integration that is
parity over a heuristic leaf should start to pay once the leaf is a learned
value whose error shrinks under search (settlrl-learn Stage 1 / experiment
0003). At that point the untested ISMCTS structure — availability-count UCB and,
above all, an MO-ISMCTS per-player opponent model replacing the paranoid frame —
becomes the next 3-4p lever, gated through `settlrl-agents bench` (n ≥ 240).
