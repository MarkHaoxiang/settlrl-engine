# settlrl-learn — internal notes

Training-side package: depends on engine + agents, never the reverse.
Anything an agent needs at play time (the plain-JAX `mlp` forward, params as
an ordinary pytree, `.npz` artifacts) is deliberately dependency-free so a
trained model can ship without training libraries.

- `features.py` — engineered blocks mirror the heuristic's terms (production,
  expansion, ports, awards): we know they carry signal, and a model that
  cannot beat the heuristic *from the heuristic's own inputs* is not worth
  shipping. `FEATURE_DIM` is computed at import via `jax.eval_shape` on a
  2-player template (the own/max/mean aggregation makes the width
  player-count invariant, so 2p suffices).
- `train.py` — full-batch SGD only; a real optimiser and the self-play data
  pipeline (over `BatchedSettlrlEnv.rollout(actor=...)`) arrive with Stage 1.
  The value head is a win-probability logit: the searches read leaves as
  `tanh(v / value_scale) = 2P(win) − 1`, so logistic targets line up with
  the June 11 calibration finding (P(win) = σ(0.053·v_heuristic)).

The gates (June 11 plan, evidence in settlrl-agents/CLAUDE.md): Stage 1 ships a
value only if `lookahead(net)` beats `lookahead(heuristic)` at ≥2σ, n≥400
(`settlrl-agents bench`); Stage 2 reruns the sims ladder — depth pays nowhere
with the stationary heuristic leaf, and that falsification is the reason this
package exists; Stage 3 (policy head, self-play iteration) only after.

## Reference: Canopy (`cullback/canopy`)

A Rust AlphaZero framework whose flagship example is a 1v1 Catan agent
(`nexus-v3`, claimed "strongest public 1v1 Catan agent" — unbenchmarked against
ours). It is the point past our leaf-is-the-ceiling gate: learned policy + WDL
value head, self-play, Gumbel improved-policy interior selection + PUCT/Dirichlet
root (800 sims), explicit chance nodes for dice and dev draws, and Single-Observer
ISMCTS that filters per-simulation legality in a custom tree — the part our
mctx-based search can't express. It is 1v1 only, so it never meets the 3-4p
paranoid-frame / opponent-model problem, and it *disables determinization during
self-play* (the net learns the Bayesian-average-over-hands policy; determinize
only at play time).

Techniques worth lifting into our Stage 1 training, both aimed at Catan's dice
variance (the variance-starved-depth problem):

- **Value-target blending** `target = (1−α)·z + α·q` (game outcome `z` blended
  with MCTS root Q), α ramped linearly 0 → max over early iters. Pure `z` is too
  noisy for a dice game; Q averages over sims once the value head is decent.
- **EMA auxiliary value heads** at horizons (e.g. `[4, 10, 30]` for ~90-move
  games), trained on `ema = α·Q[t] + (1−α)·ema`, sharing the trunk.
- **Playout-cap randomization** (KataGo): most moves a small search, a fraction
  the full budget; only full-search positions contribute policy targets, all
  contribute value targets.

Repo + METHODS.md + examples/catan/OPTIMIZATIONS.md; see [[canopy-reference]].
