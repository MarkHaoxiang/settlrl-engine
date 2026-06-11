# catan-learn — internal notes

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
  pipeline (over `BatchedCatanEnv.rollout(actor=...)`) arrive with Stage 1.
  The value head is a win-probability logit: the searches read leaves as
  `tanh(v / value_scale) = 2P(win) − 1`, so logistic targets line up with
  the June 11 calibration finding (P(win) = σ(0.053·v_heuristic)).

The gates (June 11 plan, evidence in catan-agents/CLAUDE.md): Stage 1 ships a
value only if `lookahead(net)` beats `lookahead(heuristic)` at ≥2σ, n≥400
(`catan-agents bench`); Stage 2 reruns the sims ladder — depth pays nowhere
with the stationary heuristic leaf, and that falsification is the reason this
package exists; Stage 3 (policy head, self-play iteration) only after.
