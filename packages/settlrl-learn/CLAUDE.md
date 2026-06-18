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
- `train.py` — full-batch SGD only (the toy fitter); the real loop is the
  AlphaZero modules below. The value head is a win-probability logit: the
  searches read leaves as `tanh(v / value_scale) = 2P(win) − 1`, so logistic
  targets line up with the June 11 calibration finding (P(win) =
  σ(0.053·v_heuristic)). The AZ net's logit maps in with `value_scale=2`
  (`tanh(logit/2) = 2P−1`).
- `model.py::AZParams` — the shared-trunk value+policy net (`make_az` adapts it
  onto the search's `value`/`prior` seams). Plain-JAX, so the package root
  imports it without pulling training deps.
- `graph.py` / `architectures.py` — the board-as-graph featurization
  (`board_sample` → a `Sample` of per-node/-edge/global features + the
  engineered vector, fixed topology as module constants) and the equinox
  architectures over it (`mlp_engineered` / `mlp_flat` / `deepset` / `gnn`,
  via `make_model`). Training-side (equinox/jraph), *not* imported by the
  package root; experiment 0003 composes them. `deepset`/`gnn` are invariant
  under the board's symmetry group and the player relabeling (readout pools
  over nodes / ownership is read relatively); `mlp_flat` is not, by design.
  These invariances are enforced in `tests/test_architectures.py` against the
  symmetry generators in `tests/_symmetry.py` — the board's automorphism group
  is order 6 (D3), not the bare graph's D6, because the harbors are only 3-fold
  symmetric (the port-preserving subgroup).
- **AlphaZero loop** (training-side, *not* imported by the package root — keeps
  the shipped-model path lean; experiment 0004 composes them):
  - `selfplay.py::self_play` — batched n-player self-play, the net guiding the
    re-determinizing search (`make_search_weights` for the improved policy as
    target); features are on the *true* board (net learns the belief-averaged
    value), values are the eventual win/loss of the acting seat.
  - `alphazero.py` — the flashbax item-buffer wrapper, the policy-CE +
    value-logistic loss + optax adamw `make_train_step`, `arena` (seat-swapped
    vs `lookahead(heuristic)`, the Stage-1 gate), and the `learn` loop. Value
    target is pure outcome `z` for now; Canopy's `(1−α)z + α·q` blend awaits a
    search that exposes root Q (see the Canopy reference below).
  - `azgnn.py` — the AlphaZero loop with a **GNN trunk** (`AZGraphNet` = a
    `graphnet.GraphNet` over `graph.board_sample` with value + policy heads,
    experiment 0003's recommended `gn_global`). Mirrors `selfplay`/`alphazero`
    for an equinox model: `make_az_gnn` adapts onto the search seams, `self_play`
    records the board graph, and `learn` runs the loop over a flashbax on-device
    replay. The whole `GNNState` is **eqx-serialised** every iteration for
    bit-exact resume (eqx's native serialiser fits the equinox model where
    orbax's pure-array assumption does not; the per-iteration RNG is a pure
    function of `seed` and the iteration index). `reuse` caps updates/iter at the
    AZ sample-reuse factor (the value-overfit fix), and a held-out `eval_frac`
    gives the `val_value_acc` generalization metric. Experiment 0004's `net=gnn`
    variant composes it.
  - `train_state.py::TrainState` — the whole mutable run state (params,
    optimiser moments, replay buffer, iteration, best), orbax-serialised for
    **bit-exact resume**: `learn` rebuilds the static optimiser/buffer from
    hyperparameters, restores the state into them, and continues — the
    per-iteration RNG is a pure function of `seed` and the iteration index, so a
    resumed run is bit-identical to one that never stopped (tested:
    params/opt/buffer all exact).

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
