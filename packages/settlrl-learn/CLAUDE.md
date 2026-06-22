# settlrl-learn — internal notes

Training-side package: depends on engine + agents, never the reverse.
Anything an agent needs at play time (the plain-JAX `mlp` forward, params as
an ordinary pytree, `.npz` artifacts) is deliberately dependency-free so a
trained model can ship without training libraries.

`experiment/` is the lab harness for `experiments/` (`Run`/`start_run`
bookkeeping + the pydantic/OmegaConf `Config` base) — moved here from
settlrl-agents (it is a training-side concern; this keeps the play/serve
library free of `pydantic`/`omegaconf`). *Not* imported by `__init__`, so
`import settlrl_learn` stays free of those; `pydantic`/`omegaconf` are learn
deps only because this subpackage uses them.

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
- **Network definitions live under `nn/`**; **the training loop under `training/`**.
  `nn/__init__` is import-light (no equinox/jraph) so the shipped plain-JAX path —
  `features` + `nn/mlp.py`, reached by the package root — pulls no training deps.
  A guard test (`tests/test_import_light.py`, run in a subprocess) asserts
  `import settlrl_learn` pulls no equinox/flashbax/optax/orbax/jraph.
- `nn/mlp.py::AZParams` — the shared-trunk value+policy net (`make_az` adapts it
  onto the search's `value`/`prior` seams). Plain-JAX, so the package root
  imports it without pulling training deps.
- `nn/graph.py` / `nn/architectures.py` — the board-as-graph featurization
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
- **Net definitions** (training-side, *not* imported by the package root —
  experiment 0004 and the loop compose them):
  - `nn/graphnet.py::GraphTrunk` — the shared message-passing trunk (encoders +
    layers → per-node embeddings + global + pooled readout); `GraphNet` (single
    head) and `nn/board_gnn.py::BoardGNN` both build their heads on it.
  - `nn/action_layout.py` — the static map from the flat 662 action space to its
    board structure (which actions are per-vertex / -edge / -tile vs. dense
    "other"), and `SCATTER` to place a structure-factored head's compact logits
    back into the flat vector. The robber/knight *victim* collapses to
    no-steal/steal (the opponent-relative features can't individuate victims, so
    a per-victim logit could not be player-relabel invariant).
  - `nn/board_gnn.py::BoardGNN` — the value+policy net (`GraphTrunk` over
    `nn/graph.py::board_sample`; the `gnn_seams` adapter lives here too). Value
    and policy heads **split right after the trunk** (no shared head MLP). The
    policy is **structure-factored**: a shared per-vertex / per-edge (symmetric
    endpoints) / per-tile (corner-vertex mean) head emits spatial-action logits,
    a dense head the rest, plus a per-type bias (class balance). Tests in
    `tests/test_architectures.py` enforce: value invariant, policy *equivariant*
    under board symmetry (an action at v maps to the action at σv —
    `action_permutation`), both invariant under player relabeling.

- **The training loop** (`training/`, training-side, *not* imported by the
  package root): one net-agnostic self-play → replay → train → arena loop behind
  a `Backend` seam, so the flat-MLP and board-GNN paths share it. Experiment
  0004 composes it (`net=mlp|gnn`).
  - `training/backend.py` — the `Backend` protocol (the net-specific surface:
    `init` / `seams` / `play_agent` / `setup_policy` / `observe` / `to_item` /
    `empty_item` / `init_opt` / `make_step` / `eval_metrics`) and `RunState`
    (net + optimiser moments + replay buffer + iteration + best). `RunState` is
    **eqx-serialised** (`save_run_state`) — eqx's leaf serialiser fits both an
    equinox module and a plain-JAX pytree, so it replaced orbax for *both*
    backends.
  - `training/selfplay.py::self_play` — batched n-player self-play, the search
    (net's or a fixed teacher's) guiding the re-determinizing moves and improved
    policy. The backend's `observe` records the *true* board (net learns the
    belief-averaged value); values are the acting seat's eventual win/loss.
    `setup_fn` (when given) plays the setup phase with a fixed policy and those
    positions are *not* recorded (the GNN path; the MLP path passes `None` and
    the net plays setup too).
  - `training/loop.py::learn` — the loop: per-iteration RNG is a pure function of
    `seed` and the iteration index, so `resume_from` (a `runstate.eqx`) continues
    bit-identically (tested in `tests/`-adjacent resume checks for both
    backends). `reuse` caps updates/iter at the AZ sample-reuse factor (the
    value-overfit fix); a held-out `eval_frac` feeds the backend's `eval_metrics`
    (the `val_*` generalization metrics); `teacher_value`/`teacher_iters`
    warm-start from a fixed strong search (the cold-start fix). Value target is
    pure outcome `z` for now; Canopy's `(1−α)z + α·q` blend awaits a search that
    exposes root Q (see the Canopy reference below).
  - `training/arena.py::arena` — the net's win rate vs. a `POLICIES` opponent,
    seat-swapped at 2p (`lookahead` = the Stage-1 gate; `random` = the
    lower-bound sanity check); the play agent comes from `backend.play_agent`.
  - `training/mlp_backend.py::MLPBackend` — the `AZParams` net over the
    engineered feature vector; **unmasked** policy CE + value-logistic loss,
    optax adamw, the net plays setup itself.
  - `training/gnn_backend.py::GNNBackend` — the `BoardGNN` net over the board
    graph; **masked** policy CE (softmax over the legal set only) + value loss,
    eqx-filtered optax step. `setup_policy` (a fixed `lookahead`/expectimax
    opener) plays the setup phase in both self-play and the arena;
    `make_net_agent` composes setup + the net's search; `gnn_loss` is the masked
    loss (its finiteness is contract-tested).

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
ISMCTS that filters per-simulation legality in a custom tree (our search now does
this too — `settlrl_agents.search.ismcts`, which retired the mctx engine). It is
1v1 only, so it never meets the 3-4p
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
