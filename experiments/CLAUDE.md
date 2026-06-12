# experiments — internal notes

The lab-notebook contract is in README.md (numbered immutable dirs, reports,
JOURNAL.md, git-ignored `runs/`). This file documents the shared machinery.

## `_lib.py`

`start_run` (run dir + manifest pinning git commit / uncommitted-diff digest /
config / environment), `Run.log` (metrics.jsonl), `Run.save_json`,
`Run.finish` (result.json + the printed verdict).

## `_value_fitting.py` — linear fits over the engineered features

The framework for fitting/searching weights over
`catan_agents.internal.feature_engineering.BoardFeatures`, deployed through
`value.make_linear` into one-step lookahead. One experiment = one `CONFIG`
passed to `run_experiment`:

- `features` — list of `BoardFeatures` names (the subset under study).
- `target` — the optimisation objective:
  - `"predict"`: collect positions from lookahead(heuristic) vs the opponent
    (feature *differences*, outcome labels, episode ids, in-game fractions),
    fit {logistic, sign-constrained NNLS} × {all positions, early halves},
    rank by match probes.
  - `"maximise"`: cross-entropy search over weight vectors with the measured
    seat-swapped win rate vs the opponent as the objective; starts from the
    hand weights (`HAND_WEIGHTS`, new features at 0), shares the evaluation
    seed within a generation (common random numbers) and rotates it across
    generations.
- `opponent` — the known opponent (a `POLICIES` name), used for data,
  probes, and the deployment bench.
- Budgets: `collect` (steps/batch/snapshot cadence), `maximise`
  (iterations/population/elites/eval_games/sigma), `probe_games`,
  `bench_games`, `gate_games`.

The winner is benched vs the opponent beside the hand-tuned baseline and
gated head-to-head against `lookahead(heuristic)`: pass iff the lower
2σ bound clears 50%.

Lessons baked into the design (exp 0002):

- **Select by matches, never fit metrics**: held-out AUC was flat
  (0.831–0.843) across candidates whose match probes spanned 52.8–78.0%.
- **Prediction is not control**: unconstrained outcome regression
  redistributes correlated credit (production fit at +0.008, the discard
  penalty fit *positive*); NNLS pins the signs, early-position fits force
  economy to carry signal — both exist as candidates for this reason.
- **Group the held-out split by episode** — rows within a game are
  correlated, a row-level split leaks.
- Each distinct weight vector is a fresh value closure: `evaluate` retraces
  its scan per candidate (~seconds), which is most of a maximise
  generation's overhead — budget `eval_games` accordingly.
