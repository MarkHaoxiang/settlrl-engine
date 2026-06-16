# experiments — internal notes

The lab-notebook contract is in README.md. The unit here is the experiment
*framework*: a numbered directory holding `run.py` (named variants selected
by argv), its own helper modules, and a report that accumulates one section
per concluded variant. Don't scaffold a new number for a question an
existing framework can express as a config — extend its `VARIANTS` instead;
git history is the framework's changelog, the report its conclusions.

## `_lib.py` (shared by every framework)

`start_run` (run dir + manifest pinning git commit / uncommitted-diff digest /
merged config / environment), `Run.log` (metrics.jsonl), `Run.save_json`,
`Run.finish` (result.json + the printed verdict).

## `0002_linear_value_fitting/` — linear fits over the engineered features

`value_fitting.py` optimizes weights over
`settlrl_agents.internal.feature_engineering.BoardFeatures`, deploys them
through `value.make_linear` into one-step lookahead, and gates against the
hand-tuned weights (pass iff the lower 2-sigma bound clears 50%). Config
knobs: `features` (list of `BoardFeatures` names), `target`
(`predict` — outcome fits, {logistic, sign-constrained NNLS} × {all, early
positions}, ranked by match probes; `maximise` — cross-entropy search with
the measured seat-swapped win rate as the objective, common random numbers
within a generation), `opponent` (a `POLICIES` name, or `"self"` for the
self-play ladder: each round's opponent is the current champion and a
challenger replaces it only by winning the acceptance match), `rounds`, and
the budgets (collection, CEM, probe/bench/gate games).

Lessons baked into the design:

- **Select by matches, never fit metrics**: held-out AUC was flat
  (0.831–0.843) across candidates whose match probes spanned 52.8–78.0%.
- **Prediction is not control**: unconstrained outcome regression
  redistributes correlated credit (production fit at +0.008, the discard
  penalty fit *positive*); NNLS pins the signs, early-position fits force
  economy to carry signal — both exist as candidates for this reason.
- **Fixed-opponent optimization breeds specialists**: both targets beat or
  matched the hand weights against their objective opponent and lost ~43%
  head-to-head against the hand-tuned lookahead — hence the self-play
  variant.
- **Group the held-out split by episode** — rows within a game are
  correlated, a row-level split leaks.
- Each distinct weight vector is a fresh value closure: `evaluate` retraces
  its scan per candidate (~seconds), which is most of a maximise
  generation's overhead — budget `eval_games` accordingly.
