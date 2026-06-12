# 0002 — linear value fitting (predict target)

Status: concluded (framework adopted; fitted weights not — fail on the
strict gate)

## Hypothesis

Logistic-regression weights fit on game outcomes against a known opponent
(greedy) recover or beat the hand-tuned heuristic weights when deployed in
one-step lookahead; the pipeline generalizes to any feature subset.

## Setup

`uv run python experiments/0002_linear_value_fitting/run.py` — config at the
top of run.py (the shared machinery is `experiments/_value_fitting.py`; this
report's numbers predate the framework extraction, same logic). Collect ~190k positions (3,150 episodes, B=64)
from lookahead(heuristic) vs greedy; rows are seat-0-minus-seat-1
`BoardFeatures` (21 terms, including the new `second_spot` / `reach` /
`army_lead`), labels seat-0-won. Fit a candidate matrix — three feature
subsets × {logistic, sign-constrained NNLS} × {all positions, early half} —
with episode-grouped held-out AUC; select by cheap seat-swapped *match
probes* vs the opponent (n=120); gate the winner against
lookahead(hand-tuned) at n=300.

## Results

From `runs/0002_linear_value_fitting/2026-06-12T223650Z`
(`fits.json`, `probes.json`):

- v1 (naive: train-accuracy selection, no constraints) deployed at 73.0% vs
  greedy against the hand weights' 82.5%, and 26.7% head-to-head. The fitted
  coefficients show why: `production` got **+0.008** (its credit
  redistributed into correlated `diversity`/`progress`/`vp`) and `over` came
  out *positive* (winners hold more cards) — fine for prediction, wrong for
  decisions.
- v2 candidates: held-out AUC is nearly flat across candidates (0.831–0.843)
  while their match probes span **52.8%–78.0%** — AUC cannot rank decision
  quality. Early-position fits (AUC ~0.73) probe no better. NNLS fixes the
  sign pathologies but probes below logistic on the same subset.
- Best candidate (`hand_terms/all/logistic`, selected by probe): **75.2% vs
  greedy** against the hand weights' 77.8% (statistically even at n≈200
  each), but **42.7% ± 2.8%** head-to-head against lookahead(hand-tuned) —
  gate (lower 2σ > 50%) fails.

## Decision

The framework is adopted: collect → fit subsets → probe-by-match → gate, all
reusable for any future feature set (`value.make_linear` is the deployment
seam). The fitted weights are not: outcome regression recovers
hand-tuned-level play against the known opponent from data alone — which
validates the features — but does not beat the hand weights, because
predicting *who wins* is not the same objective as ranking *successor
states*. Next lever if pursued classically: a decision-aware objective
(preference pairs over successors, or per-phase weights); otherwise this is
exactly catan-learn Stage 1's job with a nonlinear model.
