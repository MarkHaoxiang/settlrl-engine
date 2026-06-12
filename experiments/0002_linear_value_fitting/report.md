# 0002 — linear value fitting

Status: live framework. Variants concluded: predict (fail), maximise vs a
fixed opponent (fail), **self_play (pass — the first optimized weights to
beat the hand-tuned heuristic)**

## Hypothesis

Weights optimized against a known opponent (greedy) — fit to *predict*
outcomes, or searched to *maximise* the measured match win rate — recover or
beat the hand-tuned heuristic weights when deployed in one-step lookahead;
the pipeline generalizes to any feature subset.

## Setup

`uv run python experiments/0002_linear_value_fitting/run.py [predict|maximise]`
— config at the top of run.py; the shared machinery is
`experiments/_value_fitting.py` (the predict numbers below predate the
framework extraction, same logic; the maximise run originally lived as
experiment 0003 before the targets merged). Collect ~190k positions (3,150 episodes, B=64)
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

### Maximise target (CEM, from `runs/0003_cem_value_weights_vs_greedy/2026-06-12T231337Z`)

- 3 generations × 6 members × 60-game evaluations around the hand weights:
  the search shifted weight out of `race` (0.8 → 0.1) and `progress` into
  `n_dev` (+2.09) and `best_spot` — against greedy, endgame urgency pays
  less than mid-game economy.
- Deployment bench: **80.8% vs greedy** against the hand weights' 77.8% —
  the search beat its objective's baseline.
- Head-to-head vs lookahead(hand-tuned): **43.3% ± 2.8%** — gate fails,
  almost exactly where the predict target landed (42.7%).

### Self-play variant (`runs/0002_linear_value_fitting/2026-06-12T230644Z`)

Three rounds of CEM where each round's opponent is the *current champion*
(round 0: the hand weights) and a challenger is accepted only by winning the
acceptance match. Round 0 rejected (48.0%), round 1 accepted (57.3%), round
2 could not dethrone the new champion (50.0%). The final champion:

- **56.1% vs lookahead(hand-tuned)** head-to-head (n=310, lower 2σ 50.5%) —
  the gate passes;
- **80.9% vs greedy** against the hand weights' 77.8% — better on both axes,
  no specialist trade-off.

The champion's main moves off the hand weights: more `vp` (+4.2), the knight
cluster up (`knights` +0.47, `held_knights` +0.54, `n_dev` +0.40 — the army
race is underweighted by hand), `n_roads` +0.42, and `progress` down (−0.70:
hoarding toward the closest build is overrated once opponents punish it).

## Decision

The framework is adopted (`value_fitting.py` in this directory;
`value.make_linear` is the deployment seam; `run.py [variant]` selects the
config). Fixed-opponent optimization breeds specialists — predict and
maximise both lost ~43% head-to-head despite matching/beating the hand
weights against their objective opponent — and the self-play ladder is the
fix: optimizing against the evolving champion produced weights that beat the
hand-tuned heuristic at the 2σ gate *and* widened the greedy margin.
Adopting the champion as `make_heuristic` defaults is justified by the gate
but cascades (every agent shares the leaf; ladder claims re-measure) —
deferred to its own change. Further rounds/bigger budgets are config
changes; the nonlinear continuation is catan-learn Stage 1.
