# 0003 — CEM value weights vs greedy (maximise target)

Status: concluded (fail on the strict gate; the maximise path is validated)

## Hypothesis

Cross-entropy search over the hand-tuned terms' weights, with measured match
win rate vs greedy as the objective, finds weights at least as strong as the
hand-tuned ones (the predict target plateaued just below them — exp 0002).

## Setup

`uv run python experiments/0003_cem_value_weights_vs_greedy/run.py` —
maximise target of the `_value_fitting` framework: 3 CEM generations ×
6 members × 60-game evaluations, σ=0.3 around the hand weights, common
random numbers within a generation, rotated across generations.

## Results

From `runs/0003_cem_value_weights_vs_greedy/2026-06-12T225250Z` (`weights.json`):

- The search stayed near the hand weights (its starting point) with mostly
  small moves — more `n_dev` (+2.09 vs +1.50), more `best_spot`, less
  `progress` and much less `race` (+0.10 vs +0.80: against greedy the
  endgame urgency term apparently pays less than mid-game economy).
- Deployment bench: **80.8% vs greedy** against the hand weights' 77.8% —
  the search beat its objective's baseline.
- Head-to-head vs lookahead(hand-tuned): **43.3% ± 2.8%** — gate fails.

## Decision

The maximise path works and is honest about its scope: optimizing the match
win rate against a *fixed* opponent yields a greedy-specialist — better
against greedy, worse against everything that isn't greedy (exp 0002's
predict target showed the same one-sidedness from the data side). The hand
weights remain the shipped defaults because they were tuned against the
agent being gated. To beat them with this framework, the objective must be
the gate itself (maximise vs lookahead(heuristic) directly) or a pool of
opponents — both are one-line CONFIG changes for a future experiment, at a
larger eval budget (the gap to detect is a few points, 60-game evaluations
swing ±6).
