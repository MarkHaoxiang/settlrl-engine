# 0004 — AlphaZero (2-player)

Status: open (loop built; proof-of-concept only, no strength run)

## Hypothesis

A value+policy net trained by AlphaZero self-play — the re-determinizing search
as its own teacher — beats `lookahead(heuristic)` at 2p, lifting the leaf the
search ladder is stuck against (the settlrl-learn Stage-1 gate).

## Setup

`run.py [default|smoke]`. The loop lives in `settlrl_learn.training`
(composable); `run.py` only composes it with a config, per-iteration logging,
and the gate verdict. Each iteration:

1. **self-play** — net-guided re-determinizing search (`value_scale=2`); record
   each move's true-board features, the search's improved policy (target), and
   the eventual win/loss (value);
2. **buffer** — a flashbax item buffer (recent positions);
3. **train** — optax adamw on policy cross-entropy + value logistic;
4. **arena** (periodic) — seat-swapped win rate vs `lookahead(heuristic)`.

2-player only: belief is near-exact, so the multiplayer paranoid-frame problem
never arises. Stack: mctx (search), optax, flashbax, the settlrl-learn net.

## Results

Smoke (1 iteration, 8 samples, 4 sims) runs the whole loop end-to-end and
records a verdict. Component PoCs: self-play yields valid policy/value targets
(395 samples, policy rows sum to 1, balanced wins); training drops the loss
7.17 → 1.74 over 100 steps. No strength run yet — PoC scope.

## Decision

Loop adopted; not yet run at scale. Next: a real run against the gate, and
expose the search's root Q for Canopy's `(1−α)z + α·q` value blend (dice
variance).
