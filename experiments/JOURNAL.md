# Experiment journal

One line per concluded experiment — number, verdict, the load-bearing fact.
Full evidence lives in each experiment's `report.md`; raw outputs under
`runs/` (git-ignored, regenerable from the manifest's commit + config).

- 0001_bench_smoke — pass: greedy beats random 83.1% ± 4.7% (n=65, 2p);
  infrastructure worked example.
- 0002_linear_value_fitting — framework adopted, weights not (both targets):
  predict reaches hand level vs greedy (75.2% vs 77.8%), maximise beats it
  (80.8%), yet both lose head-to-head (~43%) — fixed-opponent optimization
  breeds specialists. Held-out AUC flat while match probes span 53–78%:
  select by matches, never fit metrics.
- 0002_linear_value_fitting/self_play — pass: 3-round champion-ladder CEM
  beat the hand-tuned weights 56.1% head-to-head (n=310, lower 2σ 50.5%) and
  80.9% vs greedy (hand: 77.8%) — fixed-opponent specialists fixed by making
  the opponent evolve; adoption of the weights deferred (leaf cascade).
- 0002_linear_value_fitting/self_play at 4p — the 2p edge does not transfer:
  27.0% ± 2.8% vs three hand lookaheads (chance 25%), 64.7% vs greedy tables
  (hand: 68.1%); champion reproduced bit-identically from config (framework
  determinism verified). Adoption now 2p-conditional or needs a mixed-count
  arena.
- 0002_linear_value_fitting/self_play_4p — near miss: the 4p-arena champion
  reads 30.4% ± 3.2% vs three hand lookaheads (chance 25%, lower 2σ 24.0%,
  n=230) and 69.3% vs greedy tables; 4p tuned slot stays hand-tuned pending
  an n≈600 confirmation.
- 0003_neural_board_architectures — pass (framework + first sweep, 12k
  positions): a jraph GNN over the *raw* board nearly matches the hand-tuned
  feature MLP — heuristic R² 0.978 vs 0.996, win AUC 0.825 vs 0.834 — while a
  structure-blind flat MLP on the same inputs is ≈chance (R² 0.54, AUC 0.52)
  and DeepSet sits between. Structure is what makes raw board features usable;
  a learnable leaf is within reach (settlrl-learn Stage 1 seam). Not yet
  promoted: close the win gap, then gate lookahead(gnn) through bench.
