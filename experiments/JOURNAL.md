# Experiment journal

One line per concluded experiment — number, verdict, the load-bearing fact.
Full evidence lives in each experiment's `report.md`; raw outputs under
`runs/` (git-ignored, regenerable from the manifest's commit + config).

- 0001_bench_smoke — pass: greedy beats random 83.1% ± 4.7% (n=65, 2p);
  infrastructure worked example.
- 0002_sklearn_value_weights_vs_greedy — framework adopted, weights not:
  outcome-fit logistic weights reach hand-tuned level vs greedy (75.2% vs
  77.8%) but lose head-to-head (42.7%, n=314); held-out AUC flat (0.83–0.84)
  while match probes span 53–78% — select by matches, never fit metrics.
