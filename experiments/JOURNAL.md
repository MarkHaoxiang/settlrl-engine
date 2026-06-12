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
