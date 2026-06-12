# experiments

The lab notebook: one numbered directory per experiment *framework* — a
class of related experiments sharing machinery — committed; everything a run
*produces* lands in the git-ignored `runs/` at the repo root.

## Contract

- `NNNN_slug/` — one framework. `run.py` exposes its named variants
  (`run.py [variant]`); helpers specific to the framework live beside it in
  the same directory. A framework accumulates configs and runs over time —
  the *conclusions* in its report are what stays immutable, and `runs/`
  collects many logs per framework.
- Each run is deterministic given its config (seeds included); it writes only
  under its run directory, which `_lib.start_run` creates with a manifest
  (git commit + uncommitted-diff digest, the merged config, environment).
- `NNNN_slug/report.md` — hypothesis → setup → results → decision, one
  section per concluded variant, citing the `runs/` directories the numbers
  came from.
- `JOURNAL.md` — append-only index: one verdict line per concluded finding.
- Strength claims gate through `catan-agents bench` (`--json` emits the
  machine-readable verdict) or an in-run match with the threshold asserted in
  code — the script decides pass/fail, not a reading of it.

## Workflow

```
uv run python experiments/new.py "<title>"            # scaffold a framework
uv run python experiments/NNNN_slug/run.py [variant]  # run a config of it
```

`0001_bench_smoke` is the minimal worked example;
`0002_linear_value_fitting` a multi-variant framework.
