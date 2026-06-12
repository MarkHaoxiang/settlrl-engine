# experiments

The lab notebook: one numbered directory per experiment, committed. Everything
a run *produces* lands in the git-ignored `runs/` at the repo root.

## Contract

- `NNNN_slug/run.py` — the experiment. Deterministic given its `CONFIG` dict
  (seeds live there); writes only under its run directory, which
  `_lib.start_run` creates with a manifest (git commit + uncommitted-diff
  digest, config, environment). A committed experiment is immutable — a
  follow-up gets a new number, so reports and docs can cite `NNNN` forever.
- `NNNN_slug/report.md` — hypothesis → setup → results → decision, written
  after the run and citing the `runs/` directory the numbers came from.
- `JOURNAL.md` — append-only index: one verdict line per concluded experiment.
- Strength claims gate through `catan-agents bench` (`--json` emits the
  machine-readable verdict): save the output in the run dir and assert the
  threshold in `run.py` — the script decides pass/fail, not a reading of it.

## Workflow

```
uv run python experiments/new.py "<title>"    # scaffold the next number
uv run python experiments/NNNN_slug/run.py    # run it (re-run any time)
```

`0001_bench_smoke` is the worked example.
