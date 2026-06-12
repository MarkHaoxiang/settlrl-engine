"""Run bookkeeping for experiment scripts (stdlib only).

``start_run`` creates a fresh directory under ``runs/<experiment>/<stamp>/``
and writes a manifest pinning what reproduction needs: the git commit (plus a
digest of any uncommitted diff), the config, and the environment. Metrics
stream to ``metrics.jsonl``; ``finish`` records the verdict in ``result.json``.
Nothing under ``runs/`` is tracked by git — the committed ``report.md`` cites
the run directory it drew from.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def _gpu() -> str | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or None


@dataclass
class Run:
    """One run's output directory and its append-only metrics stream."""

    dir: Path
    _step: int = 0

    def log(self, **metrics: Any) -> None:
        """Append one record to ``metrics.jsonl`` (auto ``step``, wall ``time``)."""
        record = {"step": self._step, "time": time.time(), **metrics}
        with (self.dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._step += 1

    def save_json(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2) + "\n")

    def finish(self, verdict: str, **summary: Any) -> None:
        """Record the run's verdict (``"pass"`` / ``"fail"`` / ...) and summary."""
        self.save_json("result.json", {"verdict": verdict, **summary})
        print(f"{self.dir.relative_to(_ROOT)}: {verdict} {summary}")


def start_run(experiment: Path, config: dict[str, Any]) -> Run:
    """Create ``runs/<experiment-name>/<UTC-stamp>/`` and write its manifest."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    out = _ROOT / "runs" / experiment.resolve().name / stamp
    out.mkdir(parents=True)
    diff = _git("diff", "HEAD")
    manifest = {
        "experiment": experiment.resolve().name,
        "started": stamp,
        "argv": sys.argv,
        "config": config,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_diff_sha256": sha256(diff.encode()).hexdigest() if diff else None,
        "python": platform.python_version(),
        "jax": metadata.version("jax"),
        "gpu": _gpu(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"run dir: {out.relative_to(_ROOT)}")
    return Run(out)
