"""Smoke test for the package entry point."""

from __future__ import annotations

import pytest

from catan_engine import main


def test_main_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "catan-engine" in capsys.readouterr().out
