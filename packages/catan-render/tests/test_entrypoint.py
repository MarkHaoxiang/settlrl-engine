"""The startup create-key warning (catan_render.main's operator alert)."""

import pytest
from catan_render import _create_key_warning


def test_dev_run_stays_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELOAD", "1")
    monkeypatch.delenv("CATAN_RENDER_CREATE_KEY", raising=False)
    assert _create_key_warning() is None


def test_production_without_key_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELOAD", "0")
    monkeypatch.delenv("CATAN_RENDER_CREATE_KEY", raising=False)
    assert "not set" in (_create_key_warning() or "")


def test_production_with_key_stays_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELOAD", "0")
    monkeypatch.setenv("CATAN_RENDER_CREATE_KEY", "secret")
    assert _create_key_warning() is None


def test_empty_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELOAD", "0")
    monkeypatch.setenv("CATAN_RENDER_CREATE_KEY", "")
    assert _create_key_warning() is not None
