"""Settings parse the environment: the bits manual parsing got wrong (csv lists,
typed coercion, the unprefixed deployment vars)."""

import pytest
from settlrl_app.config import Settings


def test_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "SETTLRL_APP_STATE_DIR",
        "SETTLRL_APP_MAX_ACTIVE",
        "SETTLRL_APP_ADMIN_EMAILS",
        "PORT",
        "RELOAD",
        "ROOT_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.state_dir is None and s.admin_emails == frozenset()
    assert s.max_active == 16 and s.port == 8000 and s.reload is True


def test_env_is_typed_and_csv_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SETTLRL_APP_MAX_ACTIVE", "8")
    monkeypatch.setenv("SETTLRL_APP_TURN_TIMEOUT_S", "30")
    monkeypatch.setenv("SETTLRL_APP_ADMIN_EMAILS", "a@x.com, b@y.com ")
    monkeypatch.setenv("PORT", "9000")  # unprefixed deployment var
    monkeypatch.setenv("RELOAD", "0")
    s = Settings()
    assert s.max_active == 8 and s.turn_timeout_s == 30.0  # coerced int / float
    assert s.admin_emails == frozenset({"a@x.com", "b@y.com"})  # trimmed, split
    assert s.port == 9000 and s.reload is False  # bool / int from the bare names
