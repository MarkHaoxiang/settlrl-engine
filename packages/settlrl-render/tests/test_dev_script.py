"""Smoke test for the dev launcher (``packages/settlrl-render/dev.sh``).

The script chains real CLI entry points and HTTP routes to boot the stack with a
seeded admin + registered bot service. Actually running all three (JAX, node) is
too heavy for CI, so this pins the script's *contract* instead: valid bash, and
every command / endpoint it leans on still exists — so a rename can't silently
rot the launcher.
"""

import subprocess
from pathlib import Path

import tomllib
from fastapi import FastAPI
from fastapi.testclient import TestClient
from settlrl_render.bots.bot_service import create_bot_app
from settlrl_render.game.games import GameRegistry
from settlrl_render.server import create_app

PKG = Path(__file__).resolve().parents[1]
SCRIPT = PKG / "dev.sh"


def _paths(app: FastAPI) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_dev_script_is_valid_executable_bash() -> None:
    assert SCRIPT.exists(), "dev.sh is missing"
    assert SCRIPT.stat().st_mode & 0o111, "dev.sh is not executable"
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_dev_script_uses_real_cli_entry_points() -> None:
    scripts = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["scripts"]
    text = SCRIPT.read_text()
    for name in ("settlrl-render", "settlrl-render-bot"):
        assert name in scripts, f"{name} is no longer a project script"
        assert name in text, f"dev.sh no longer launches {name}"


def test_dev_script_endpoints_still_exist() -> None:
    text = SCRIPT.read_text()
    app = create_app(GameRegistry(), admin_emails=frozenset({"dev@example.com"}))
    with TestClient(app):  # lifespan creates the (in-memory) tables and tasks
        served = _paths(app)
    for path in (
        "/api/bots",
        "/api/auth/register",
        "/api/auth/login",
        "/api/admin/bot-providers",
    ):
        assert path in text, f"dev.sh no longer calls {path}"
        assert path in served, f"the server no longer serves {path}"
    assert "/catalog" in text and "/catalog" in _paths(create_bot_app())


def test_dev_script_seeds_an_admin() -> None:
    # Registering the bot service only works if the seeded account is admin,
    # which depends on its email being passed as a configured admin email.
    text = SCRIPT.read_text()
    assert "SETTLRL_RENDER_ADMIN_EMAILS" in text
    assert "$ADMIN_EMAIL" in text or "${ADMIN_EMAIL}" in text
