"""The server's runtime configuration, read once at startup.

pydantic-settings does the coercion, validation, and precedence: a real
environment variable wins, else a key from the local ``.secrets/.env`` file
(git-ignored — for values like admin emails kept out of the committed compose
file), else the default declared here. ``SETTLRL_APP_*`` are this app's own
knobs; the unprefixed ``ROOT_PATH`` / ``HOST`` / ``PORT`` / ``RELOAD`` follow
the usual deployment conventions.
"""

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SETTLRL_APP_", env_file=".secrets/.env", extra="ignore"
    )

    # Persistence + runtime knobs (SETTLRL_APP_*).
    state_dir: str | None = None  # a dir to journal games into (None = in-memory)
    turn_timeout_s: float = 0.0  # auto-play an idle human turn after this (0 = off)
    max_active: int = 16  # concurrent running games before new creators queue
    user_db: str | None = None  # explicit SQLite path (else settlrl.db under state_dir)
    # Comma-separated (NoDecode: skip the JSON pre-parse so the splitter sees it).
    admin_emails: Annotated[frozenset[str], NoDecode] = frozenset()

    # Deployment vars (shared conventions, unprefixed).
    root_path: str = Field(default="", validation_alias="ROOT_PATH")  # proxy prefix
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    reload: bool = Field(default=True, validation_alias="RELOAD")  # dev; 0 in prod

    @field_validator("admin_emails", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept a comma-separated env string (the JSON default can't)."""
        if isinstance(value, str):
            return [e.strip() for e in value.split(",") if e.strip()]
        return value
