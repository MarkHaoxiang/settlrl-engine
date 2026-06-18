"""The single async storage layer.

One SQLAlchemy 2.0 engine (async, on aiosqlite) backs everything that
persists: user accounts and login tokens (fastapi-users), and the game journals
(:mod:`settlrl_app.storage.store`). ``create_app`` builds one :class:`Database` and
wires it everywhere; :meth:`Database.init` creates the tables at startup.

A file under the state dir persists across restarts; with no state dir the db is
in-memory (tests / stateless runs) — kept on a single shared connection so every
session sees the same data.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

# Import fastapi_users.db before its sqlalchemy adapter: the two import each
# other circularly, and entering via the adapter first (which can happen when
# this module is imported before any fastapi_users.* module) leaves
# fastapi_users.db half-initialised — it silently drops SQLAlchemyUserDatabase.
import fastapi_users.db  # noqa: F401
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyBaseAccessTokenTableUUID
from sqlalchemy import JSON, Float, ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from settlrl_app.ratings import INITIAL_MU, INITIAL_SIGMA


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    """A user account (fastapi-users). Admin == ``is_superuser``."""


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    """A login token, for the fastapi-users database auth strategy."""


class GameRow(Base):
    """A persisted game's immutable header (its setup).

    ``finished_at`` is null while the game is live and a wall-clock timestamp once
    it ends — finished rows are kept as a replayable history (live rows are
    restored into the registry on boot; finished ones are not). ``winner`` is the
    winning seat; ``owners`` maps an account's user-id to the seats it held (so a
    user's history needs no event scan). All three are null while unfinished."""

    __tablename__ = "game"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    header: Mapped[dict[str, Any]] = mapped_column(JSON)
    finished_at: Mapped[float | None] = mapped_column(Float, default=None, index=True)
    winner: Mapped[int | None] = mapped_column(Integer, default=None)
    owners: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)


class Rating(Base):
    """One Elo rating: a subject's standing at a given player count.

    Accounts and bots share the leaderboard; ``subject_kind`` (``"account"`` /
    ``"bot"``) tells them apart and ``subject_id`` is the account's user-id or
    the bot's name. Ratings are bucketed by ``n_players`` — a separate ladder per
    game size (a 2p and a 4p rating for the same subject are independent rows).
    Skill is the openskill ``(mu, sigma)`` pair; the displayed number is derived
    from it. ``name`` is the cached display label (a bot's name, an account's
    handle)."""

    __tablename__ = "rating"

    subject_kind: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(String, primary_key=True)
    n_players: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    mu: Mapped[float] = mapped_column(Float, default=INITIAL_MU)
    sigma: Mapped[float] = mapped_column(Float, default=INITIAL_SIGMA)
    games: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[float] = mapped_column(Float, default=0.0)


class GameLog(Base):
    """A game's full ordered event log (moves, seat claims, chat) as one JSON
    document — one row per game, rewritten in place as the game advances. Kept
    out of :class:`GameRow` so listing/restoring headers doesn't drag the log."""

    __tablename__ = "game_log"

    game_id: Mapped[str] = mapped_column(
        String, ForeignKey("game.id", ondelete="CASCADE"), primary_key=True
    )
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSON)


class Database:
    """The app's async engine and session factory (one per app)."""

    def __init__(self, db_path: str | None) -> None:
        if db_path:
            self.engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        else:
            # In-memory: one shared connection so every session sees the same db.
            self.engine = create_async_engine(
                "sqlite+aiosqlite://",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        """Create the tables (idempotent)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def session(self) -> AsyncIterator[AsyncSession]:
        """A request-scoped session (FastAPI dependency)."""
        async with self.sessionmaker() as session:
            yield session
