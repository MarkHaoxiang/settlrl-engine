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
from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    """A user account (fastapi-users). Admin == ``is_superuser``."""


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    """A login token, for the fastapi-users database auth strategy."""


class GameRow(Base):
    """A persisted game's immutable header (its setup)."""

    __tablename__ = "game"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    header: Mapped[dict[str, Any]] = mapped_column(JSON)


class GameEvent(Base):
    """One ordered event in a game's journal (a move, seat claim, or chat line)."""

    __tablename__ = "game_event"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(
        String, ForeignKey("game.id", ondelete="CASCADE"), index=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


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
