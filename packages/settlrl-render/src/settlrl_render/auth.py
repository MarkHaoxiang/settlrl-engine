"""User accounts and authentication, via fastapi-users on the shared async DB.

Accounts are optional: anonymous play (claim a seat, get a per-seat token) is
unchanged. Signing in gives a persistent identity (seats follow the account) and
marks some users as **admins** — fastapi-users *superusers* — who manage the bot
services. Login uses the OAuth2 password flow; tokens live in the ``access_token``
table (fastapi-users :class:`DatabaseStrategy`), so logout truly revokes and the
account system shares the one storage layer (:mod:`settlrl_render.db`).

Emails in ``admin_emails`` are promoted to superuser on register and on every
login, so an operator can anoint an admin from configuration alone.
"""

import secrets
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi_users import (
    BaseUserManager,
    FastAPIUsers,
    InvalidPasswordException,
    UUIDIDMixin,
    schemas,
)
from fastapi_users.authentication import AuthenticationBackend, BearerTransport
from fastapi_users.authentication.strategy import AccessTokenDatabase, DatabaseStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyAccessTokenDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from .db import AccessToken, Database, User

# A login token lasts this long; presenting it past expiry is a fresh login.
ACCESS_TOKEN_TTL_S = 30 * 24 * 3600


class UserRead(schemas.BaseUser[uuid.UUID]):
    pass


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass


class Auth:
    """The account system bound to one :class:`Database`.

    Exposes the ``/api/auth`` + ``/api/users`` routers and the FastAPI
    dependencies other routes reuse: :attr:`optional_user` (None when
    unauthenticated), :attr:`current_user` (401 otherwise), and
    :attr:`admin_user` (403 unless a superuser).
    """

    def __init__(
        self, db: Database, admin_emails: frozenset[str] = frozenset()
    ) -> None:
        self.db = db
        admins = frozenset(e.strip().lower() for e in admin_emails if e.strip())
        # reset/verify flows aren't exposed, but fastapi-users requires the
        # secrets to exist; an ephemeral one is fine (no token is ever issued).
        token_secret = secrets.token_urlsafe(32)

        async def get_user_db(
            session: Annotated[AsyncSession, Depends(db.session)],
        ) -> AsyncIterator[SQLAlchemyUserDatabase[User, uuid.UUID]]:
            yield SQLAlchemyUserDatabase(session, User)

        async def get_access_token_db(
            session: Annotated[AsyncSession, Depends(db.session)],
        ) -> AsyncIterator[SQLAlchemyAccessTokenDatabase[AccessToken]]:
            yield SQLAlchemyAccessTokenDatabase(session, AccessToken)

        class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
            reset_password_token_secret = token_secret
            verification_token_secret = token_secret

            async def validate_password(
                self, password: str, user: schemas.BaseUserCreate | User
            ) -> None:
                if len(password) < 8:
                    raise InvalidPasswordException(
                        "password must be at least 8 characters"
                    )

            async def _sync_admin(self, user: User) -> None:
                if user.email.lower() in admins and not user.is_superuser:
                    await self.user_db.update(user, {"is_superuser": True})

            async def on_after_register(
                self, user: User, request: Request | None = None
            ) -> None:
                await self._sync_admin(user)

            async def on_after_login(
                self,
                user: User,
                request: Request | None = None,
                response: Response | None = None,
            ) -> None:
                await self._sync_admin(user)

        async def get_user_manager(
            user_db: Annotated[
                SQLAlchemyUserDatabase[User, uuid.UUID], Depends(get_user_db)
            ],
        ) -> AsyncIterator[UserManager]:
            yield UserManager(user_db)

        def get_strategy(
            access_token_db: Annotated[
                AccessTokenDatabase[AccessToken], Depends(get_access_token_db)
            ],
        ) -> DatabaseStrategy[User, uuid.UUID, AccessToken]:
            return DatabaseStrategy(
                access_token_db, lifetime_seconds=ACCESS_TOKEN_TTL_S
            )

        backend = AuthenticationBackend(
            name="db",
            transport=BearerTransport(tokenUrl="api/auth/login"),
            get_strategy=get_strategy,
        )
        fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [backend])

        self.current_user = fastapi_users.current_user(active=True)
        self.optional_user = fastapi_users.current_user(active=True, optional=True)
        self.admin_user = fastapi_users.current_user(active=True, superuser=True)

        router = APIRouter()
        router.include_router(
            fastapi_users.get_auth_router(backend), prefix="/api/auth", tags=["auth"]
        )
        router.include_router(
            fastapi_users.get_register_router(UserRead, UserCreate),
            prefix="/api/auth",
            tags=["auth"],
        )
        router.include_router(
            fastapi_users.get_users_router(UserRead, UserUpdate),
            prefix="/api/users",
            tags=["users"],
        )
        self.router = router
