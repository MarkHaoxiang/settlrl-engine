"""User accounts and authentication — an optional layer over anonymous play.

Accounts give a player a persistent identity across games and devices, and mark
some users as **admins** (who may register external bot providers — see
:mod:`settlrl_render.providers`). It deliberately reuses the app's existing
security model rather than bolting on a JWT/ORM stack: the server is
single-process and already mints and checks opaque per-seat bearer tokens, so a
login here mints the same kind of token, checked server-side and persisted in
SQLite beside the games. Password hashing is stdlib :func:`hashlib.scrypt`;
login speaks FastAPI's OAuth2 password flow, so it slots into the interactive
docs and standard clients. (Swapping in signed JWTs later is localized to
:meth:`UserStore.mint_session` / :meth:`UserStore.user_for_token`.)

Everything is additive: with no token presented, every existing route behaves
exactly as before.
"""

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

# scrypt work factors (RFC 7914 interactive-login range). Stored in each hash so
# a later bump still verifies old passwords.
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_DK_LEN = 32

_TOKEN_BYTES = 24
# A login token lasts this long; presenting it past expiry is a fresh login.
DEFAULT_SESSION_TTL_S = 30 * 24 * 3600.0

_MIN_PASSWORD = 8
_MAX_PASSWORD = 128


def hash_password(password: str) -> str:
    """A self-describing scrypt hash: ``scrypt$N$r$p$salt_hex$dk_hex``."""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DK_LEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check ``password`` against a :func:`hash_password` digest (constant-time)."""
    try:
        scheme, n, r, p, salt_hex, dk_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(dk_hex) // 2,
        )
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(dk.hex(), dk_hex)


@dataclass(frozen=True)
class User:
    id: int
    email: str
    is_admin: bool


class AuthError(Exception):
    """A account-operation failure the routes turn into a 4xx (e.g. a duplicate
    email)."""


class UserStore:
    """Accounts and login sessions in SQLite (a file under the state dir, or an
    in-memory db when none is configured). Thread-safe: one connection guarded
    by a lock, since FastAPI runs the sync routes in a threadpool."""

    def __init__(self, path: str | None = None) -> None:
        # check_same_thread=False: the threadpool hands the connection between
        # workers; the lock below serialises every access.
        self._conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    pw_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at REAL NOT NULL
                );
                """
            )

    def create_user(self, email: str, password: str, is_admin: bool = False) -> User:
        email = email.strip().lower()
        with self._lock, self._conn:
            try:
                cur = self._conn.execute(
                    "INSERT INTO users (email, pw_hash, is_admin, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (email, hash_password(password), int(is_admin), time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthError("email already registered") from exc
            return User(id=cast(int, cur.lastrowid), email=email, is_admin=is_admin)

    def authenticate(self, email: str, password: str) -> User | None:
        """The matching user when the password checks out, else None."""
        email = email.strip().lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT id, pw_hash, is_admin FROM users WHERE email = ?", (email,)
            ).fetchone()
        if row is None or not verify_password(password, row[1]):
            return None
        return User(id=row[0], email=email, is_admin=bool(row[2]))

    def set_admin(self, email: str, is_admin: bool = True) -> User | None:
        """Promote / demote a user by email (None if there is no such user)."""
        email = email.strip().lower()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE users SET is_admin = ? WHERE email = ?", (int(is_admin), email)
            )
            if cur.rowcount == 0:
                return None
            row = self._conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
        return User(id=row[0], email=email, is_admin=is_admin)

    def mint_session(self, user_id: int, ttl: float) -> str:
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, time.time() + ttl),
            )
        return token

    def user_for_token(self, token: str) -> User | None:
        """The user a live session token belongs to (None if unknown/expired)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT u.id, u.email, u.is_admin, s.expires_at"
                " FROM sessions s JOIN users u ON u.id = s.user_id"
                " WHERE s.token = ?",
                (token,),
            ).fetchone()
        if row is None or row[3] < time.time():
            return None
        return User(id=row[0], email=row[1], is_admin=bool(row[2]))

    def revoke(self, token: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# -- wire models --------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=_MIN_PASSWORD, max_length=_MAX_PASSWORD)


class UserModel(BaseModel):
    id: int
    email: str
    is_admin: bool

    @classmethod
    def of(cls, user: User) -> "UserModel":
        return cls(id=user.id, email=user.email, is_admin=user.is_admin)


class TokenModel(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserModel


class Auth:
    """The account system bound to one :class:`UserStore`.

    Exposes the ``/api/auth`` router plus FastAPI dependencies other routes
    reuse: :attr:`optional_user` (None when unauthenticated), :attr:`current_user`
    (401 otherwise), and :attr:`admin_user` (403 unless an admin). Emails listed
    in ``admin_emails`` are granted admin on register and on every login, so an
    operator can anoint an admin from configuration alone.
    """

    def __init__(
        self,
        store: UserStore,
        admin_emails: frozenset[str] = frozenset(),
        session_ttl: float = DEFAULT_SESSION_TTL_S,
    ) -> None:
        self.store = store
        self.admin_emails = frozenset(
            e.strip().lower() for e in admin_emails if e.strip()
        )
        self.session_ttl = session_ttl
        scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

        # Annotated form (FastAPI's preferred style); these resolve at definition
        # time because this module does not use `from __future__ import
        # annotations`, so the closure-local dependencies are real objects rather
        # than unresolvable strings.
        def optional_user(
            token: Annotated[str | None, Depends(scheme)],
        ) -> User | None:
            return store.user_for_token(token) if token else None

        def current_user(user: Annotated[User | None, Depends(optional_user)]) -> User:
            if user is None:
                raise HTTPException(
                    status_code=401,
                    detail="not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user

        def admin_user(user: Annotated[User, Depends(current_user)]) -> User:
            if not user.is_admin:
                raise HTTPException(status_code=403, detail="admin only")
            return user

        self.optional_user = optional_user
        self.current_user = current_user
        self.admin_user = admin_user
        self._scheme = scheme
        self.router = self._build_router()

    def _build_router(self) -> APIRouter:
        router = APIRouter(prefix="/api/auth", tags=["auth"])

        @router.post("/register", status_code=201)
        def register(req: RegisterRequest) -> UserModel:
            try:
                user = self.store.create_user(
                    req.email, req.password, is_admin=self._is_admin_email(req.email)
                )
            except AuthError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return UserModel.of(user)

        @router.post("/login")
        def login(
            form: Annotated[OAuth2PasswordRequestForm, Depends()],
        ) -> TokenModel:
            user = self.store.authenticate(form.username, form.password)
            if user is None:
                raise HTTPException(
                    status_code=401,
                    detail="incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # Configured admin emails are kept in sync on each login.
            if self._is_admin_email(user.email) and not user.is_admin:
                self.store.set_admin(user.email, True)
                user = User(id=user.id, email=user.email, is_admin=True)
            token = self.store.mint_session(user.id, self.session_ttl)
            return TokenModel(access_token=token, user=UserModel.of(user))

        @router.post("/logout", status_code=204)
        def logout(token: Annotated[str | None, Depends(self._scheme)]) -> None:
            if token:
                self.store.revoke(token)

        @router.get("/me")
        def me(user: Annotated[User, Depends(self.current_user)]) -> UserModel:
            return UserModel.of(user)

        return router

    def _is_admin_email(self, email: str) -> bool:
        return email.strip().lower() in self.admin_emails
