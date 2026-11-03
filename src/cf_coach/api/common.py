"""Pieces every route file uses: the error class, the DB session, the handle path type.

Kept apart from app.py so route files can import them without importing the app
(app.py imports the route files, so the other way round would be a circular import).
"""

import os
from typing import Annotated, Iterator

import redis
from dotenv import load_dotenv
from fastapi import Depends, Path
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cf_coach.api.contract import HANDLE_PATTERN, ErrorCode, ErrorOut
from cf_coach.db.models import User
from cf_coach.db.session import get_engine


class ApiError(Exception):
    """Raise it in a route to answer with the one error shape (ErrorOut).

        raise ApiError(404, "not_found", "TitanCoder1258 has not been synced yet")
    -> HTTP 404  {"error": {"code": "not_found", "message": "...", "retry_after_s": null}}
    """

    def __init__(self, status: int, code: ErrorCode, message: str,
                 retry_after_s: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s


def get_session() -> Iterator[Session]:
    """A DB session for one request, closed when the response is sent.

    A FastAPI "dependency" (routes use it through DbSession below). The code before
    `yield` runs before the route, the code after it (the end of the `with`, which closes
    the session) runs after. Tests swap it for a fake with app.dependency_overrides.
    """
    with Session(get_engine()) as session:
        yield session


# A route asks for a session by naming this as a parameter's type:
#     def get_profile(handle: Handle, session: DbSession): ...
DbSession = Annotated[Session, Depends(get_session)]

_redis = None


def get_redis() -> redis.Redis:
    """One Redis client per process, made on first use (it keeps its own connection pool).
    decode_responses=True: values come back as str, not bytes."""
    global _redis
    if _redis is None:
        load_dotenv()
        _redis = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    return _redis


RedisConn = Annotated[redis.Redis, Depends(get_redis)]


def find_user(session: Session, handle: str) -> User | None:
    """CF handles are case-insensitive: "tourist" finds the row stored as "tourist" or
    "Tourist". The row keeps CF's own spelling (sync_user stores what user.info says)."""
    return session.scalar(select(User).where(func.lower(User.handle) == handle.lower()))


def synced_user(session: Session, handle: str) -> User:
    """The user's row, or a 404 if this handle was never synced."""
    user = find_user(session, handle)
    if user is None or user.synced_at is None:
        raise ApiError(404, "not_found", f"{handle} has not been synced yet")
    return user


# /api/users/{handle}/...: checked before the route runs; a bad handle is a 422
Handle = Annotated[str, Path(min_length=3, max_length=24, pattern=HANDLE_PATTERN)]

# Added to each route's `responses=` so /openapi.json (and the generated TypeScript
# types) show the error shape too, not only the success shape.
ERRORS = {404: {"model": ErrorOut}, 422: {"model": ErrorOut}, 500: {"model": ErrorOut}}
