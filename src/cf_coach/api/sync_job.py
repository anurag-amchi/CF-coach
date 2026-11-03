"""Syncing a handle from the web: start it, run it after the response, report its state.

POST /api/users/{h}/sync  start_sync() takes the handle's lock and writes "queued";
                          FastAPI runs run_sync() after the 202 reply has been sent
GET  /api/users/{h}/sync  sync_status(): the state in Redis, else what Postgres knows

Two Redis keys per handle, lower-cased so "Tourist" and "tourist" share them:
  sync:lock:<h>    set with NX ("only if it doesn't exist"): only the request that creates
                   it starts a sync. Deleted when the job ends; expires after LOCK_TTL_S
                   if the process dies mid-sync.
  sync:status:<h>  JSON {state, new_submissions, error}. "queued"/"running" expire like
                   the lock, so a crash can't leave "running" forever; "done"/"error"
                   last a day.

For now the job runs inside the API process (FastAPI BackgroundTasks). Slice 8 moves it
to a separate worker with a queue, plus the daily problemset refresh.
Design reasons: docs/decisions.md #59.
"""

import json
import logging
from datetime import datetime, timezone

import redis
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cf_coach.analytics.tag_profile import refresh_tag_skills
from cf_coach.api.common import find_user
from cf_coach.api.contract import SyncStatusOut
from cf_coach.db.models import Problem
from cf_coach.db.session import get_engine
from cf_coach.sync.cf_client import CFError
from cf_coach.sync.sync import sync_problemset, sync_user

log = logging.getLogger("cf_coach.api")

LOCK_TTL_S = 15 * 60          # a first sync of a 100k-submission handle is ~200 pages x 2 s
STATUS_TTL_S = 24 * 3600


def lock_key(handle: str) -> str:
    return f"sync:lock:{handle.lower()}"


def status_key(handle: str) -> str:
    return f"sync:status:{handle.lower()}"


def write_status(r: redis.Redis, handle: str, state: str,
                 new_submissions: int | None = None, error: str | None = None) -> None:
    ttl = LOCK_TTL_S if state in ("queued", "running") else STATUS_TTL_S
    value = {"state": state, "new_submissions": new_submissions, "error": error}
    r.set(status_key(handle), json.dumps(value), ex=ttl)


def start_sync(r: redis.Redis, handle: str) -> bool:
    """True = this request got the lock and must start the job.
    False = a sync of this handle is already queued or running; don't start another."""
    if not r.set(lock_key(handle), "1", nx=True, ex=LOCK_TTL_S):
        return False
    write_status(r, handle, "queued")
    return True


def sync_steps(handle: str) -> int:
    """The real work, the same steps as the CLI: problemset (only on an empty database),
    the user's new submissions, then their tag_skills rows. Returns the new submissions.
    Tests replace this function, so they need neither Codeforces nor Postgres."""
    with Session(get_engine()) as session:
        have_problems = session.scalar(select(func.count()).select_from(Problem)) > 0
    if not have_problems:
        sync_problemset()
    summary = sync_user(handle)
    refresh_tag_skills(summary["handle"], datetime.now(timezone.utc))   # CF's spelling
    return summary["fetched"]


def cf_error_message(handle: str, e: CFError) -> str:
    """CFError text is for logs ("user.info: HTTP 400 handles: User with handle x not
    found"); the user gets one plain sentence."""
    if "not found" in str(e):
        return f"Codeforces has no user called {handle}"
    return "Codeforces isn't answering right now; try again in a few minutes"


def run_sync(r: redis.Redis, handle: str) -> None:
    """The background job. It runs after the response is sent, when the request's DB
    session is already closed, so every step opens its own session."""
    write_status(r, handle, "running")
    try:
        new = sync_steps(handle)
        write_status(r, handle, "done", new_submissions=new)
    except CFError as e:
        log.warning("sync %s: %s", handle, e)
        write_status(r, handle, "error", error=cf_error_message(handle, e))
    except Exception:
        log.exception("sync %s crashed", handle)
        write_status(r, handle, "error", error="the sync failed on our side")
    finally:
        r.delete(lock_key(handle))          # free the handle even after a crash


def sync_status(session: Session, r: redis.Redis, handle: str) -> SyncStatusOut:
    """Redis knows about the current or latest sync; Postgres knows when the last one
    finished. No Redis entry (expired, or synced from the CLI) -> "done" if Postgres has
    a sync time, else "never"."""
    user = find_user(session, handle)
    raw = r.get(status_key(handle))
    saved = json.loads(raw) if raw else {}
    synced_at = user.synced_at if user else None
    return SyncStatusOut(
        handle=user.handle if user else handle,
        state=saved.get("state") or ("done" if synced_at else "never"),
        synced_at=synced_at,
        new_submissions=saved.get("new_submissions"),
        error=saved.get("error"),
    )
