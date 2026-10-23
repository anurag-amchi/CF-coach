"""Sync Codeforces data into Postgres.

sync_problemset()  - all problems with ratings and tags. Shared by every user.
sync_user(handle)  - a user's profile and their NEW submissions only (incremental).

Both are idempotent: running them twice leaves the same rows as running them once,
because every write is an upsert keyed on Codeforces' own ids.
Design reasons: docs/decisions.md #17.

Run it:  uv run python -m cf_coach.sync.sync TitanCoder1258
"""

import logging
import sys
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from cf_coach.db.models import Problem, Submission, User
from cf_coach.db.session import get_engine
from cf_coach.sync.cf_client import get_problemset, get_user_info, get_user_submissions

logger = logging.getLogger(__name__)

PAGE_SIZE = 500                   # submissions per user.status call
PRACTICE_TIME = 2**31 - 1         # CF's fake relativeTimeSeconds (2147483647) for practice
UNFINISHED = {None, "TESTING"}    # verdicts that can still change


# ---------- Problems ----------

def sync_problemset() -> int:
    """Upsert every problem. Returns how many problems were written."""
    problems = get_problemset()["problems"]
    rows = [
        {
            "contest_id": p["contestId"],
            "idx": p["index"],
            "name": p["name"],
            "rating": p.get("rating"),     # missing for new or unrated problems
            "tags": p.get("tags", []),
        }
        for p in problems
    ]

    # INSERT ... ON CONFLICT (contest_id, idx) DO UPDATE: new problems are inserted;
    # existing ones get their latest name, rating and tags (a new problem gets its
    # rating days after the contest, so DO NOTHING would keep it unrated forever).
    stmt = insert(Problem)
    stmt = stmt.on_conflict_do_update(
        index_elements=["contest_id", "idx"],
        set_={"name": stmt.excluded.name, "rating": stmt.excluded.rating, "tags": stmt.excluded.tags},
    )
    with Session(get_engine()) as session, session.begin():
        session.execute(stmt, rows)
    return len(rows)


# ---------- Submissions ----------

def _fetch_new_submissions(handle: str, cursor: int | None) -> tuple[list[dict], int]:
    """Page through user.status (newest first) until we reach the cursor.

    cursor = the newest submission id already saved (None on a first sync).
    Returns (new submissions, number of pages fetched).
    """
    new, start, pages = [], 1, 0
    while True:
        page = get_user_submissions(handle, start=start, count=PAGE_SIZE)
        pages += 1
        for s in page:
            if cursor is not None and s["id"] <= cursor:
                return new, pages          # everything from here on is already saved
            new.append(s)
        if len(page) < PAGE_SIZE:
            return new, pages              # reached the user's oldest submission
        start += PAGE_SIZE


def _submission_row(s: dict, handle: str) -> dict:
    """Turn one CF submission (JSON) into one row for the submissions table."""
    rel = s["relativeTimeSeconds"]
    return {
        "id": s["id"],
        "handle": handle,
        "contest_id": s["problem"]["contestId"],
        "idx": s["problem"]["index"],
        "verdict": s.get("verdict"),
        "created_at": datetime.fromtimestamp(s["creationTimeSeconds"], tz=UTC),
        "participant_type": s["author"]["participantType"],
        "relative_time_s": None if rel == PRACTICE_TIME else rel,
    }


def _next_cursor(cursor: int | None, new: list[dict]) -> int | None:
    """The newest id we can safely skip next time.

    Normally that's the newest new id. But if a submission is still being judged, its
    verdict will change, so stop the cursor just below it: the next sync fetches it
    again and the upsert updates its verdict.
    """
    unfinished = [s["id"] for s in new if s.get("verdict") in UNFINISHED]
    if unfinished:
        return min(unfinished) - 1
    if new:
        return max(s["id"] for s in new)
    return cursor


def sync_user(handle: str) -> dict:
    """Sync one user's profile and new submissions. Returns a small summary."""
    info = get_user_info([handle])[0]
    handle = info["handle"]                # CF's spelling (handles are case-insensitive)

    # 1. Read where the last sync stopped. Short session: no CF calls while it's open.
    with Session(get_engine()) as session:
        cursor = session.scalar(select(User.last_synced_sub_id).where(User.handle == handle))

    # 2. Fetch from CF, outside any transaction (this can take many seconds).
    new, pages = _fetch_new_submissions(handle, cursor)
    rows = [_submission_row(s, handle) for s in new if "contestId" in s["problem"]]
    skipped = len(new) - len(rows)         # rare: old problemset-only problems with no contest

    # 3. Write everything in ONE transaction: the user, the submissions and the new
    #    cursor are saved together or not at all. If it crashes halfway, the cursor
    #    hasn't moved, so the next run fetches the same submissions again (and the
    #    upsert makes that harmless).
    with Session(get_engine()) as session, session.begin():
        user_stmt = insert(User).values(handle=handle, rating=info.get("rating"))
        session.execute(user_stmt.on_conflict_do_update(
            index_elements=["handle"], set_={"rating": user_stmt.excluded.rating},
        ))

        if rows:
            # Same submission twice -> same id -> the conflict updates the verdict
            # (TESTING -> OK) instead of adding a duplicate row.
            sub_stmt = insert(Submission)
            sub_stmt = sub_stmt.on_conflict_do_update(
                index_elements=["id"], set_={"verdict": sub_stmt.excluded.verdict},
            )
            session.execute(sub_stmt, rows)

        new_cursor = _next_cursor(cursor, new)
        user = session.get(User, handle)
        user.last_synced_sub_id = new_cursor
        user.synced_at = datetime.now(UTC)

        total = session.scalar(select(func.count()).select_from(Submission).where(Submission.handle == handle))

    # Plain variables only here: after the block, the session is closed and `user`
    # can no longer load its attributes.
    # "fetched" = rows sent to the upsert; on a re-sync these may already exist.
    return {"handle": handle, "pages": pages, "fetched": len(rows), "skipped": skipped,
            "total": total, "cursor": new_cursor}


# ---------- Manual check ----------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    handle = sys.argv[1] if len(sys.argv) > 1 else "TitanCoder1258"

    print("problems written:", sync_problemset())
    print("first sync: ", sync_user(handle))
    print("second sync:", sync_user(handle))
