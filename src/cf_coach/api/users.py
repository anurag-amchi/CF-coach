"""/api/users/{handle}/... routes (slice 3): sync, sync status, profile, progress.

Same split as the tools: a pure *_out() function builds the response from plain rows
(unit-tested), and the route does only the DB read. Handles are case-insensitive.

Try it:  uv run python -m cf_coach.api.app   then open
         http://127.0.0.1:8000/api/users/TitanCoder1258/profile
         http://127.0.0.1:8000/api/users/TitanCoder1258/progress?since=2026-10-01
Design reasons: docs/decisions.md #58, #59.
"""

from datetime import date, datetime, time, timedelta, timezone
from statistics import median

from fastapi import APIRouter, BackgroundTasks

from cf_coach.agent.tools import first_solves_since, problem_id, skill_changes, topic_tags
from cf_coach.analytics.attempts import SubmissionRow, build_attempts, load_rows
from cf_coach.analytics.elo import tag_skills
from cf_coach.analytics.selection import IGNORED_TAGS, is_special, round_skill
from cf_coach.analytics.signals import TagSignals, contest_solve_times, tag_signals, upsolve_backlog
from cf_coach.analytics.tag_profile import DEFAULT_START_RATING
from cf_coach.api.common import ERRORS, ApiError, DbSession, Handle, RedisConn, synced_user
from cf_coach.api.contract import (
    ProfileOut, ProgressOut, SolvedOut, SyncStatusOut, TagChangeOut, TagOut,
)
from cf_coach.api.sync_job import run_sync, start_sync, sync_status

router = APIRouter(prefix="/api/users", tags=["users"])

PROGRESS_DAYS = 7              # no ?since= -> the last week (plans are weekly)
MAX_SOLVED_SHOWN = 50          # solved_total still counts them all


# ---------- sync ----------

@router.post("/{handle}/sync", status_code=202, responses=ERRORS)
def post_sync(handle: Handle, session: DbSession, r: RedisConn,
              background: BackgroundTasks) -> SyncStatusOut:
    # 202 Accepted: "started, not finished". The browser then polls GET .../sync.
    # Calling it again while a sync runs starts nothing and just returns the status.
    if start_sync(r, handle):
        background.add_task(run_sync, r, handle)    # runs after this reply is sent
    return sync_status(session, r, handle)


@router.get("/{handle}/sync", responses=ERRORS)
def get_sync(handle: Handle, session: DbSession, r: RedisConn) -> SyncStatusOut:
    return sync_status(session, r, handle)


# ---------- profile ----------


def profile_out(handle: str, rating: int | None, synced_at: datetime,
                rows: list[SubmissionRow], now: datetime) -> ProfileOut:
    """The profile page's numbers, from one user's submissions (joined with problems).

    Computed from the rows on each request, like find_problems does, so the page shows
    the same skills the plan's facts lines use.
    """
    start = rating if rating is not None else DEFAULT_START_RATING   # unrated -> 1400
    skills = tag_skills(build_attempts(rows, now), start)
    signals = tag_signals(rows, now)

    # The same backlog get_upsolve_backlog shows the agent: special rounds left out (#57)
    backlog = [b for b in upsolve_backlog(rows) if not is_special(b.tags)]
    backlog_by_tag: dict[str, int] = {}
    for b in backlog:
        for tag in b.tags:
            backlog_by_tag[tag] = backlog_by_tag.get(tag, 0) + 1

    tags = []
    for tag, elo in sorted(skills.items(), key=lambda kv: (kv[1].skill, kv[0])):  # weakest first
        if tag in IGNORED_TAGS:
            continue
        s = signals.get(tag, TagSignals())
        avg_wrong = s.avg_wrong_before_ac
        tags.append(TagOut(
            tag=tag, skill=round(elo.skill), level=round_skill(elo.skill),
            attempts=elo.attempts, solved=s.solved, abandon_rate=round(s.abandon_rate, 2),
            avg_wrong_before_ac=None if avg_wrong is None else round(avg_wrong, 2),
            backlog=backlog_by_tag.get(tag, 0)))

    minutes = [m for _, m in contest_solve_times(rows)]
    return ProfileOut(
        handle=handle, rating=rating, synced_at=synced_at,
        solved_total=len({(r.contest_id, r.idx) for r in rows if r.verdict == "OK"}),
        backlog_total=len(backlog),
        median_solve_minutes=round(median(minutes), 1) if minutes else None,
        tags=tags)


@router.get("/{handle}/profile", responses=ERRORS)
def get_profile(handle: Handle, session: DbSession) -> ProfileOut:
    # A plain `def`, not `async def`: SQLAlchemy's session blocks while it waits for
    # Postgres, so FastAPI runs this in a worker thread and the server stays free.
    user = synced_user(session, handle)
    return profile_out(user.handle, user.rating, user.synced_at,
                       load_rows(session, user.handle), datetime.now(timezone.utc))


# ---------- progress ----------

def progress_out(rows: list[SubmissionRow], since: date, now: datetime,
                 start: int) -> ProgressOut:
    """Problems first solved on or after `since` (newest first) and how each tag's skill
    moved since then. The same helpers as the get_progress_since tool, so the page and
    the agent agree; only the list length (50, not 30) and the API models differ."""
    since_dt = datetime.combine(since, time(), tzinfo=timezone.utc)   # midnight UTC
    solves = first_solves_since(rows, since_dt)
    return ProgressOut(
        since=since,
        solved_total=len(solves),
        solved=[SolvedOut(id=problem_id(r.contest_id, r.idx), rating=r.rating,
                          tags=topic_tags(r.tags), solved_on=r.created_at.date())
                for r in solves[:MAX_SOLVED_SHOWN]],
        # the tool returns its own TagChangeOut class; copy the fields into the API's
        skill_changes=[TagChangeOut(**c.model_dump())
                       for c in skill_changes(rows, since_dt, now, start)],
    )


@router.get("/{handle}/progress", responses=ERRORS)
def get_progress(handle: Handle, session: DbSession, since: date | None = None) -> ProgressOut:
    # ?since=2026-10-01 -> FastAPI turns the text into a `date`; "2026-13-01" is a 422
    now = datetime.now(timezone.utc)
    since = since or now.date() - timedelta(days=PROGRESS_DAYS)
    if since > now.date():
        raise ApiError(422, "validation", f"since: {since} is in the future")
    user = synced_user(session, handle)
    start = user.rating if user.rating is not None else DEFAULT_START_RATING
    return progress_out(load_rows(session, user.handle), since, now, start)
