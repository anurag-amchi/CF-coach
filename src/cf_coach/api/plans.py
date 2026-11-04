"""/api/plans routes (slice 5): the current plan and the saved plans, read from Postgres.

Same split as users.py: pure functions (read_plan, plan_out) build the response from plain
rows and are unit-tested; the routes only do the DB reads. The plans table is keyed by CF
handle, so an account sees the plans of the handle it set with PATCH /api/me.

Try it:  uv run python -m cf_coach.api.app   then (signed in) open
         http://127.0.0.1:8000/api/plans/current
Design reasons: docs/decisions.md #61.
"""

import logging

from fastapi import APIRouter
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cf_coach.agent.tools import SavedConstraints, SavedItem, plan_summary
from cf_coach.analytics.selection import load_solved
from cf_coach.api.auth import AUTH_ERRORS, CurrentAccount, OptionalAccount, SignedIn
from cf_coach.api.common import ApiError, DbSession
from cf_coach.api.contract import PlanItemOut, PlanOut, PlansOut
from cf_coach.db.models import Plan, Problem

log = logging.getLogger("cf_coach.api")
router = APIRouter(prefix="/api/plans", tags=["plans"])

MAX_PLANS_SHOWN = 20           # GET /api/plans: the newest 20


def problem_url(contest_id: int, idx: str) -> str:
    return f"https://codeforces.com/problemset/problem/{contest_id}/{idx}"


def read_plan(constraints: dict, items: list) -> tuple[SavedConstraints, list[SavedItem]] | None:
    """The stored JSON as the models save_plan wrote it, or None for a plan saved before
    items had id / rating / gap / facts (decisions #35, #40): those can't be shown."""
    try:
        return (SavedConstraints.model_validate(constraints),
                [SavedItem.model_validate(i) for i in items])
    except ValidationError:
        return None


def plan_out(plan_id: int, handle: str, created_at, constraints: SavedConstraints,
             items: list[SavedItem], names: dict[tuple[int, str], str],
             solved: set[tuple[int, str]]) -> PlanOut:
    """One saved plan for the page. `solved` comes from the latest sync, so a problem
    solved after the plan was made shows as done without changing the stored plan."""
    return PlanOut(
        id=str(plan_id), handle=handle, created_at=created_at,
        difficulty=constraints.difficulty, hours=constraints.hours,
        avoid_tags=constraints.avoid_tags,
        summary=plan_summary(constraints, items),
        items=[PlanItemOut(
            id=i.id,
            # every saved item came from the problems table; the id is only a fallback
            name=names.get((i.contest_id, i.idx), i.id),
            url=problem_url(i.contest_id, i.idx),
            tag=i.tag, rating=i.rating, gap=i.gap, facts=i.facts, reason=i.reason,
            solved=(i.contest_id, i.idx) in solved,
        ) for i in items],          # stored easiest first (#36): keep that order
        saved=True, expires_at=None)


# ---------- DB reads (the tests replace these three) ----------

def load_plans(session: Session, handle: str, active_only: bool, limit: int) -> list[Plan]:
    query = select(Plan).where(func.lower(Plan.handle) == handle.lower())
    if active_only:
        query = query.where(Plan.status == "active")
    return list(session.scalars(query.order_by(Plan.id.desc()).limit(limit)))


def load_names(session: Session, keys: set[tuple[int, str]]) -> dict[tuple[int, str], str]:
    """Problem names for these (contest_id, idx) keys, in one query."""
    if not keys:
        return {}
    rows = session.execute(select(Problem.contest_id, Problem.idx, Problem.name)
                           .where(Problem.contest_id.in_({c for c, _ in keys})))
    return {(c, i): name for c, i, name in rows if (c, i) in keys}


def plans_out(session: Session, rows: list[Plan]) -> list[PlanOut]:
    """Rows -> PlanOut, skipping (and logging) plans in the old format."""
    readable = []
    for row in rows:
        parsed = read_plan(row.constraints, row.items)
        if parsed is None:
            log.info("plan %s is in the old format; not shown", row.id)
        else:
            readable.append((row, *parsed))
    if not readable:
        return []
    keys = {(i.contest_id, i.idx) for _, _, items in readable for i in items}
    names = load_names(session, keys)
    solved = load_solved(session, readable[0][0].handle)   # one handle per request
    return [plan_out(row.id, row.handle, row.created_at, c, items, names, solved)
            for row, c, items in readable]


def account_handle(me: SignedIn) -> str:
    if me.account.handle is None:
        raise ApiError(404, "not_found", "set your Codeforces handle first")
    return me.account.handle


# ---------- routes ----------

@router.get("/current", responses=AUTH_ERRORS)
def get_current_plan(session: DbSession, me: OptionalAccount) -> PlanOut:
    if me is None:
        # Slice 4c: a guest's plan lives in Redis under its guest cookie.
        raise ApiError(404, "not_found", "no plan yet: make one first")
    plans = plans_out(session, load_plans(session, account_handle(me), active_only=True, limit=1))
    if not plans:
        raise ApiError(404, "not_found", "no plan yet: make one first")
    return plans[0]


@router.get("", responses=AUTH_ERRORS)
def get_plans(session: DbSession, me: CurrentAccount) -> PlansOut:
    # Newest first: the active plan, then the ones it replaced.
    rows = load_plans(session, account_handle(me), active_only=False, limit=MAX_PLANS_SHOWN)
    return PlansOut(plans=plans_out(session, rows))
