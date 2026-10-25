"""The agent's tools: thin wrappers over the analytics code.

Each tool is  fn(ctx, args) -> output model.
- ctx (ToolContext) is filled in by the loop: whose data, the DB session, "now".
- args is an input object from schemas.py, already checked by parse_args.
A tool raises ToolError when the model asked for something impossible (an unknown tag,
for example). The loop sends that message back to the model so it can try again.

Each tool = a database read + a pure *_out() function that builds the output. The pure
parts are unit-tested; the database parts are checked by running this file.

All 6: get_tag_profile, get_upsolve_backlog, find_problems, get_progress_since,
validate_plan, save_plan (the only one that writes).

Run on real data:  uv run python -m cf_coach.agent.tools TitanCoder1258 [--save]
Design reasons: docs/decisions.md #26, #28, #29.
"""

import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from cf_coach.agent.schemas import (
    BacklogItemOut, BacklogOut, FindProblemsInput, FindProblemsOut, GetProgressSinceInput,
    GetTagProfileInput, GetUpsolveBacklogInput, PlanConstraints, PlanItem, ProblemOut,
    ProgressOut, SavePlanInput, SavePlanOut, SolvedOut, TagChangeOut, TagProfileOut,
    TagSkillOut, ValidatePlanInput, ValidationOut,
)
from cf_coach.analytics.attempts import SubmissionRow, build_attempts, load_rows
from cf_coach.analytics.elo import tag_skills
from cf_coach.analytics.selection import (
    BAND, IGNORED_TAGS, Candidate, ContestPrepStrategy, ProblemRow, Profile,
    WeaknessFirstStrategy, build_profile,
    load_problems, load_solved, running_contest_ids, select_candidates,
)
from cf_coach.analytics.signals import BacklogItem, upsolve_backlog
from cf_coach.analytics.tag_profile import start_rating
from cf_coach.db.models import Plan, TagSkill
from cf_coach.db.session import get_engine

# Keys must match the Literal in FindProblemsInput.strategy (a test checks this).
STRATEGIES = {"weakness_first": WeaknessFirstStrategy, "contest_prep": ContestPrepStrategy}


def topic_tags(tags) -> list[str]:
    """Drop CF markers like '*special' (April Fools): they aren't topics to practise."""
    return [t for t in tags if t not in IGNORED_TAGS]


class ToolError(Exception):
    """The request can't be done. The message is written for the model to read."""


@dataclass
class ToolContext:
    handle: str
    session: Session
    now: datetime
    running_contests: set[int] | None = None   # None = not fetched yet
    saved_plan_id: int | None = None           # set by save_plan; the loop logs it

    def get_running_contests(self) -> set[int]:
        """Fetched from CF once per run, on first use, then reused by every tool call."""
        if self.running_contests is None:
            self.running_contests = running_contest_ids()
        return self.running_contests


# ---------- get_tag_profile ----------

def profile_out(rating: int, rows: list[TagSkill]) -> TagProfileOut:
    """Weakest tag first, numbers rounded."""
    tags = [TagSkillOut(tag=r.tag, skill=round(r.skill), attempts=r.attempts,
                        abandon_rate=round(r.abandon_rate, 2))
            for r in sorted(rows, key=lambda r: (r.skill, r.tag)) if r.tag not in IGNORED_TAGS]
    return TagProfileOut(rating=rating, tags=tags)


def get_tag_profile(ctx: ToolContext, args: GetTagProfileInput) -> TagProfileOut:
    rating = start_rating(ctx.session, ctx.handle)
    rows = ctx.session.scalars(select(TagSkill).where(TagSkill.handle == ctx.handle)).all()
    return profile_out(rating, list(rows))


# ---------- get_upsolve_backlog ----------

def backlog_out(items: list[BacklogItem], limit: int) -> BacklogOut:
    return BacklogOut(total=len(items), items=[
        BacklogItemOut(contest_id=b.contest_id, idx=b.idx, rating=b.rating,
                       tags=topic_tags(b.tags), last_tried=b.last_tried_at.date())
        for b in items[:limit]])


def get_upsolve_backlog(ctx: ToolContext, args: GetUpsolveBacklogInput) -> BacklogOut:
    return backlog_out(upsolve_backlog(load_rows(ctx.session, ctx.handle)), args.limit)


# ---------- find_problems ----------

def check_tags(asked: list[str], known: set[str]) -> None:
    """Raise ToolError for any tag CF doesn't have, listing the real ones."""
    unknown = [t for t in asked if t not in known]
    if unknown:
        raise ToolError(f"unknown tag(s) {unknown}. Valid tags: {', '.join(sorted(known))}")


def problem_out(c: Candidate) -> ProblemOut:
    p = c.problem
    return ProblemOut(contest_id=p.contest_id, idx=p.idx, name=p.name, rating=p.rating,
                      tags=topic_tags(p.tags), matched_tag=c.tag, tag_skill=round(c.tag_skill),
                      score=round(c.score, 2))


def all_tags(problems: list[ProblemRow]) -> set[str]:
    return {t for p in problems for t in p.tags} - IGNORED_TAGS


def find_problems(ctx: ToolContext, args: FindProblemsInput) -> FindProblemsOut:
    problems = load_problems(ctx.session)
    check_tags(([args.tag] if args.tag else []) + args.exclude_tags, all_tags(problems))
    candidates = select_candidates(
        problems,
        load_solved(ctx.session, ctx.handle),
        build_profile(ctx.session, ctx.handle, ctx.now),
        ctx.get_running_contests(),
        STRATEGIES[args.strategy](),
        tag=args.tag,
        excluded_tags=frozenset(args.exclude_tags),
        count=args.count,
    )
    return FindProblemsOut(problems=[problem_out(c) for c in candidates])


# ---------- get_progress_since ----------

MAX_SOLVED_LISTED = 30         # caps the output size; solved_total still counts all


def first_solves_since(rows: list[SubmissionRow], since: datetime) -> list[SubmissionRow]:
    """The first AC of every problem whose FIRST AC is at or after `since`. Newest first.

    A problem solved before `since` and re-submitted after it is not new progress.
    """
    first_ac: dict[tuple[int, str], SubmissionRow] = {}
    for r in rows:
        key = (r.contest_id, r.idx)
        if r.verdict == "OK" and (key not in first_ac or r.created_at < first_ac[key].created_at):
            first_ac[key] = r
    new = [r for r in first_ac.values() if r.created_at >= since]
    return sorted(new, key=lambda r: r.created_at, reverse=True)


def skill_changes(rows: list[SubmissionRow], since: datetime, now: datetime,
                  start: int) -> list[TagChangeOut]:
    """Tag skill as it was at `since` vs now. Only tags that moved, biggest move first.

    "Skill at `since`" = the same Elo run on only the submissions made before `since`,
    with `since` as "now" (so the 10-day give-up rule is judged at that time).
    A tag first tried after `since` starts from `start`, as Elo itself does.
    """
    before = tag_skills(build_attempts([r for r in rows if r.created_at < since], since), start)
    after = tag_skills(build_attempts(rows, now), start)
    changes = []
    for tag, elo in after.items():
        if tag in IGNORED_TAGS:
            continue
        b = round(before[tag].skill) if tag in before else start
        a = round(elo.skill)
        if a != b:
            changes.append(TagChangeOut(tag=tag, before=b, after=a))
    return sorted(changes, key=lambda c: (-abs(c.after - c.before), c.tag))


def progress_out(rows: list[SubmissionRow], since: date, now: datetime,
                 start: int) -> ProgressOut:
    since_dt = datetime.combine(since, time(), tzinfo=timezone.utc)   # midnight UTC
    solves = first_solves_since(rows, since_dt)
    return ProgressOut(
        since=since,
        solved_total=len(solves),
        solved=[SolvedOut(contest_id=r.contest_id, idx=r.idx, rating=r.rating,
                          tags=topic_tags(r.tags), solved_on=r.created_at.date())
                for r in solves[:MAX_SOLVED_LISTED]],
        skill_changes=skill_changes(rows, since_dt, now, start),
    )


def get_progress_since(ctx: ToolContext, args: GetProgressSinceInput) -> ProgressOut:
    today = ctx.now.date()
    if args.since > today:
        raise ToolError(f"since={args.since} is in the future; today is {today}")
    return progress_out(load_rows(ctx.session, ctx.handle), args.since, ctx.now,
                        start_rating(ctx.session, ctx.handle))


# ---------- validate_plan / save_plan ----------

MINUTES_PER_PROBLEM = 60       # a problem 100-300 above your skill takes about an hour


def plan_errors(items: list[PlanItem], constraints: PlanConstraints,
                problems: dict[tuple[int, str], ProblemRow], solved: set[tuple[int, str]],
                profile: Profile, running: set[int]) -> list[str]:
    """Every rule the plan breaks, one line each. Empty list = the plan is valid.

    Deterministic: the same plan and data always give the same answer. The band rule is
    the same one find_problems uses, so anything find_problems returned passes it.
    """
    errors = []
    if constraints.hours is not None:
        most = constraints.hours * 60 // MINUTES_PER_PROBLEM
        if len(items) > most:
            errors.append(f"{len(items)} problems is too many for {constraints.hours} hours: "
                          f"at most {most} (about {MINUTES_PER_PROBLEM} min each)")
    avoid = set(constraints.avoid_tags)
    seen = set()
    for n, item in enumerate(items, start=1):
        key = (item.contest_id, item.idx)
        where = f"item {n} ({item.contest_id}{item.idx})"
        if key in seen:
            errors.append(f"{where}: listed twice")
            continue
        seen.add(key)
        p = problems.get(key)
        if p is None:
            errors.append(f"{where}: no such problem in the problemset")
            continue
        if key in solved:
            errors.append(f"{where}: already solved")
        if p.contest_id in running:
            errors.append(f"{where}: contest {p.contest_id} is still running")
        if hit := avoid & set(p.tags):
            errors.append(f"{where}: has avoided tag(s) {sorted(hit)}")
        if item.tag in IGNORED_TAGS or item.tag not in p.tags:
            errors.append(f"{where}: '{item.tag}' is not one of its tags {topic_tags(p.tags)}")
            continue
        if p.rating is None:
            errors.append(f"{where}: unrated, so it can't be matched to a skill band")
            continue
        skill = profile.skills.get(item.tag, profile.rating)
        if not (skill + BAND[0] <= p.rating <= skill + BAND[1]):
            errors.append(f"{where}: rating {p.rating} is outside the {item.tag} band "
                          f"{round(skill + BAND[0])}-{round(skill + BAND[1])}")
    return errors


def check_plan(ctx: ToolContext, args: ValidatePlanInput) -> list[str]:
    """Load what the rules need, then run them. Shared by validate_plan and save_plan."""
    problems = {(p.contest_id, p.idx): p for p in load_problems(ctx.session)}
    return plan_errors(args.items, args.constraints, problems,
                       load_solved(ctx.session, ctx.handle),
                       build_profile(ctx.session, ctx.handle, ctx.now),
                       ctx.get_running_contests())


def validate_plan(ctx: ToolContext, args: ValidatePlanInput) -> ValidationOut:
    errors = check_plan(ctx, args)
    return ValidationOut(ok=not errors, errors=errors)


def save_plan(ctx: ToolContext, args: SavePlanInput) -> SavePlanOut:
    """Validate again, then save as the user's one active plan (older ones -> replaced)."""
    if ctx.saved_plan_id is not None:
        raise ToolError(f"a plan was already saved in this run (plan {ctx.saved_plan_id})")
    errors = check_plan(ctx, args)
    if errors:
        raise ToolError("plan not saved, it breaks these rules: " + "; ".join(errors)
                        + ". Fix them and call validate_plan again.")
    session = ctx.session
    try:
        session.execute(update(Plan)
                        .where(Plan.handle == ctx.handle, Plan.status == "active")
                        .values(status="replaced"))
        plan = Plan(handle=ctx.handle, constraints=args.constraints.model_dump(),
                    items=[item.model_dump() for item in args.items], status="active")
        session.add(plan)
        session.flush()            # sends the INSERT, so Postgres assigns plan.id
        plan_id = plan.id
        session.commit()           # the UPDATE and the INSERT land together, or neither does
    except Exception:
        session.rollback()
        raise
    ctx.saved_plan_id = plan_id
    return SavePlanOut(plan_id=plan_id, items=len(args.items))


# name -> function. The input class for each name is in schemas.TOOLS.
TOOL_FUNCTIONS = {
    "get_tag_profile": get_tag_profile,
    "get_upsolve_backlog": get_upsolve_backlog,
    "find_problems": find_problems,
    "get_progress_since": get_progress_since,
    "validate_plan": validate_plan,
    "save_plan": save_plan,
}


# ---------- Manual check: each tool on real data ----------

if __name__ == "__main__":
    handle = sys.argv[1]
    calls = [
        ("get_tag_profile", GetTagProfileInput()),
        ("get_upsolve_backlog", GetUpsolveBacklogInput(limit=3)),
        ("find_problems", FindProblemsInput(tag="dp", count=3)),
        ("find_problems", FindProblemsInput(strategy="contest_prep", exclude_tags=["geometry"],
                                            count=3)),
        ("find_problems", FindProblemsInput(tag="DP")),
        ("get_progress_since", GetProgressSinceInput(since=date(2026, 9, 1))),
        ("get_progress_since", GetProgressSinceInput(since=date(2025, 1, 1))),
        ("get_progress_since", GetProgressSinceInput(since=date(2030, 1, 1))),
    ]
    with Session(get_engine()) as session:
        ctx = ToolContext(handle, session, datetime.now(timezone.utc))
        for name, args in calls:
            try:
                out = TOOL_FUNCTIONS[name](ctx, args).model_dump_json()
                print(f"\n{name}({args!r})  [{len(out)} chars]\n{out}")
            except ToolError as e:
                print(f"\n{name}({args!r})  ToolError: {e}")

        # A plan built from find_problems' own output, and a broken copy of it.
        found = find_problems(ctx, FindProblemsInput(tag="dp", count=2)).problems
        good = [PlanItem(contest_id=p.contest_id, idx=p.idx, tag=p.matched_tag,
                         reason="dp is below your other tags") for p in found]
        bad = good + [
            PlanItem(contest_id=1797, idx="B", tag="greedy", reason="backlog"),  # too easy
            PlanItem(contest_id=2051, idx="A", tag="greedy", reason="x"),        # solved
            PlanItem(contest_id=1, idx="Z", tag="dp", reason="x"),               # doesn't exist
            good[0],                                                             # duplicate
        ]
        bad_plan = ValidatePlanInput(items=bad, constraints=PlanConstraints(
            hours=3, avoid_tags=["trees"]))
        good_plan = SavePlanInput(items=good, constraints=PlanConstraints(hours=3))
        for title, plan in (("broken plan", bad_plan), ("good plan", good_plan)):
            print(f"\nvalidate_plan, {title}:\n{validate_plan(ctx, plan).model_dump_json(indent=1)}")

        if "--save" in sys.argv:
            print("\nsave_plan:", save_plan(ctx, good_plan))
            try:
                save_plan(ctx, good_plan)
            except ToolError as e:
                print("save_plan again:", e)
