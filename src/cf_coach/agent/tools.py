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

The user's own constraints (hours, avoided tags; ToolContext.user_constraints) are applied
by find_problems, validate_plan and save_plan whatever the model passes.
save_plan writes through ToolContext.plan_store: Postgres (DbPlanStore) unless the caller
passes another store (the evals use MemoryPlanStore).

Plans name problems by id ("2242D"). find_problems and get_upsolve_backlog remember every
id they return (ToolContext.offered), and validate_plan / save_plan accept only those.

Run on real data:  uv run python -m cf_coach.agent.tools TitanCoder1258 [--save]
Design reasons: docs/decisions.md #26, #28, #29, #32, #34, #35.
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone

from pydantic import BaseModel
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


def problem_id(contest_id: int, idx: str) -> str:
    """(2242, "D") -> "2242D", the id the model sees and plans with.

    Only ever built, never parsed back: contest 921 has indices "01".."14", so "92101"
    can't be split by a pattern. The split is kept in ToolContext.offered instead.
    """
    return f"{contest_id}{idx}"


@dataclass(frozen=True)
class Offer:
    """A problem a tool returned in this run, so a plan may use it."""
    contest_id: int
    idx: str
    tag: str | None        # the tag find_problems matched it to; None = from the backlog


class SavedItem(BaseModel):
    """A plan item as stored: the model's id and reason, plus what the code worked out."""
    id: str
    contest_id: int
    idx: str
    tag: str               # the tag it practises (see plan_tag)
    reason: str


class PlanStore(ABC):
    """Where save_plan writes a validated plan: Postgres normally, memory in the evals."""

    @abstractmethod
    def save(self, handle: str, constraints: PlanConstraints, items: list[SavedItem]) -> int:
        """Store the plan as the user's active one; return its id."""


@dataclass
class ToolContext:
    handle: str
    session: Session
    now: datetime
    running_contests: set[int] | None = None   # None = not fetched yet
    saved_plan_id: int | None = None           # set by save_plan; the loop logs it
    # What the user set outside the chat (CLI flags, a form). Enforced on every plan,
    # whatever the model passes. See effective_constraints().
    user_constraints: PlanConstraints = field(default_factory=PlanConstraints)
    plan_store: PlanStore | None = None        # None = the database (DbPlanStore)
    # Every problem id a tool returned in this run. A plan may only use these.
    offered: dict[str, Offer] = field(default_factory=dict)

    def get_running_contests(self) -> set[int]:
        """Fetched from CF once per run, on first use, then reused by every tool call."""
        if self.running_contests is None:
            self.running_contests = running_contest_ids()
        return self.running_contests

    def offer(self, contest_id: int, idx: str, tag: str | None) -> None:
        """Remember a returned problem. A find_problems tag is never replaced by None:
        the same problem can come back from the backlog later."""
        pid = problem_id(contest_id, idx)
        if tag is not None or pid not in self.offered:
            self.offered[pid] = Offer(contest_id, idx, tag)


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
        BacklogItemOut(id=problem_id(b.contest_id, b.idx), rating=b.rating,
                       tags=topic_tags(b.tags), last_tried=b.last_tried_at.date())
        for b in items[:limit]])


def get_upsolve_backlog(ctx: ToolContext, args: GetUpsolveBacklogInput) -> BacklogOut:
    items = upsolve_backlog(load_rows(ctx.session, ctx.handle))
    for b in items[:args.limit]:           # only the ones the model is shown
        ctx.offer(b.contest_id, b.idx, None)
    return backlog_out(items, args.limit)


# ---------- find_problems ----------

def check_tags(asked: list[str], known: set[str]) -> None:
    """Raise ToolError for any tag CF doesn't have, listing the real ones."""
    unknown = [t for t in asked if t not in known]
    if unknown:
        raise ToolError(f"unknown tag(s) {unknown}. Valid tags: {', '.join(sorted(known))}")


def problem_out(c: Candidate) -> ProblemOut:
    p = c.problem
    return ProblemOut(id=problem_id(p.contest_id, p.idx), name=p.name, rating=p.rating,
                      tags=topic_tags(p.tags), matched_tag=c.tag, tag_skill=round(c.tag_skill),
                      score=round(c.score, 2))


def all_tags(problems: list[ProblemRow]) -> set[str]:
    return {t for p in problems for t in p.tags} - IGNORED_TAGS


def find_problems(ctx: ToolContext, args: FindProblemsInput) -> FindProblemsOut:
    problems = load_problems(ctx.session)
    check_tags(([args.tag] if args.tag else []) + args.avoid_tags, all_tags(problems))
    candidates = select_candidates(
        problems,
        load_solved(ctx.session, ctx.handle),
        build_profile(ctx.session, ctx.handle, ctx.now),
        ctx.get_running_contests(),
        STRATEGIES[args.strategy](),
        tag=args.tag,
        # the user's avoided tags too, so the model never sees a problem it can't plan
        excluded_tags=frozenset(args.avoid_tags + ctx.user_constraints.avoid_tags),
        count=args.count,
    )
    for c in candidates:
        ctx.offer(c.problem.contest_id, c.problem.idx, c.tag)
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
        solved=[SolvedOut(id=problem_id(r.contest_id, r.idx), rating=r.rating,
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


def user_constraints(session: Session, hours: int | None,
                     avoid_tags: list[str]) -> PlanConstraints:
    """Constraints the user typed outside the chat. Checked once here, then put in ToolContext."""
    if avoid_tags:
        check_tags(avoid_tags, all_tags(load_problems(session)))
    return PlanConstraints(hours=hours, avoid_tags=avoid_tags)


def effective_constraints(user: PlanConstraints, model: PlanConstraints) -> PlanConstraints:
    """What a plan is checked against: the user's hours win; avoided tags from both count.

    The model's own constraints (read from the chat text) still apply, so a model that
    copies them correctly loses nothing; one that drops them can't weaken the user's.
    """
    hours = user.hours if user.hours is not None else model.hours
    tags = user.avoid_tags + [t for t in model.avoid_tags if t not in user.avoid_tags]
    return PlanConstraints(hours=hours, avoid_tags=tags)


def clean_id(raw: str) -> str:
    """' 2242d ' -> '2242D': forgive spacing and case, nothing else."""
    return raw.strip().upper()


def tag_skill(profile: Profile, tag: str) -> float:
    return profile.skills.get(tag, profile.rating)    # an untried tag starts at the CF rating


def in_band(profile: Profile, tag: str, rating: int) -> bool:
    """The same band rule find_problems uses, so anything it returned passes."""
    skill = tag_skill(profile, tag)
    return skill + BAND[0] <= rating <= skill + BAND[1]


def plan_tag(offer: Offer, p: ProblemRow, profile: Profile) -> str | None:
    """The tag a planned problem practises; None if no tag's band fits its rating.

    From find_problems: the tag it was matched to. From the backlog (no matched tag): its
    weakest tag whose band holds the rating.
    """
    if offer.tag is not None:
        return offer.tag
    fits = [t for t in topic_tags(p.tags) if p.rating is not None and in_band(profile, t, p.rating)]
    return min(fits, key=lambda t: (tag_skill(profile, t), t)) if fits else None


def plan_errors(items: list[PlanItem], constraints: PlanConstraints, offered: dict[str, Offer],
                problems: dict[tuple[int, str], ProblemRow], solved: set[tuple[int, str]],
                profile: Profile, running: set[int]) -> list[str]:
    """Every rule the plan breaks, one line each. Empty list = the plan is valid.

    Deterministic: the same plan and data always give the same answer.
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
        pid = clean_id(item.id)
        where = f"item {n} ({pid})"
        if pid in seen:
            errors.append(f"{where}: listed twice")
            continue
        seen.add(pid)
        offer = offered.get(pid)
        if offer is None:
            errors.append(f"{where}: no tool returned this id in this conversation; "
                          "plan only ids from find_problems or get_upsolve_backlog")
            continue
        key = (offer.contest_id, offer.idx)
        p = problems.get(key)
        if p is None:                      # a backlog problem CF left out of the problemset
            errors.append(f"{where}: no such problem in the problemset")
            continue
        if key in solved:
            errors.append(f"{where}: already solved")
        if p.contest_id in running:
            errors.append(f"{where}: contest {p.contest_id} is still running")
        if hit := avoid & set(p.tags):
            errors.append(f"{where}: has avoided tag(s) {sorted(hit)}")
        if p.rating is None:
            errors.append(f"{where}: unrated, so it can't be matched to a skill band")
            continue
        tag = plan_tag(offer, p, profile)
        if tag is None:
            errors.append(f"{where}: rating {p.rating} is outside the skill band of every one "
                          f"of its tags {topic_tags(p.tags)}")
        elif not in_band(profile, tag, p.rating):
            skill = tag_skill(profile, tag)
            errors.append(f"{where}: rating {p.rating} is outside the {tag} band "
                          f"{round(skill + BAND[0])}-{round(skill + BAND[1])}")
    return errors


def saved_items(items: list[PlanItem], offered: dict[str, Offer],
                problems: dict[tuple[int, str], ProblemRow], profile: Profile) -> list[SavedItem]:
    """What gets stored. Only call on a plan with no errors (every id offered, every tag found)."""
    out = []
    for item in items:
        offer = offered[clean_id(item.id)]
        p = problems[(offer.contest_id, offer.idx)]
        out.append(SavedItem(id=problem_id(offer.contest_id, offer.idx),
                             contest_id=offer.contest_id, idx=offer.idx,
                             tag=plan_tag(offer, p, profile), reason=item.reason))
    return out


def check_plan(ctx: ToolContext, items: list[PlanItem],
               constraints: PlanConstraints) -> tuple[list[str], list[SavedItem]]:
    """Load what the rules need, then run them. Shared by validate_plan and save_plan.

    Returns (errors, the items as they would be saved); the items are [] when there are errors.
    """
    problems = {(p.contest_id, p.idx): p for p in load_problems(ctx.session)}
    profile = build_profile(ctx.session, ctx.handle, ctx.now)
    errors = plan_errors(items, constraints, ctx.offered, problems,
                         load_solved(ctx.session, ctx.handle), profile,
                         ctx.get_running_contests())
    return errors, [] if errors else saved_items(items, ctx.offered, problems, profile)


def validate_plan(ctx: ToolContext, args: ValidatePlanInput) -> ValidationOut:
    constraints = effective_constraints(ctx.user_constraints, args.constraints)
    errors, _ = check_plan(ctx, args.items, constraints)
    return ValidationOut(ok=not errors, errors=errors)


def save_plan(ctx: ToolContext, args: SavePlanInput) -> SavePlanOut:
    """Validate again, then save as the user's one active plan (older ones -> replaced).

    The saved constraints are the ones that were enforced, not just what the model sent.
    """
    if ctx.saved_plan_id is not None:
        raise ToolError(f"a plan was already saved in this run (plan {ctx.saved_plan_id})")
    constraints = effective_constraints(ctx.user_constraints, args.constraints)
    errors, items = check_plan(ctx, args.items, constraints)
    if errors:
        raise ToolError("plan not saved, it breaks these rules: " + "; ".join(errors)
                        + ". Fix them and call validate_plan again.")
    store = ctx.plan_store if ctx.plan_store is not None else DbPlanStore(ctx.session)
    plan_id = store.save(ctx.handle, constraints, items)
    ctx.saved_plan_id = plan_id
    return SavePlanOut(plan_id=plan_id, items=len(args.items))


class DbPlanStore(PlanStore):
    """The real store: one active plan per user in the plans table."""

    def __init__(self, session: Session):
        self.session = session

    def save(self, handle, constraints, items):
        session = self.session
        try:
            session.execute(update(Plan)
                            .where(Plan.handle == handle, Plan.status == "active")
                            .values(status="replaced"))
            plan = Plan(handle=handle, constraints=constraints.model_dump(),
                        items=[item.model_dump() for item in items], status="active")
            session.add(plan)
            session.flush()        # sends the INSERT, so Postgres assigns plan.id
            plan_id = plan.id
            session.commit()       # the UPDATE and the INSERT land together, or neither does
        except Exception:
            session.rollback()
            raise
        return plan_id


@dataclass
class MemoryPlanStore(PlanStore):
    """Keeps plans in a dict: the evals use it so they never replace the user's real plan."""
    plans: dict[int, tuple[PlanConstraints, list[SavedItem]]] = field(default_factory=dict)

    def save(self, handle, constraints, items):
        plan_id = len(self.plans) + 1
        self.plans[plan_id] = (constraints, list(items))
        return plan_id


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
        ("find_problems", FindProblemsInput(strategy="contest_prep", avoid_tags=["geometry"],
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

        # A plan built from the tools' own output, and a broken copy of it.
        found = find_problems(ctx, FindProblemsInput(tag="dp", count=2)).problems
        backlog = get_upsolve_backlog(ctx, GetUpsolveBacklogInput(limit=3)).items
        print(f"\noffered so far: {sorted(ctx.offered)}")
        good = [PlanItem(id=p.id, reason="dp is below your other tags") for p in found]
        bad = good + [PlanItem(id=b.id, reason="backlog") for b in backlog] + [
            PlanItem(id="2051A", reason="x"),             # real and solved, but never offered
            PlanItem(id="1Z", reason="x"),                # made up
            PlanItem(id=good[0].id.lower(), reason="x"),  # duplicate, in lower case
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
