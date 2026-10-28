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
Plan items: the model writes only the id and a one-sentence reason (no two alike);
code adds the tag, rating, gap and a facts line with the numbers (item_facts).

Design reasons: docs/decisions.md #26, #28, #29, #32, #34, #35, #40.
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
    DIFFICULTY_GAPS, IGNORED_TAGS, Candidate, ContestPrepStrategy, Difficulty, ProblemRow, Profile,
    WeaknessFirstStrategy, band_for, build_profile, round_skill,
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
    rating: int
    gap: int               # rating above the rounded tag skill: its ladder level
    facts: str             # the numbers behind it, written by code (item_facts)
    reason: str            # the model's one sentence: why this problem


class SavedConstraints(PlanConstraints):
    """What a saved plan was checked against: the constraints plus the difficulty."""
    difficulty: Difficulty = "normal"


class PlanStore(ABC):
    """Where save_plan writes a validated plan: Postgres normally, memory in the evals."""

    @abstractmethod
    def save(self, handle: str, constraints: SavedConstraints, items: list[SavedItem]) -> int:
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
    # Which band find_problems searches and validate_plan checks. Set by the caller
    # (request.plan_limits: the parsed request + flags); the model can't change it.
    difficulty: Difficulty = "normal"
    intent: str = "plan"                       # "question" -> read-only tools (loop.py)
    count: int | None = None                   # exact number of problems; None = no rule
    focus_tags: list[str] = field(default_factory=list)   # every item must practise one
    # set by save_plan with saved_plan_id; the loop builds the closing summary from it
    saved_plan: tuple[SavedConstraints, list[SavedItem]] | None = None
    # did the latest validate_plan return ok? The loop nudges a model that stops after an
    # ok validation without calling save_plan (decision #41)
    validated_ok: bool = False
    # tag -> exact number of problems, when code picked the week's tags (no focus tags in
    # the request; see week_quota). Empty = no rule. Set by the loop. Decision #43.
    tag_quota: dict[str, int] = field(default_factory=dict)
    # tag -> the levels (gaps) its problems must sit at, so the week climbs (week_ladder).
    # Empty = no level rule. Set by the loop with tag_quota. Decision #44.
    tag_gaps: dict[str, list[int]] = field(default_factory=dict)
    # weak tags left out of the week: their search had no problem at some level (#49)
    week_skipped: list[str] = field(default_factory=list)
    # Loaded on first use, then reused for the rest of the run (#50). Nothing in a run
    # changes them: the sync happens before it, and save_plan writes only to plans.
    _problems: list[ProblemRow] | None = field(default=None, repr=False)
    _solved: set[tuple[int, str]] | None = field(default=None, repr=False)
    _profile: Profile | None = field(default=None, repr=False)

    def plan_focus(self) -> list[str]:
        """The tags a plan item must practise: the asked ones, else the week's picked ones."""
        return self.focus_tags or list(self.tag_quota)

    def get_running_contests(self) -> set[int]:
        """Fetched from CF once per run, on first use, then reused by every tool call."""
        if self.running_contests is None:
            self.running_contests = running_contest_ids()
        return self.running_contests

    def problems(self) -> list[ProblemRow]:
        """The whole problemset (~11k rows), read once per run."""
        if self._problems is None:
            self._problems = load_problems(self.session)
        return self._problems

    def solved(self) -> set[tuple[int, str]]:
        if self._solved is None:
            self._solved = load_solved(self.session, self.handle)
        return self._solved

    def profile(self) -> Profile:
        """Skills + signals, built from all the user's submissions: the slow part for big
        histories (rainboy: 13.8k submissions, ~1 s), so built once per run."""
        if self._profile is None:
            self._profile = build_profile(self.session, self.handle, self.now)
        return self._profile

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
                      gap=c.gap, score=round(c.score, 2))


def all_tags(problems: list[ProblemRow]) -> set[str]:
    return {t for p in problems for t in p.tags} - IGNORED_TAGS


def taken_by_other_tags(ctx: ToolContext, tag: str | None) -> set[tuple[int, str]]:
    """Problems an earlier find_problems returned for a different tag (#48).

    A problem has one tag in a plan: the one it was found under. If a greedy + brute force
    problem came back in both searches, the second search would re-label it and a "greedy"
    pick would count as brute force. So each later search skips them.
    """
    return {(o.contest_id, o.idx) for o in ctx.offered.values()
            if o.tag is not None and o.tag != tag}


def find_problems(ctx: ToolContext, args: FindProblemsInput) -> FindProblemsOut:
    problems = ctx.problems()
    check_tags(([args.tag] if args.tag else []) + args.avoid_tags, all_tags(problems))
    candidates = select_candidates(
        problems,
        # skipped like solved ones: already offered under another tag
        ctx.solved() | taken_by_other_tags(ctx, args.tag),
        ctx.profile(),
        ctx.get_running_contests(),
        STRATEGIES[args.strategy](),
        tag=args.tag,
        # the user's avoided tags too, so the model never sees a problem it can't plan
        excluded_tags=frozenset(args.avoid_tags + ctx.user_constraints.avoid_tags),
        count=args.count,
        difficulty=ctx.difficulty,
    )
    for c in candidates:
        ctx.offer(c.problem.contest_id, c.problem.idx, c.tag)
    return FindProblemsOut(difficulty=ctx.difficulty,
                           problems=[problem_out(c) for c in candidates])


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

MINUTES_PER_PROBLEM = 60       # a problem at or a bit above your level takes about an hour


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


def in_band(profile: Profile, tag: str, rating: int, difficulty: Difficulty) -> bool:
    """The same band rule find_problems uses, so anything it returned passes."""
    lo, hi = band_for(tag_skill(profile, tag), difficulty)
    return lo <= rating <= hi


def plan_tag(offer: Offer, p: ProblemRow, profile: Profile, difficulty: Difficulty,
             focus: list[str] = ()) -> str | None:
    """The tag a planned problem practises; None if no tag's band fits its rating.

    From find_problems: the tag it was matched to. From the backlog (no matched tag), or
    when that tag isn't one the user asked for: its weakest (focus) tag whose band holds
    the rating.
    """
    if offer.tag is not None and (not focus or offer.tag in focus):
        return offer.tag
    fits = [t for t in topic_tags(p.tags)
            if (not focus or t in focus)
            and p.rating is not None and in_band(profile, t, p.rating, difficulty)]
    return min(fits, key=lambda t: (tag_skill(profile, t), t)) if fits else None


MIN_ATTEMPTS = 5     # a tag with fewer attempts is not "measured": its skill is mostly the start value


def week_quota(tags: list[TagSkillOut], n: int, avoid: list[str] = ()) -> dict[str, int]:
    """The week's tags when the request names none: 2 problems from the weakest measured
    tag, 1 from each of the next ones, n in total. {} if there aren't enough measured tags.

    `tags` is get_tag_profile's list (weakest first). Tags with fewer than MIN_ATTEMPTS
    attempts are skipped: 3 tries on dfs say little about dfs (#21). They can still be
    planned when the user asks for them by name (focus tags).
    """
    measured = [t.tag for t in tags if t.attempts >= MIN_ATTEMPTS and t.tag not in avoid]
    if n < 1:
        return {}
    if n == 1:
        return {measured[0]: 1} if measured else {}
    if len(measured) < n - 1:                  # 2 + (n - 2) x 1 needs n - 1 tags
        return {}
    return {measured[0]: 2} | {t: 1 for t in measured[1:n - 1]}


def week_ladder(quota: dict[str, int], difficulty: Difficulty = "normal") -> dict[str, list[int]]:
    """The level (gap) of every problem in the week, spread evenly over the band, easiest first.

    normal, {geometry: 2, strings: 1, trees: 1, combinatorics: 1}:
        gaps   +0, +0, +100, +100, +200
        tags   geometry, strings, trees, combinatorics, geometry (one of each, then the extras)
        -> geometry [0, 200], strings [0], trees [100], combinatorics [100]
    The weakest tag climbs inside itself; weaker tags start lower than stronger ones.
    """
    n = sum(quota.values())
    if n == 0:
        return {}
    lo, hi = DIFFICULTY_GAPS[difficulty]
    levels = list(range(lo, hi + 1, 100))
    gaps = [levels[i * len(levels) // n] for i in range(n)]
    order = list(quota) + [t for t, k in quota.items() for _ in range(k - 1)]
    ladder: dict[str, list[int]] = {}
    for tag, gap in zip(order, gaps):
        ladder.setdefault(tag, []).append(gap)
    return ladder


def levels_text(gaps: list[int]) -> str:
    """[0, 200] -> '+0 and +200'"""
    shown = [f"{g:+d}" for g in sorted(gaps)]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def level_roles(gaps: list[int], difficulty: Difficulty = "normal") -> str:
    """Each level as a role word + its gap, for the week note (#46). The words give the
    model something to say in a reason instead of the number.

    normal (band +0..+200): [0, 200] -> 'warm-up (gap +0) and stretch (gap +200)'
    """
    lo, hi = DIFFICULTY_GAPS[difficulty]

    def role(g: int) -> str:
        return "warm-up" if g <= lo else "stretch" if g >= hi else "step-up"

    shown = [f"{role(g)} (gap {g:+d})" for g in sorted(gaps)]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def plan_size(count: int | None, hours: int | None) -> int | None:
    """How many problems the plan will have: the exact count, else what the hours allow."""
    want = wanted_count(count, hours)
    if want is None and hours is not None:
        want = hours * 60 // MINUTES_PER_PROBLEM
    return want


def wanted_count(count: int | None, hours: int | None) -> int | None:
    """The exact plan size: the asked count, capped by what the hours allow."""
    if count is None:
        return None
    return count if hours is None else min(count, hours * 60 // MINUTES_PER_PROBLEM)


def plan_errors(items: list[PlanItem], constraints: PlanConstraints, offered: dict[str, Offer],
                problems: dict[tuple[int, str], ProblemRow], solved: set[tuple[int, str]],
                profile: Profile, running: set[int],
                difficulty: Difficulty = "normal", count: int | None = None,
                focus: list[str] = (), quota: dict[str, int] | None = None,
                gaps: dict[str, list[int]] | None = None) -> list[str]:
    """Every rule the plan breaks, one line each. Empty list = the plan is valid.

    Deterministic: the same plan and data always give the same answer.
    quota (tag -> exact count) is the week's tags code picked (#43); it also acts as the
    focus for re-matching an item's tag. gaps (tag -> levels) is the week's ladder (#44).
    """
    quota = quota or {}
    gaps = gaps or {}
    match_focus = list(focus) or list(quota)
    used: dict[str, list[int]] = {}            # quota tag -> the levels of its items
    used_ids: dict[str, list[str]] = {}        # quota tag -> the ids of its items
    errors = []
    hours = constraints.hours
    want = wanted_count(count, hours)
    if want is not None:
        if len(items) != want:
            why = f"the user asked for {count}"
            if want < count:
                why += f", and {hours} hours allow at most {want}"
            errors.append(f"the plan has {len(items)} problems; it must have exactly {want} "
                          f"({why})")
    elif hours is not None:
        most = hours * 60 // MINUTES_PER_PROBLEM
        if len(items) > most:
            errors.append(f"{len(items)} problems is too many for {hours} hours: "
                          f"at most {most} (about {MINUTES_PER_PROBLEM} min each)")
    avoid = set(constraints.avoid_tags)
    seen = set()
    reasons: dict[str, int] = {}               # normalised reason -> the item that used it first
    for n, item in enumerate(items, start=1):
        pid = clean_id(item.id)
        where = f"item {n} ({pid})"
        if pid in seen:
            errors.append(f"{where}: listed twice")
            continue
        seen.add(pid)
        said = " ".join(item.reason.lower().split())
        if said in reasons:
            errors.append(f"{where}: same reason as item {reasons[said]}; "
                          "give each problem its own reason")
        else:
            reasons[said] = n
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
        tag = plan_tag(offer, p, profile, difficulty, match_focus)
        if tag is None and focus and not set(focus) & set(p.tags):
            errors.append(f"{where}: has none of the tags the user asked for {list(focus)}")
        elif tag is None and quota and not set(quota) & set(p.tags):
            errors.append(f"{where}: has none of this week's tags {list(quota)}")
        elif tag is None:
            errors.append(f"{where}: rating {p.rating} is outside the {difficulty} band of "
                          f"every one of its {'asked ' if match_focus else ''}tags "
                          f"{[t for t in topic_tags(p.tags) if not match_focus or t in match_focus]}")
        elif not in_band(profile, tag, p.rating, difficulty):
            lo, hi = band_for(tag_skill(profile, tag), difficulty)
            errors.append(f"{where}: rating {p.rating} is outside the {difficulty} {tag} "
                          f"band {lo}-{hi}")
        elif tag in quota:
            used.setdefault(tag, []).append(p.rating - round_skill(tag_skill(profile, tag)))
            used_ids.setdefault(tag, []).append(pid)
    for tag, want in quota.items():
        have = used.get(tag, [])
        if len(have) != want:
            # name the items, so the model sees which tag each problem counted for (#48)
            which = f" ({', '.join(used_ids[tag])})" if have else ""
            errors.append(f"this week needs exactly {want} {tag} problem(s); "
                          f"the plan has {len(have)}{which}")
        elif tag in gaps and sorted(have) != sorted(gaps[tag]):
            level = round_skill(tag_skill(profile, tag))
            ratings = ", ".join(str(level + g) for g in sorted(gaps[tag]))
            errors.append(f"this week's {tag} problem(s) must be at level "
                          f"{levels_text(gaps[tag])} (rated {ratings}); "
                          f"the plan has {levels_text(have)}")
    return errors


def item_facts(tag: str, rating: int, profile: Profile, from_backlog: bool) -> str:
    """The numbers behind one plan item, written by code so they are always right:

        dp skill 1415 (level 1400) · 26 tries · 12% abandoned · 1500 = level +100
    """
    level = round_skill(tag_skill(profile, tag))
    if tag in profile.skills:
        head = f"{tag} skill {round(profile.skills[tag])} (level {level})"
        s = profile.signals.get(tag)
        if s is not None and s.attempts:
            head += f" · {s.attempts} tries · {round(s.abandon_rate * 100)}% abandoned"
    else:
        head = f"{tag} untried (level {level}, from your rating)"
    facts = f"{head} · {rating} = level {rating - level:+d}"
    return facts + " · in your upsolve backlog" if from_backlog else facts


def saved_items(items: list[PlanItem], offered: dict[str, Offer],
                problems: dict[tuple[int, str], ProblemRow], profile: Profile,
                difficulty: Difficulty = "normal", focus: list[str] = ()) -> list[SavedItem]:
    """What gets stored, easiest first. Only call on a plan with no errors.

    Sorted by gap (stable, so the model's order stays within a level): the saved plan is
    a ladder whatever order the model listed it in.
    """
    out = []
    for item in items:
        offer = offered[clean_id(item.id)]
        p = problems[(offer.contest_id, offer.idx)]
        tag = plan_tag(offer, p, profile, difficulty, focus)
        out.append(SavedItem(id=problem_id(offer.contest_id, offer.idx),
                             contest_id=offer.contest_id, idx=offer.idx, tag=tag,
                             rating=p.rating,
                             gap=p.rating - round_skill(tag_skill(profile, tag)),
                             facts=item_facts(tag, p.rating, profile, offer.tag is None),
                             reason=item.reason.strip()))
    return sorted(out, key=lambda s: s.gap)


def plan_summary(constraints: SavedConstraints, items: list[SavedItem]) -> str:
    """The run's closing text, built from the saved plan: no table, no extra LLM call.

    "Saved 3 problems, easiest first, rated 1400-1600. Tags: dp x3. Difficulty: normal
    (+0 to +200 around each tag's level)."
    """
    tags: dict[str, int] = {}
    for i in items:                            # first-seen order, i.e. easiest first
        tags[i.tag] = tags.get(i.tag, 0) + 1
    ratings = [i.rating for i in items]
    lo, hi = DIFFICULTY_GAPS[constraints.difficulty]
    parts = [f"Saved {len(items)} problem{'' if len(items) == 1 else 's'}, easiest first, "
             f"rated {min(ratings)}-{max(ratings)}.",
             "Tags: " + ", ".join(f"{t} x{n}" for t, n in tags.items()) + ".",
             f"Difficulty: {constraints.difficulty} ({lo:+d} to {hi:+d} around each tag's level)."]
    if constraints.hours is not None:
        parts.append(f"Hours: {constraints.hours}.")
    if constraints.avoid_tags:
        parts.append(f"Avoiding: {', '.join(constraints.avoid_tags)}.")
    return " ".join(parts)


def check_plan(ctx: ToolContext, items: list[PlanItem],
               constraints: PlanConstraints) -> tuple[list[str], list[SavedItem]]:
    """Load what the rules need, then run them. Shared by validate_plan and save_plan.

    Returns (errors, the items as they would be saved); the items are [] when there are errors.
    """
    problems = {(p.contest_id, p.idx): p for p in ctx.problems()}
    profile = ctx.profile()
    errors = plan_errors(items, constraints, ctx.offered, problems, ctx.solved(), profile,
                         ctx.get_running_contests(), ctx.difficulty, ctx.count, ctx.focus_tags,
                         ctx.tag_quota, ctx.tag_gaps)
    return errors, [] if errors else saved_items(items, ctx.offered, problems, profile,
                                                 ctx.difficulty, ctx.plan_focus())


def validate_plan(ctx: ToolContext, args: ValidatePlanInput) -> ValidationOut:
    constraints = effective_constraints(ctx.user_constraints, args.constraints)
    errors, _ = check_plan(ctx, args.items, constraints)
    ctx.validated_ok = not errors
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
    saved = SavedConstraints(**constraints.model_dump(), difficulty=ctx.difficulty)
    plan_id = store.save(ctx.handle, saved, items)
    ctx.saved_plan_id = plan_id
    ctx.saved_plan = (saved, items)
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
    plans: dict[int, tuple[SavedConstraints, list[SavedItem]]] = field(default_factory=dict)

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
        good = [PlanItem(id=p.id, reason=f"dp step {n}: below your other tags")
                for n, p in enumerate(found, start=1)]
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

        # The ladder: the model lists 3 dp problems hardest first; the saved order is easy first.
        ladder = find_problems(ctx, FindProblemsInput(tag="dp", count=3)).problems
        errors, items = check_plan(ctx, [PlanItem(id=p.id, reason=f"why {p.id}")
                                         for p in reversed(ladder)], PlanConstraints())
        print(f"\nmodel's order: {[(p.id, p.gap) for p in reversed(ladder)]}"
              f"\nsaved order:   {[(s.id, s.gap) for s in items]}  errors={errors}")
        for s in items:
            print(f"  {s.id}: {s.facts}")
        print(plan_summary(SavedConstraints(hours=3), items))
        ctx.difficulty = "harder"
        harder = find_problems(ctx, FindProblemsInput(tag="dp", count=3))
        print(f"harder: {[(p.id, p.rating, p.gap) for p in harder.problems]}")
        ctx.difficulty = "normal"

        if "--save" in sys.argv:
            print("\nsave_plan:", save_plan(ctx, good_plan))
            try:
                save_plan(ctx, good_plan)
            except ToolError as e:
                print("save_plan again:", e)
