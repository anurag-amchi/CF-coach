"""The agent's tools: thin wrappers over the analytics code.

Each tool is  fn(ctx, args) -> output model.
- ctx (ToolContext) is filled in by the loop: whose data, the DB session, "now".
- args is an input object from schemas.py, already checked by parse_args.
A tool raises ToolError when the model asked for something impossible (an unknown tag,
for example). The loop sends that message back to the model so it can try again.

Each tool = a database read + a pure *_out() function that builds the output. The pure
parts are unit-tested; the database parts are checked by running this file.

Built so far: get_tag_profile, get_upsolve_backlog, find_problems.

Run on real data:  uv run python -m cf_coach.agent.tools TitanCoder1258
Design reasons: docs/decisions.md #26.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from cf_coach.agent.schemas import (
    BacklogItemOut, BacklogOut, FindProblemsInput, FindProblemsOut, GetTagProfileInput,
    GetUpsolveBacklogInput, ProblemOut, TagProfileOut, TagSkillOut,
)
from cf_coach.analytics.attempts import load_rows
from cf_coach.analytics.selection import (
    IGNORED_TAGS, Candidate, ContestPrepStrategy, ProblemRow, WeaknessFirstStrategy, build_profile,
    load_problems, load_solved, running_contest_ids, select_candidates,
)
from cf_coach.analytics.signals import BacklogItem, upsolve_backlog
from cf_coach.analytics.tag_profile import start_rating
from cf_coach.db.models import TagSkill
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


# name -> function. The input class for each name is in schemas.TOOLS.
TOOL_FUNCTIONS = {
    "get_tag_profile": get_tag_profile,
    "get_upsolve_backlog": get_upsolve_backlog,
    "find_problems": find_problems,
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
    ]
    with Session(get_engine()) as session:
        ctx = ToolContext(handle, session, datetime.now(timezone.utc))
        for name, args in calls:
            try:
                out = TOOL_FUNCTIONS[name](ctx, args).model_dump_json()
                print(f"\n{name}({args!r})  [{len(out)} chars]\n{out}")
            except ToolError as e:
                print(f"\n{name}({args!r})  ToolError: {e}")
