"""Candidate selection (plan §4.2c): which unsolved problems to practise next.

A problem is a candidate for one of its tags when:
- it is rated, not solved yet, and not from a contest that is still running (guardrail),
- it has none of the excluded tags,
- its rating is in the band [tag skill + 100, tag skill + 300].

How candidates are ranked is a Strategy: an object with a score() method. The caller
picks one (WeaknessFirstStrategy, ContestPrepStrategy) and select_candidates() works
the same with any of them, so a new ranking never changes the calling code.

Run on real data:  uv run python -m cf_coach.analytics.selection TitanCoder1258
Design reasons: docs/decisions.md #21.
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from cf_coach.analytics.attempts import build_attempts, load_rows
from cf_coach.analytics.elo import tag_skills
from cf_coach.analytics.signals import TagSignals, tag_signals
from cf_coach.db.models import Problem, Submission, User
from cf_coach.db.session import get_engine

BAND = (100, 300)              # practise 100 to 300 above your skill on the tag
RECENCY_WINDOW = 500           # contest ids; ~3.5 years of CF rounds
IGNORED_TAGS = {"*special"}    # a CF marker (April Fools etc.), not a topic


@dataclass(frozen=True)
class ProblemRow:
    contest_id: int
    idx: str
    name: str
    rating: int | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    """What the strategies may look at about the user."""
    rating: int
    skills: dict[str, float]
    signals: dict[str, TagSignals] = field(default_factory=dict)
    newest_contest_id: int = 0


@dataclass(frozen=True)
class Candidate:
    problem: ProblemRow
    tag: str                   # the tag whose band it falls in
    tag_skill: float
    score: float


def recency(problem: ProblemRow, profile: Profile) -> float:
    """1.0 for the newest contest, falling to 0.0 at RECENCY_WINDOW contests older.

    CF contest ids grow over time, so the id is used as the date (no extra API call).
    """
    age = profile.newest_contest_id - problem.contest_id
    return max(0.0, 1 - age / RECENCY_WINDOW)


# ---------- Strategies ----------

class ScoringStrategy(ABC):
    """Higher score = recommend first. Every strategy must implement score()."""
    name: str

    @abstractmethod
    def score(self, problem: ProblemRow, tag: str, profile: Profile) -> float: ...


class WeaknessFirstStrategy(ScoringStrategy):
    """Your weakest tags first: low skill and high abandon rate. Recency breaks ties."""
    name = "weakness_first"

    def score(self, problem, tag, profile):
        weakness = (max(profile.skills.values()) - profile.skills.get(tag, profile.rating)) / 100
        s = profile.signals.get(tag)
        abandon = s.abandon_rate if s else 0.0
        return weakness + 3 * abandon + 0.5 * recency(problem, profile)


class ContestPrepStrategy(ScoringStrategy):
    """Recent contest problems, in tags where you are slow during contests."""
    name = "contest_prep"

    def score(self, problem, tag, profile):
        s = profile.signals.get(tag)
        slow = (s.median_solve_min / 30) if s and s.median_solve_min else 0.0
        return 2 * recency(problem, profile) + slow


# ---------- Selection ----------

def select_candidates(
    problems: list[ProblemRow],
    solved: set[tuple[int, str]],
    profile: Profile,
    running_contests: set[int],
    strategy: ScoringStrategy,
    tag: str | None = None,
    excluded_tags: frozenset[str] = frozenset(),
    count: int = 10,
) -> list[Candidate]:
    """Top `count` candidates, best first, at most one entry per problem.

    tag=None: any tag the user has a skill for. tag="dp": only dp; if the user has never
    tried dp, their CF rating is used as the dp skill.
    """
    best: dict[tuple[int, str], Candidate] = {}
    for p in problems:
        key = (p.contest_id, p.idx)
        if p.rating is None or key in solved or p.contest_id in running_contests:
            continue
        if excluded_tags & set(p.tags):
            continue
        for t in p.tags:
            if t in IGNORED_TAGS or (tag is not None and t != tag):
                continue
            if t in profile.skills:
                skill = profile.skills[t]
            elif t == tag:
                skill = float(profile.rating)
            else:
                continue
            if not (skill + BAND[0] <= p.rating <= skill + BAND[1]):
                continue
            c = Candidate(p, t, skill, strategy.score(p, t, profile))
            if key not in best or c.score > best[key].score:
                best[key] = c
    ranked = sorted(best.values(),
                    key=lambda c: (-c.score, -c.problem.contest_id, c.problem.idx))
    return ranked[:count]


# ---------- Loading from Postgres / Codeforces ----------

def load_problems(session: Session) -> list[ProblemRow]:
    rows = session.execute(select(Problem.contest_id, Problem.idx, Problem.name,
                                  Problem.rating, Problem.tags))
    return [ProblemRow(c, i, n, r, tuple(t)) for c, i, n, r, t in rows]


def load_solved(session: Session, handle: str) -> set[tuple[int, str]]:
    rows = session.execute(select(Submission.contest_id, Submission.idx).distinct()
                           .where(Submission.handle == handle, Submission.verdict == "OK"))
    return {(c, i) for c, i in rows}


def build_profile(session: Session, handle: str, now: datetime) -> Profile:
    rating = session.scalar(select(User.rating).where(User.handle == handle))
    rows = load_rows(session, handle)
    skills = {t: e.skill for t, e in tag_skills(build_attempts(rows, now), rating).items()}
    newest = session.scalar(select(Problem.contest_id).order_by(Problem.contest_id.desc()).limit(1))
    return Profile(rating, skills, tag_signals(rows, now), newest)


def running_contest_ids() -> set[int]:
    """Contests not FINISHED yet (BEFORE, CODING, SYSTEM_TEST...). One CF API call."""
    from cf_coach.sync.cf_client import get_contests   # needs Redis; import only when used
    return {c["id"] for c in get_contests() if c["phase"] != "FINISHED"}


# ---------- Manual check ----------

if __name__ == "__main__":
    handle = sys.argv[1]
    with Session(get_engine()) as session:
        profile = build_profile(session, handle, datetime.now(timezone.utc))
        problems = load_problems(session)
        solved = load_solved(session, handle)
    running = running_contest_ids()
    print(f"{len(problems)} problems, {len(solved)} solved, running contests: {sorted(running)}")

    def show(title, cands):
        print(f"\n{title}")
        for c in cands:
            p = c.problem
            print(f"  {c.score:5.2f}  {p.contest_id}{p.idx:<4} {p.rating}  "
                  f"[{c.tag} {c.tag_skill:.0f}]  {p.name}")

    for strategy in (WeaknessFirstStrategy(), ContestPrepStrategy()):
        show(strategy.name, select_candidates(problems, solved, profile, running, strategy))
    show("weakness_first, tag=dp", select_candidates(
        problems, solved, profile, running, WeaknessFirstStrategy(), tag="dp", count=5))
    show("weakness_first, excluding geometry", select_candidates(
        problems, solved, profile, running, WeaknessFirstStrategy(),
        excluded_tags=frozenset({"geometry"}), count=5))
