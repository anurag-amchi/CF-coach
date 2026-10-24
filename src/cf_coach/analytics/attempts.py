"""Turn raw submissions into Attempts (one per problem) for the Elo model.

Two parts:
- build_attempts(): pure. Takes plain rows and a `now`, returns Attempts. Unit-tested.
- load_rows(): reads submissions JOIN problems for one handle from Postgres.

Run on real data:  uv run python -m cf_coach.analytics.attempts TitanCoder1258
Design reasons: docs/decisions.md #18 (what an attempt is), #19 (which verdicts count).
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from cf_coach.analytics.elo import Attempt, tag_skills
from cf_coach.db.models import Problem, Submission, User
from cf_coach.db.session import get_engine

GIVE_UP_AFTER = timedelta(days=10)
# Verdicts that say nothing about whether the idea was right, so they are not "wrong
# attempts": a compile error, a submission still being judged (None / TESTING), or one
# CF skipped (it was replaced by a later submission in the same contest).
NOT_COUNTED = {None, "TESTING", "COMPILATION_ERROR", "SKIPPED"}


@dataclass(frozen=True)
class SubmissionRow:
    """One submission joined with its problem: everything build_attempts needs."""
    contest_id: int
    idx: str
    verdict: str | None
    created_at: datetime
    rating: int | None
    tags: tuple[str, ...]
    participant_type: str = "PRACTICE"   # CONTESTANT, OUT_OF_COMPETITION, VIRTUAL, PRACTICE
    relative_time_s: int | None = None   # seconds into the contest; None for practice


def build_attempts(rows: list[SubmissionRow], now: datetime) -> list[Attempt]:
    """Group submissions by problem and make one Attempt per problem.

    - Solved: wrong_attempts = counted non-OK submissions BEFORE the first AC (later
      ones are ignored); ended_at = time of the first AC.
    - Not solved, last submission older than GIVE_UP_AFTER: given up.
    - Not solved, touched recently: still in progress, left out.
    - Not solved and nothing counted (only compile errors / still judging): left out.
    """
    by_problem: dict[tuple[int, str], list[SubmissionRow]] = {}
    for row in rows:
        by_problem.setdefault((row.contest_id, row.idx), []).append(row)

    attempts = []
    for subs in by_problem.values():
        subs.sort(key=lambda s: s.created_at)
        first = subs[0]

        wrong = 0
        solved_at = None
        for s in subs:
            if s.verdict == "OK":
                solved_at = s.created_at
                break
            if s.verdict not in NOT_COUNTED:
                wrong += 1

        if solved_at is not None:
            attempts.append(Attempt(first.rating, first.tags, True, wrong, solved_at))
            continue

        last_at = subs[-1].created_at
        if wrong == 0 or now - last_at <= GIVE_UP_AFTER:
            continue
        attempts.append(Attempt(first.rating, first.tags, False, wrong, last_at))
    return attempts


def load_rows(session: Session, handle: str) -> list[SubmissionRow]:
    """All submissions of `handle` whose problem is in the problemset.

    INNER JOIN on purpose: submissions on problems we don't have (gym etc.) have no
    rating or tags, so they can't be used for Elo anyway (decision #14).
    """
    stmt = (
        select(Submission.contest_id, Submission.idx, Submission.verdict,
               Submission.created_at, Problem.rating, Problem.tags,
               Submission.participant_type, Submission.relative_time_s)
        .join(Problem, (Problem.contest_id == Submission.contest_id)
                       & (Problem.idx == Submission.idx))
        .where(Submission.handle == handle)
    )
    return [SubmissionRow(c, i, v, t, r, tuple(tags), p, rel)
            for c, i, v, t, r, tags, p, rel in session.execute(stmt)]


# ---------- Manual check: per-tag skill for a real user ----------

if __name__ == "__main__":
    handle = sys.argv[1]
    with Session(get_engine()) as session:
        start = session.scalar(select(User.rating).where(User.handle == handle))
        rows = load_rows(session, handle)

    attempts = build_attempts(rows, now=datetime.now(timezone.utc))
    solved = sum(a.solved for a in attempts)
    print(f"{handle}: rating {start}, {len(rows)} submissions -> {len(attempts)} attempts "
          f"({solved} solved, {len(attempts) - solved} given up)\n")

    skills = tag_skills(attempts, start_rating=start)
    print(f"{'tag':<28}{'skill':>8}{'attempts':>10}")
    for tag, t in sorted(skills.items(), key=lambda kv: kv[1].skill):
        print(f"{tag:<28}{t.skill:>8.0f}{t.attempts:>10}")
