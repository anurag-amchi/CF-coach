"""Behavioural signals per tag (plan §4.2b). Pure: rows in, numbers out.

- abandon rate:        given-up attempts / attempts (given up = WA, never AC, > 10 days)
- avg wrong before AC: over solved attempts
- upsolve backlog:     problems submitted to during a contest and never accepted since
- contest solve time:  minutes spent per problem solved in a contest

Run on real data:  uv run python -m cf_coach.analytics.signals TitanCoder1258
Design reasons: docs/decisions.md #20.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median

from sqlalchemy.orm import Session

from cf_coach.analytics.attempts import NOT_COUNTED, SubmissionRow, build_attempts, load_rows
from cf_coach.db.session import get_engine

# Submissions made while a contest clock was running (a virtual contest counts too).
CONTEST_TYPES = {"CONTESTANT", "OUT_OF_COMPETITION", "VIRTUAL"}


@dataclass(frozen=True)
class BacklogItem:
    """A contest problem the user tried and still hasn't solved."""
    contest_id: int
    idx: str
    rating: int | None
    tags: tuple[str, ...]
    last_tried_at: datetime


@dataclass
class TagSignals:
    attempts: int = 0
    solved: int = 0
    given_up: int = 0
    wrong_before_ac: int = 0                  # summed over solved attempts
    backlog: int = 0
    solve_minutes: list[float] = field(default_factory=list)

    @property
    def abandon_rate(self) -> float:
        return self.given_up / self.attempts if self.attempts else 0.0

    @property
    def avg_wrong_before_ac(self) -> float | None:
        return self.wrong_before_ac / self.solved if self.solved else None

    @property
    def median_solve_min(self) -> float | None:
        return median(self.solve_minutes) if self.solve_minutes else None


def upsolve_backlog(rows: list[SubmissionRow]) -> list[BacklogItem]:
    """Problems with a judged contest submission and no AC ever (in any mode). Newest first.

    Judged = not in NOT_COUNTED. A contest problem with only SKIPPED or compile-error
    submissions has an unknown result, so it is not put in the backlog.
    """
    solved = {(r.contest_id, r.idx) for r in rows if r.verdict == "OK"}
    last_try: dict[tuple[int, str], SubmissionRow] = {}
    for r in rows:
        key = (r.contest_id, r.idx)
        if (r.participant_type not in CONTEST_TYPES or key in solved
                or r.verdict in NOT_COUNTED):
            continue
        if key not in last_try or r.created_at > last_try[key].created_at:
            last_try[key] = r
    items = [BacklogItem(r.contest_id, r.idx, r.rating, r.tags, r.created_at)
             for r in last_try.values()]
    return sorted(items, key=lambda b: b.last_tried_at, reverse=True)


def contest_solve_times(rows: list[SubmissionRow]) -> list[tuple[tuple[str, ...], float]]:
    """(tags, minutes) for every problem solved during a contest.

    relative_time_s is time since the contest started, so for a D it includes A to C.
    Time spent on a problem = its first AC minus the previous AC in the same contest
    (or minus the start, for the first solve).
    """
    first_ac: dict[tuple[int, str], SubmissionRow] = {}
    for r in rows:
        if (r.verdict != "OK" or r.participant_type not in CONTEST_TYPES
                or r.relative_time_s is None):
            continue
        key = (r.contest_id, r.idx)
        if key not in first_ac or r.relative_time_s < first_ac[key].relative_time_s:
            first_ac[key] = r

    by_contest: dict[int, list[SubmissionRow]] = {}
    for r in first_ac.values():
        by_contest.setdefault(r.contest_id, []).append(r)

    result = []
    for solves in by_contest.values():
        solves.sort(key=lambda r: r.relative_time_s)
        prev = 0
        for r in solves:
            result.append((r.tags, (r.relative_time_s - prev) / 60))
            prev = r.relative_time_s
    return result


def tag_signals(rows: list[SubmissionRow], now: datetime) -> dict[str, TagSignals]:
    """All four signals, per tag."""
    result: dict[str, TagSignals] = {}

    for a in build_attempts(rows, now):
        if a.rating is None:      # same attempts as Elo, so the counts match tag_skills
            continue
        for tag in a.tags:
            s = result.setdefault(tag, TagSignals())
            s.attempts += 1
            if a.solved:
                s.solved += 1
                s.wrong_before_ac += a.wrong_attempts
            else:
                s.given_up += 1

    for item in upsolve_backlog(rows):
        for tag in item.tags:
            result.setdefault(tag, TagSignals()).backlog += 1

    for tags, minutes in contest_solve_times(rows):
        for tag in tags:
            result.setdefault(tag, TagSignals()).solve_minutes.append(minutes)

    return result


# ---------- Manual check: signals for a real user ----------

if __name__ == "__main__":
    handle = sys.argv[1]
    with Session(get_engine()) as session:
        rows = load_rows(session, handle)
    now = datetime.now(timezone.utc)

    backlog = upsolve_backlog(rows)
    print(f"Upsolve backlog: {len(backlog)} problems")
    for b in backlog:
        print(f"  {b.contest_id}{b.idx:<4} rating {b.rating}  last tried "
              f"{b.last_tried_at:%Y-%m-%d}  {', '.join(b.tags)}")

    def fmt(x):
        return "-" if x is None else f"{x:.1f}"

    print(f"\n{'tag':<26}{'att':>5}{'abandon':>9}{'avg WA':>8}{'backlog':>9}{'solve min':>11}")
    signals = tag_signals(rows, now)
    for tag, s in sorted(signals.items(), key=lambda kv: -kv[1].attempts):
        print(f"{tag:<26}{s.attempts:>5}{s.abandon_rate:>9.0%}{fmt(s.avg_wrong_before_ac):>8}"
              f"{s.backlog:>9}{fmt(s.median_solve_min):>11}")
