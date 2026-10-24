"""Elo-style skill per tag.

Every attempt is a "match" between the user's skill on a tag and the problem's rating:
    P(solve)  = 1 / (1 + 10 ** ((rating - skill) / 400))
    skill_new = skill + K * (outcome - P(solve))

- Solving a hard problem (low P) raises skill a lot; solving an easy one barely moves it.
- Failing an easy problem (high P) drops skill a lot; failing a hard one barely moves it.

This file is pure maths: no database, no clock. The caller builds the Attempt list.
Design reasons: docs/decisions.md #18.
"""

from dataclasses import dataclass
from datetime import datetime

K = 32                       # max change per attempt, shared by the problem's tags
MIN_SOLVED_OUTCOME = 0.5     # a solve never scores below this, however many WAs came first
WRONG_ATTEMPT_PENALTY = 0.1  # each wrong submission before AC takes 0.1 off a solve


@dataclass(frozen=True)
class Attempt:
    """One problem the user tried: first submission -> accepted, or given up."""
    rating: int | None        # None = unrated problem (skipped)
    tags: tuple[str, ...]
    solved: bool
    wrong_attempts: int       # wrong submissions before the first AC
    ended_at: datetime        # time of the AC, or of the last submission if given up


@dataclass
class TagElo:
    skill: float
    attempts: int


def p_solve(skill: float, rating: int) -> float:
    """Expected chance of solving a problem of `rating` with `skill` (0.0 to 1.0)."""
    return 1 / (1 + 10 ** ((rating - skill) / 400))


def outcome(attempt: Attempt) -> float:
    """1.0 = clean solve, down to 0.5 for a solve after many WAs, 0.0 = given up."""
    if not attempt.solved:
        return 0.0
    return max(MIN_SOLVED_OUTCOME, 1 - WRONG_ATTEMPT_PENALTY * attempt.wrong_attempts)


def tag_skills(attempts: list[Attempt], start_rating: int, k: float = K) -> dict[str, TagElo]:
    """Replay attempts in time order and return the final skill for every tag."""
    result: dict[str, TagElo] = {}
    for attempt in sorted(attempts, key=lambda a: a.ended_at):
        if attempt.rating is None or not attempt.tags:
            continue
        score = outcome(attempt)
        k_per_tag = k / len(attempt.tags)
        for tag in attempt.tags:
            if tag not in result:
                result[tag] = TagElo(skill=float(start_rating), attempts=0)
            t = result[tag]
            t.skill += k_per_tag * (score - p_solve(t.skill, attempt.rating))
            t.attempts += 1
    return result
