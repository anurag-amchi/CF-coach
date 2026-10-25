"""The eval cases: a request (plus optional --hours / --avoid) and what must be true after.

The expectations are the target behaviour from decision #33, not today's: the baseline
is expected to fail some of them. Gaps are relative to the rounded tag skill, so the
cases keep working as the user's skills change.
Bands (#33): normal = +0..+200 (default), harder = +100..+300, easier = -100..+100.
Every case also gets checks.UNIVERSAL (guardrails, tool use, summary format).
"""

from dataclasses import dataclass, field

from cf_coach.evals.checks import (
    Check, at_most, avoids, band, count, ladder, mentions_weakest_tag, no_plan, saved,
    saved_hours, tags_in,
)

NORMAL, HARDER, EASIER = (0, 200), (100, 300), (-100, 100)


@dataclass(frozen=True)
class Case:
    name: str
    request: str
    checks: list[Check]
    hours: int | None = None               # like --hours
    avoid: list[str] = field(default_factory=list)   # like --avoid


CASES = [
    Case("dp_upskill",
         "Give me 3 problems of DP according to my skill level so that I can upskill",
         [saved(), count(3), tags_in("dp"), ladder(0, 100, 200)]),
    Case("hours_in_text",
         "I have 4 hours this week. No geometry please.",
         [saved(), at_most(4), avoids("geometry"), saved_hours(4), band(*NORMAL)]),
    Case("hours_as_flags",
         "Plan my week.",
         [saved(), at_most(4), avoids("geometry"), saved_hours(4), band(*NORMAL)],
         hours=4, avoid=["geometry"]),
    Case("count_above_hours",
         "Give me 5 dp problems this week",
         [saved(), count(3), tags_in("dp"), band(*NORMAL)],
         hours=3),
    Case("default_week",
         "Plan my practice for this week.",
         [saved(), count(5), band(*NORMAL)]),
    Case("two_tags",
         "I want to practise greedy and binary search. 4 problems.",
         [saved(), count(4), tags_in("greedy", "binary search"), band(*NORMAL)]),
    Case("challenge",
         "Challenge me: 3 hard dp problems.",
         [saved(), count(3), tags_in("dp"), band(*HARDER)]),
    Case("rusty",
         "I'm rusty after a break. Give me 3 easy warm-up problems.",
         [saved(), count(3), band(*EASIER)]),
    Case("too_hard",
         "Give me 3 problems rated 2400.",
         [band(-100, 300)]),                # refusing is fine; planning 2400s is not
    Case("running_contest",
         "Give me problems from the contest that is running right now.",
         []),                               # the guardrail is in UNIVERSAL (no running contest)
    Case("weakest_tag_question",
         "What is my weakest tag right now?",
         [no_plan(), mentions_weakest_tag()]),
]
