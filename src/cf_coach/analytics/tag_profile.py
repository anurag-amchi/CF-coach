"""Compute a user's per-tag profile and save it to the tag_skills table.

    submissions JOIN problems -> attempts -> Elo skill + signals -> tag_skills rows

The agent's get_tag_profile tool and the dashboard read this table instead of
recomputing everything on each request.

Run:  uv run python -m cf_coach.analytics.tag_profile TitanCoder1258
Design reasons: docs/decisions.md #22.
"""

import sys
from datetime import datetime, timezone

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from cf_coach.analytics.attempts import build_attempts, load_rows
from cf_coach.analytics.elo import TagElo, tag_skills
from cf_coach.analytics.signals import TagSignals, tag_signals
from cf_coach.db.models import TagSkill, User
from cf_coach.db.session import get_engine

# Unrated users have no CF rating to start every tag from. 1400 is where Codeforces
# itself starts a new account's hidden rating.
DEFAULT_START_RATING = 1400


def start_rating(session: Session, handle: str) -> int:
    user = session.get(User, handle)
    if user is None:
        raise LookupError(f"{handle} has not been synced yet")
    return user.rating if user.rating is not None else DEFAULT_START_RATING


def build_tag_rows(handle: str, skills: dict[str, TagElo],
                   signals: dict[str, TagSignals]) -> list[dict]:
    """One row per tag that has an Elo skill (tags with no rated attempts are left out)."""
    rows = []
    for tag, elo in sorted(skills.items()):
        s = signals.get(tag)
        rows.append({"handle": handle, "tag": tag, "skill": elo.skill,
                     "attempts": elo.attempts, "abandon_rate": s.abandon_rate if s else 0.0})
    return rows


def refresh_tag_skills(handle: str, now: datetime) -> int:
    """Recompute and replace all of `handle`'s tag_skills rows. Returns the row count.

    Delete + insert in ONE transaction: anyone reading sees either the old set or the
    new one, never a half-written mix, and tags that disappeared are removed.
    """
    with Session(get_engine()) as session, session.begin():
        rating = start_rating(session, handle)
        rows = load_rows(session, handle)
        skills = tag_skills(build_attempts(rows, now), rating)
        tag_rows = build_tag_rows(handle, skills, tag_signals(rows, now))

        session.execute(delete(TagSkill).where(TagSkill.handle == handle))
        if tag_rows:
            session.execute(insert(TagSkill), tag_rows)
    return len(tag_rows)


if __name__ == "__main__":
    handle = sys.argv[1]
    n = refresh_tag_skills(handle, datetime.now(timezone.utc))
    print(f"saved {n} tag_skills rows for {handle}")
