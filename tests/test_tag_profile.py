from datetime import datetime, timedelta

import pytest

from cf_coach.analytics.attempts import SubmissionRow, build_attempts
from cf_coach.analytics.elo import tag_skills
from cf_coach.analytics.signals import tag_signals
from cf_coach.analytics.tag_profile import build_tag_rows

NOW = datetime(2026, 10, 24)


def sub(verdict, days_ago, idx, rating=1400, tags=("dp",)):
    return SubmissionRow(2000, idx, verdict, NOW - timedelta(days=days_ago), rating, tags)


ROWS = [
    sub("OK", 30, "A", tags=("dp", "math")),
    sub("WRONG_ANSWER", 30, "B"),                 # dp given up
    sub("WRONG_ANSWER", 30, "C", rating=None),    # unrated: no Elo, no signals
    sub("WRONG_ANSWER", 2, "D", tags=("graphs",)),  # in progress: no attempt yet
]


def rows_for(rows):
    skills = tag_skills(build_attempts(rows, NOW), start_rating=1400)
    return build_tag_rows("me", skills, tag_signals(rows, NOW))


def test_one_row_per_tag_with_rated_attempts():
    assert [r["tag"] for r in rows_for(ROWS)] == ["dp", "math"]   # no graphs


def test_row_values_come_from_elo_and_signals():
    dp = rows_for(ROWS)[0]
    assert dp["handle"] == "me"
    assert dp["attempts"] == 2
    assert dp["abandon_rate"] == pytest.approx(0.5)


def test_elo_and_signals_count_the_same_attempts():
    skills = tag_skills(build_attempts(ROWS, NOW), start_rating=1400)
    signals = tag_signals(ROWS, NOW)
    for tag, elo in skills.items():
        assert signals[tag].attempts == elo.attempts


def test_no_attempts_gives_no_rows():
    assert rows_for([]) == []
