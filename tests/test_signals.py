from datetime import datetime, timedelta

import pytest

from cf_coach.analytics.attempts import SubmissionRow
from cf_coach.analytics.signals import contest_solve_times, tag_signals, upsolve_backlog

NOW = datetime(2026, 10, 24)


def sub(verdict, days_ago=30, contest=2000, idx="A", rating=1400, tags=("dp",),
        ptype="PRACTICE", rel_min=None):
    rel = None if rel_min is None else rel_min * 60
    return SubmissionRow(contest, idx, verdict, NOW - timedelta(days=days_ago), rating,
                         tags, ptype, rel)


# ---- upsolve backlog ----

def test_backlog_has_unsolved_contest_problem():
    [b] = upsolve_backlog([sub("WRONG_ANSWER", ptype="CONTESTANT", rel_min=50)])
    assert (b.contest_id, b.idx) == (2000, "A")


def test_backlog_drops_problem_upsolved_later_in_practice():
    rows = [sub("WRONG_ANSWER", 30, ptype="CONTESTANT", rel_min=50), sub("OK", 20)]
    assert upsolve_backlog(rows) == []


def test_backlog_ignores_practice_only_problems():
    assert upsolve_backlog([sub("WRONG_ANSWER")]) == []


def test_backlog_ignores_contest_problems_with_only_skipped_or_ce():
    rows = [sub("SKIPPED", ptype="OUT_OF_COMPETITION", rel_min=20),
            sub("COMPILATION_ERROR", idx="B", ptype="CONTESTANT", rel_min=30)]
    assert upsolve_backlog(rows) == []


def test_backlog_is_newest_first():
    rows = [sub("WRONG_ANSWER", 30, idx="C", ptype="CONTESTANT", rel_min=90),
            sub("WRONG_ANSWER", 5, contest=2001, idx="D", ptype="VIRTUAL", rel_min=80)]
    assert [b.idx for b in upsolve_backlog(rows)] == ["D", "C"]


# ---- contest solve times ----

def times(rows):
    """{problem index: minutes} from contest_solve_times (tags identify the problem)."""
    return {tags[0]: m for tags, m in contest_solve_times(rows)}


def test_solve_time_adds_up_gaps_before_own_submissions():
    rows = [sub("OK", idx="A", tags=("A",), ptype="CONTESTANT", rel_min=10),
            sub("WRONG_ANSWER", idx="B", tags=("B",), ptype="CONTESTANT", rel_min=20),
            sub("OK", idx="B", tags=("B",), ptype="CONTESTANT", rel_min=35)]
    assert times(rows) == {"A": 10, "B": 25}      # B: (20-10) + (35-20)


def test_time_between_switching_problems_goes_to_the_right_problem():
    # WA on A at 10, AC on B at 25, AC on A at 40
    rows = [sub("WRONG_ANSWER", idx="A", tags=("A",), ptype="CONTESTANT", rel_min=10),
            sub("OK", idx="B", tags=("B",), ptype="CONTESTANT", rel_min=25),
            sub("OK", idx="A", tags=("A",), ptype="CONTESTANT", rel_min=40)]
    assert times(rows) == {"B": 15, "A": 25}      # A = 40 - 15 spent on B


def test_time_on_an_unsolved_problem_is_not_given_to_the_next_solve():
    rows = [sub("OK", idx="A", tags=("A",), ptype="CONTESTANT", rel_min=10),
            sub("WRONG_ANSWER", idx="C", tags=("C",), ptype="CONTESTANT", rel_min=30),
            sub("OK", idx="B", tags=("B",), ptype="CONTESTANT", rel_min=50)]
    assert times(rows) == {"A": 10, "B": 20}      # C's 20 minutes are lost, not B's


def test_solve_time_uses_first_ac_and_ignores_practice():
    rows = [sub("OK", idx="A", ptype="CONTESTANT", rel_min=10),
            sub("OK", idx="A", ptype="CONTESTANT", rel_min=40),   # resubmit: ignored
            sub("OK", idx="B")]                                    # practice: ignored
    assert [m for _, m in contest_solve_times(rows)] == [10]


def test_solve_times_restart_per_contest():
    rows = [sub("OK", contest=2000, ptype="CONTESTANT", rel_min=10),
            sub("OK", contest=2001, ptype="CONTESTANT", rel_min=12)]
    assert sorted(m for _, m in contest_solve_times(rows)) == [10, 12]


# ---- per-tag summary ----

def test_tag_signals_abandon_rate_and_avg_wrong():
    rows = [sub("WRONG_ANSWER", 30, idx="A"), sub("OK", 29, idx="A"),   # solved, 1 WA
            sub("OK", 30, idx="B"),                                    # solved, 0 WA
            sub("WRONG_ANSWER", 30, idx="C"),                          # given up
            sub("WRONG_ANSWER", 30, idx="D")]                          # given up
    s = tag_signals(rows, NOW)["dp"]
    assert s.attempts == 4
    assert s.abandon_rate == pytest.approx(0.5)
    assert s.avg_wrong_before_ac == pytest.approx(0.5)
    assert s.median_solve_min is None


def test_tag_signals_skip_unrated_attempts():
    rows = [sub("WRONG_ANSWER", 30, rating=None)]
    assert "dp" not in tag_signals(rows, NOW)
