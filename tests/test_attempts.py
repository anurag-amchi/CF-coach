from datetime import datetime, timedelta

from cf_coach.analytics.attempts import SubmissionRow, build_attempts

NOW = datetime(2026, 10, 24)


def sub(verdict, days_ago, contest=2000, idx="A", rating=1400, tags=("dp",)):
    return SubmissionRow(contest, idx, verdict, NOW - timedelta(days=days_ago), rating, tags)


def test_clean_solve():
    [a] = build_attempts([sub("OK", 30)], NOW)
    assert a.solved and a.wrong_attempts == 0
    assert a.ended_at == NOW - timedelta(days=30)


def test_counts_only_wrong_attempts_before_first_ac():
    rows = [sub("WRONG_ANSWER", 30), sub("TIME_LIMIT_EXCEEDED", 29), sub("OK", 28),
            sub("WRONG_ANSWER", 27)]          # after the AC: ignored
    [a] = build_attempts(rows, NOW)
    assert a.solved and a.wrong_attempts == 2
    assert a.ended_at == NOW - timedelta(days=28)   # first AC, not the last submission


def test_input_order_does_not_matter():
    rows = [sub("OK", 28), sub("WRONG_ANSWER", 30)]
    [a] = build_attempts(rows, NOW)
    assert a.wrong_attempts == 1


def test_compile_error_and_skipped_are_not_wrong():
    rows = [sub("COMPILATION_ERROR", 30), sub("SKIPPED", 29), sub("OK", 28)]
    [a] = build_attempts(rows, NOW)
    assert a.wrong_attempts == 0


def test_given_up_after_10_days():
    [a] = build_attempts([sub("WRONG_ANSWER", 20), sub("WRONG_ANSWER", 11)], NOW)
    assert not a.solved and a.wrong_attempts == 2
    assert a.ended_at == NOW - timedelta(days=11)   # last submission


def test_recent_unsolved_is_in_progress():
    # first WA long ago, but touched 3 days ago -> still in progress
    assert build_attempts([sub("WRONG_ANSWER", 40), sub("WRONG_ANSWER", 3)], NOW) == []


def test_only_compile_errors_is_left_out():
    assert build_attempts([sub("COMPILATION_ERROR", 40)], NOW) == []


def test_one_attempt_per_problem():
    rows = [sub("OK", 30, idx="A"), sub("WRONG_ANSWER", 30, idx="B"),
            sub("OK", 29, idx="B"), sub("OK", 30, contest=2001, idx="A")]
    attempts = build_attempts(rows, NOW)
    assert len(attempts) == 3
    assert sorted(a.wrong_attempts for a in attempts) == [0, 0, 1]
