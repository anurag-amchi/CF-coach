import random
from datetime import datetime, timedelta

import pytest

from cf_coach.analytics.elo import Attempt, outcome, p_solve, tag_skills

T0 = datetime(2026, 1, 1)


def make(rating, tags, solved=True, wrong=0, day=0):
    return Attempt(rating, tuple(tags), solved, wrong, T0 + timedelta(days=day))


def test_p_solve_equal_is_half():
    assert p_solve(1400, 1400) == 0.5


def test_p_solve_400_harder_is_one_in_eleven():
    assert p_solve(1400, 1800) == pytest.approx(1 / 11)


@pytest.mark.parametrize("solved, wrong, expected", [
    (True, 0, 1.0),
    (True, 2, 0.8),
    (True, 10, 0.5),   # floor
    (False, 3, 0.0),
])
def test_outcome(solved, wrong, expected):
    assert outcome(make(1400, ["dp"], solved, wrong)) == pytest.approx(expected)


def test_clean_solve_at_equal_rating_gains_half_k():
    skills = tag_skills([make(1400, ["dp"])], start_rating=1400)
    assert skills["dp"].skill == pytest.approx(1416)   # 1400 + 32 * (1 - 0.5)
    assert skills["dp"].attempts == 1


def test_given_up_loses_half_k():
    skills = tag_skills([make(1400, ["dp"], solved=False)], start_rating=1400)
    assert skills["dp"].skill == pytest.approx(1384)


def test_k_is_split_across_tags():
    skills = tag_skills([make(1400, ["dp", "greedy"])], start_rating=1400)
    assert skills["dp"].skill == pytest.approx(1408)
    assert skills["greedy"].skill == pytest.approx(1408)


def test_unrated_and_untagged_are_skipped():
    skills = tag_skills([make(None, ["dp"]), make(1400, [])], start_rating=1400)
    assert skills == {}


def test_input_order_does_not_matter():
    attempts = [make(1200 + 100 * i, ["dp"], solved=i % 2 == 0, day=i) for i in range(10)]
    shuffled = attempts[:]
    random.Random(0).shuffle(shuffled)
    assert tag_skills(shuffled, 1400) == tag_skills(attempts, 1400)
