from cf_coach.analytics.selection import (
    ContestPrepStrategy, Profile, ProblemRow, ScoringStrategy, WeaknessFirstStrategy,
    select_candidates,
)
from cf_coach.analytics.signals import TagSignals

PROFILE = Profile(rating=1400, skills={"dp": 1400.0, "greedy": 1600.0},
                  newest_contest_id=2000)


def prob(contest, idx, rating, tags=("dp",)):
    return ProblemRow(contest, idx, f"{contest}{idx}", rating, tuple(tags))


def keys(cands):
    return [f"{c.problem.contest_id}{c.problem.idx}" for c in cands]


def pick(problems, solved=(), running=(), strategy=None, **kw):
    return select_candidates(problems, set(solved), PROFILE, set(running),
                             strategy or WeaknessFirstStrategy(), **kw)


def test_band_is_skill_plus_100_to_300():
    problems = [prob(1990, "A", 1400), prob(1990, "B", 1500), prob(1990, "C", 1700),
                prob(1990, "D", 1800)]
    assert sorted(keys(pick(problems))) == ["1990B", "1990C"]


def test_solved_unrated_and_running_are_excluded():
    problems = [prob(1990, "A", 1600), prob(1991, "A", None), prob(1999, "A", 1600),
                prob(1992, "A", 1600)]
    out = pick(problems, solved={(1990, "A")}, running={1999})
    assert keys(out) == ["1992A"]


def test_excluded_tag_drops_problem_even_with_other_tags():
    out = pick([prob(1990, "A", 1600, ("dp", "geometry"))],
               excluded_tags=frozenset({"geometry"}))
    assert out == []


def test_multi_tag_problem_appears_once_with_its_best_tag():
    # 1800 is in the greedy band (1700-1900) but not in dp's (1500-1700)
    out = pick([prob(1990, "A", 1600, ("dp", "greedy")), prob(1990, "B", 1800, ("dp", "greedy"))])
    assert {keys([c])[0]: c.tag for c in out} == {"1990A": "dp", "1990B": "greedy"}


def test_tag_filter_and_untried_tag_uses_rating():
    problems = [prob(1990, "A", 1600, ("dp",)), prob(1990, "B", 1600, ("graphs",))]
    assert keys(pick(problems, tag="dp")) == ["1990A"]
    [c] = pick(problems, tag="graphs")            # never tried: skill = rating 1400
    assert c.tag_skill == 1400.0


def test_weakness_first_prefers_weaker_tag():
    problems = [prob(1990, "A", 1600, ("dp",)), prob(1990, "B", 1800, ("greedy",))]
    assert keys(pick(problems)) == ["1990A", "1990B"]


def test_contest_prep_prefers_recent_and_slow_tags():
    profile = Profile(1400, {"dp": 1400.0, "math": 1400.0},
                      {"dp": TagSignals(solve_minutes=[60.0]), "math": TagSignals(solve_minutes=[6.0])},
                      newest_contest_id=2000)
    def order(problems):
        return keys(select_candidates(problems, set(), profile, set(), ContestPrepStrategy()))

    # same tag: newer contest first
    assert order([prob(1800, "A", 1600, ("dp",)), prob(2000, "A", 1600, ("dp",))]) \
        == ["2000A", "1800A"]
    # same contest: the tag you're slow at (dp, 60 min) before the fast one (math, 6 min)
    assert order([prob(2000, "A", 1600, ("math",)), prob(2000, "B", 1600, ("dp",))]) \
        == ["2000B", "2000A"]


def test_any_strategy_can_be_plugged_in():
    class EasiestFirst(ScoringStrategy):
        name = "easiest_first"

        def score(self, problem, tag, profile):
            return -problem.rating

    problems = [prob(1990, "A", 1700), prob(1990, "B", 1500)]
    assert keys(pick(problems, strategy=EasiestFirst())) == ["1990B", "1990A"]


def test_count_limits_results():
    problems = [prob(1990, chr(65 + i), 1600) for i in range(5)]
    assert len(pick(problems, count=3)) == 3
