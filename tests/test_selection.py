from cf_coach.analytics.selection import (
    Candidate, ContestPrepStrategy, Profile, ProblemRow, ScoringStrategy, WeaknessFirstStrategy,
    band_for, ladder_order, round_skill, select_candidates,
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


def test_skill_rounds_to_the_nearest_100_halves_up():
    assert [round_skill(s) for s in (1415, 1449.9, 1450, 1351, 1349)] == [1400, 1400, 1500,
                                                                         1400, 1300]


def test_bands_by_difficulty():
    assert band_for(1415) == (1400, 1600)              # normal: your level to +200
    assert band_for(1415, "harder") == (1500, 1700)
    assert band_for(1415, "easier") == (1300, 1500)
    assert band_for(1450) == (1500, 1700)              # 1450 rounds up to 1500


def test_level_never_goes_below_the_easiest_cf_rating():
    # rehuMachli (rating 400, skill ~407): a 400-600 band held no problems at all (#55)
    assert [round_skill(s) for s in (407, 0, 749, 750, 851)] == [800, 800, 800, 800, 900]
    assert band_for(407) == (800, 1000)
    assert band_for(407, "harder") == (900, 1100)
    assert band_for(407, "easier") == (800, 900)       # not 700-900: nothing is rated 700


def test_a_beginner_gets_800_problems_at_gap_0():
    beginner = Profile(rating=400, skills={"dp": 407.0}, newest_contest_id=2000)
    problems = [prob(1990, "A", 800), prob(1990, "B", 1000), prob(1990, "C", 1100)]
    cands = select_candidates(problems, set(), beginner, set(), WeaknessFirstStrategy())
    assert sorted(keys(cands)) == ["1990A", "1990B"]
    assert sorted(c.gap for c in cands) == [0, 200]    # 800 is the warm-up, not "+400"


def test_normal_band_is_level_to_plus_200():
    problems = [prob(1990, "A", 1300), prob(1990, "B", 1400), prob(1990, "C", 1600),
                prob(1990, "D", 1700)]
    assert sorted(keys(pick(problems))) == ["1990B", "1990C"]     # dp 1400: 1400-1600
    assert sorted(keys(pick(problems, difficulty="harder"))) == ["1990C", "1990D"]
    assert sorted(keys(pick(problems, difficulty="easier"))) == ["1990A", "1990B"]


def cand(idx, rating, score):
    return Candidate(prob(1990, idx, rating), "dp", 1400.0, score)


def test_ladder_order_is_round_robin_easiest_level_first():
    ranked = [cand("A", 1600, 9), cand("B", 1400, 8), cand("C", 1400, 7), cand("D", 1500, 6),
              cand("E", 1600, 5)]                          # best first
    assert keys(ladder_order(ranked, 10)) == ["1990B", "1990D", "1990A", "1990C", "1990E"]
    assert [c.gap for c in ladder_order(ranked, 3)] == [0, 100, 200]


def test_results_come_in_ladder_order():
    problems = [prob(1990, "A", 1600), prob(1990, "B", 1600), prob(1990, "C", 1400),
                prob(1990, "D", 1500)]
    assert [c.gap for c in pick(problems)] == [0, 100, 200, 200]


def test_weak_other_tags_beat_a_newer_contest():
    profile = Profile(1400, {"dp": 1400.0, "greedy": 1600.0, "strings": 1300.0},
                      newest_contest_id=2000)
    problems = [prob(2000, "A", 1500, ("dp", "greedy")),     # newest, other tag strong
                prob(1990, "A", 1500, ("dp", "strings"))]    # older, other tag weak
    out = select_candidates(problems, set(), profile, set(), WeaknessFirstStrategy(), tag="dp")
    assert keys(out) == ["1990A", "2000A"]


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
    # 1800 is in the greedy band (1600-1800) but not in dp's (1400-1600)
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
    class OldestFirst(ScoringStrategy):
        name = "oldest_first"

        def score(self, problem, tag, profile):
            return -problem.contest_id

    problems = [prob(1990, "A", 1500), prob(1980, "A", 1500)]     # same level
    assert keys(pick(problems)) == ["1990A", "1980A"]                     # newer first
    assert keys(pick(problems, strategy=OldestFirst())) == ["1980A", "1990A"]


def test_count_limits_results():
    problems = [prob(1990, chr(65 + i), 1600) for i in range(5)]
    assert len(pick(problems, count=3)) == 3
