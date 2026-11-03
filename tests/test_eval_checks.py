from cf_coach.evals.cases import CASES
from cf_coach.evals.checks import (
    UNIVERSAL, PlannedProblem, RunView, at_most, avoids, band, count, distinct_reasons, ladder,
    mentions_weakest_tag, no_numbers_in_reasons, no_plan, no_rejected_calls, no_table, not_running, round_skill, saved,
    saved_hours, starts_with_profile, step_log, tags_in, trace, unsolved,
)


def problem(idx="A", rating=1500, tag="dp", tags=("dp",), skill=1415.0, reason=None,
            solved=False, running=False):
    return PlannedProblem(2000, idx, rating, tags, tag, reason or f"reason {idx}", skill,
                          solved, running)


def view(plan=None, outcome="saved", text="Done.", **kw):
    return RunView(outcome=outcome, final_text=text, plan=plan, **kw)


def test_round_skill_is_nearest_hundred_halves_up():
    assert [round_skill(s) for s in (1415, 1449.9, 1450, 1392, 1510)] == [1400, 1400, 1500, 1400, 1500]
    assert [round_skill(s) for s in (407, 749)] == [800, 800]      # floor: CF's easiest (#55)


def test_gap_is_from_the_rounded_skill():
    assert problem(rating=1600, skill=1415).gap == 200
    assert problem(rating=None).gap is None


def test_ladder_wants_the_gaps_in_order():
    plan = [problem("A", 1400), problem("B", 1500), problem("C", 1600)]
    assert ladder(0, 100, 200)(view(plan)) is None
    wrong = [problem("A", 1700), problem("B", 1700), problem("C", 1600)]   # plan 5
    assert ladder(0, 100, 200)(view(wrong)) == "gaps [300, 300, 200], want [0, 100, 200]"


def test_band_names_the_problems_outside_it():
    plan = [problem("A", 1500), problem("B", 1700), problem("C", None)]
    assert band(0, 200)(view(plan)) == "outside: 2000B +300, 2000C unrated"


def test_plan_checks_pass_when_no_plan_was_saved():
    refused = view(plan=None, outcome="no_plan")
    for check in (count(3), band(0, 200), tags_in("dp"), avoids("geometry"), unsolved()):
        assert check(refused) is None
    assert saved()(refused) == "outcome no_plan"
    assert no_plan()(refused) is None


def test_count_tags_and_avoids():
    plan = [problem("A"), problem("B", tag="trees", tags=("trees", "geometry"))]
    v = view(plan)
    assert count(2)(v) is None and at_most(1)(v) == "got 2"
    assert tags_in("dp")(v) == "other tags: 2000B [trees]"
    assert avoids("geometry")(v) == "has avoided tags: 2000B ['geometry']"


def test_guardrails():
    plan = [problem("A", solved=True), problem("B", running=True)]
    assert unsolved()(view(plan)) == "already solved: 2000A"
    assert not_running()(view(plan)) == "running: 2000B"


def test_reasons_summary_and_hours():
    same = [problem("A", reason="Your dp skill is 1415."), problem("B", reason="your DP skill is 1415. ")]
    assert distinct_reasons()(view(same)) == "some reasons are the same"
    assert no_table()(view(text="| a | b |\n|---|---|")) == "summary has a table"
    assert no_table()(view(text="Three dp problems, easy to hard.")) is None
    assert saved_hours(4)(view([problem()], saved_hours=None)) == "saved hours = None"


def test_numbers_in_reasons_fail_but_digits_in_tag_names_dont():
    plan = [problem("A", reason="Fits the +0 gap requirement."),
            problem("B", reason="A 2-sat warm-up for your graph work.", tags=("2-sat", "graphs")),
            problem("C", reason="Trains counting on small cases.")]
    assert no_numbers_in_reasons()(view(plan)) == \
        'numbers in: 2000A "Fits the +0 gap requirement."'
    assert no_numbers_in_reasons()(view(plan[1:])) is None


def test_trace_counts_tool_calls_and_rejections():
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "plan"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "get_tag_profile", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "user", "content": "Your last tool call was rejected and not run: ..."},
        {"role": "assistant", "content": "done"},
    ]
    assert trace(messages) == (["get_tag_profile"], 1)
    assert step_log(messages) == [
        {"user": "plan"},
        {"call": "get_tag_profile", "args": "{}"},
        {"result": "{}"},
        {"user": "Your last tool call was rejected and not run: ..."},
        {"text": "done"},
    ]


def test_tool_use_and_question_checks():
    v = view(None, outcome="no_plan", text="Geometry is your weakest tag.",
             tools_called=["find_problems"], rejected_calls=2, weakest_tag="geometry")
    assert starts_with_profile()(v) == "first call: find_problems"
    assert no_rejected_calls()(v) == "2 rejected"
    assert mentions_weakest_tag()(v) is None


def test_case_names_are_unique_and_every_case_gets_the_universal_checks():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))
    assert {c.name for c in UNIVERSAL} >= {"no rejected calls", "no running contest"}
