from datetime import date, datetime, timezone
from typing import get_args

import pytest

import cf_coach.agent.tools as tools
from cf_coach.agent.schemas import (
    TOOLS, FindProblemsInput, GetProgressSinceInput, PlanConstraints, PlanItem, SavePlanInput,
    ValidatePlanInput,
)
from cf_coach.agent.tools import (
    STRATEGIES, TOOL_FUNCTIONS, MemoryPlanStore, ToolContext, ToolError, all_tags, backlog_out,
    check_tags, effective_constraints, find_problems, first_solves_since, get_progress_since,
    plan_errors, problem_out, profile_out, progress_out, save_plan, user_constraints,
    validate_plan,
)
from cf_coach.analytics.attempts import SubmissionRow
from cf_coach.analytics.selection import Candidate, ProblemRow, Profile
from cf_coach.analytics.signals import BacklogItem
from cf_coach.db.models import TagSkill

NOW = datetime(2026, 10, 25, 12, 0)


def skill_row(tag, skill, attempts=5, abandon=0.0):
    return TagSkill(handle="me", tag=tag, skill=skill, attempts=attempts, abandon_rate=abandon)


def test_profile_is_weakest_first_and_rounded():
    out = profile_out(1400, [skill_row("greedy", 1510.4), skill_row("dp", 1415.6, abandon=0.1234)])
    assert out.rating == 1400
    assert [t.tag for t in out.tags] == ["dp", "greedy"]
    assert (out.tags[0].skill, out.tags[0].abandon_rate) == (1416, 0.12)


def test_profile_hides_special_marker():
    out = profile_out(1400, [skill_row("*special", 1300), skill_row("dp", 1400)])
    assert [t.tag for t in out.tags] == ["dp"]


def test_backlog_total_counts_everything_but_items_are_cut_to_limit():
    items = [BacklogItem(2000, i, 1500, ("dp", "*special"), NOW) for i in "ABC"]
    out = backlog_out(items, limit=2)
    assert out.total == 3
    assert [b.idx for b in out.items] == ["A", "B"]
    assert out.items[0].tags == ["dp"]
    assert out.items[0].last_tried.isoformat() == "2026-10-25"   # a date, not a datetime


def test_problem_out_rounds_and_keeps_matched_tag():
    p = ProblemRow(2000, "B", "Name", 1600, ("dp", "greedy"))
    out = problem_out(Candidate(p, "dp", 1415.6, 1.2345))
    assert (out.contest_id, out.idx, out.matched_tag) == (2000, "B", "dp")
    assert (out.tag_skill, out.score) == (1416, 1.23)


def test_unknown_tag_is_an_error_that_lists_valid_tags():
    known = all_tags([ProblemRow(1, "A", "x", 800, ("dp", "*special")),
                      ProblemRow(1, "B", "y", 900, ("greedy",))])
    assert known == {"dp", "greedy"}
    check_tags(["dp"], known)                        # no error
    with pytest.raises(ToolError, match=r"unknown tag\(s\) \['DP'\]. Valid tags: dp, greedy"):
        check_tags(["DP"], known)


def test_strategy_names_match_the_schema():
    assert set(STRATEGIES) == set(get_args(FindProblemsInput.model_fields["strategy"].annotation))


def test_every_tool_function_has_a_schema():
    assert set(TOOL_FUNCTIONS) <= set(TOOLS)


def test_running_contests_are_fetched_once_per_run(monkeypatch):
    fetches = []
    monkeypatch.setattr(tools, "running_contest_ids", lambda: fetches.append(1) or {2100})
    ctx = ToolContext("me", session=None, now=NOW)
    assert ctx.get_running_contests() == {2100}
    assert ctx.get_running_contests() == {2100}
    assert len(fetches) == 1


# ---------- get_progress_since ----------

TODAY = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
SINCE = date(2026, 10, 1)


def at(day, contest, idx, verdict="OK", rating=1600, tags=("dp",)):
    """A submission on 2026-10-<day>, noon UTC."""
    return SubmissionRow(contest, idx, verdict, datetime(2026, 10, day, 12, tzinfo=timezone.utc),
                         rating, tags)


def test_only_first_solves_after_since_count():
    rows = [
        at(1, 2000, "A"),                     # first AC on the `since` day itself: counts
        at(2, 2000, "A"),                     # re-submit of A: not a second solve
        at(3, 2000, "B"),
        at(5, 2000, "C", verdict="WRONG_ANSWER"),   # never solved
    ]
    # 1900A: solved in September, re-solved in October -> old progress, not new
    old = SubmissionRow(1900, "A", "OK", datetime(2026, 9, 1, tzinfo=timezone.utc), 1500, ("dp",))
    again = SubmissionRow(1900, "A", "OK", datetime(2026, 10, 9, tzinfo=timezone.utc), 1500, ("dp",))
    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    solves = first_solves_since(rows + [old, again], since)
    assert [(r.contest_id, r.idx) for r in solves] == [(2000, "B"), (2000, "A")]   # newest first


def test_progress_counts_all_but_lists_at_most_the_cap(monkeypatch):
    monkeypatch.setattr(tools, "MAX_SOLVED_LISTED", 2)
    rows = [at(d, 2000 + d, "A") for d in (2, 3, 4)]
    out = progress_out(rows, SINCE, TODAY, start=1400)
    assert out.solved_total == 3
    assert [s.contest_id for s in out.solved] == [2004, 2003]
    assert out.solved[0].solved_on == date(2026, 10, 4)


def test_skill_changes_only_list_tags_that_moved():
    rows = [
        # greedy: solved in September only, so its skill is the same at `since` and now
        SubmissionRow(1, "A", "OK", datetime(2026, 9, 1, tzinfo=timezone.utc), 1400, ("greedy",)),
        at(10, 2000, "A", rating=1800, tags=("dp", "*special")),  # hard dp solve after since
    ]
    out = progress_out(rows, SINCE, TODAY, start=1400)
    assert [c.tag for c in out.skill_changes] == ["dp"]           # greedy didn't move, no *special
    dp = out.skill_changes[0]
    assert dp.before == 1400 and dp.after > 1400                  # first tried after since


def test_future_since_is_a_tool_error():
    ctx = ToolContext("me", session=None, now=TODAY)
    with pytest.raises(ToolError, match="in the future"):
        get_progress_since(ctx, GetProgressSinceInput(since=date(2026, 10, 26)))


# ---------- validate_plan / save_plan ----------

def plan_setup():
    """Skill 1400 on dp: the dp band is 1500-1700."""
    problems = {
        (2000, "A"): ProblemRow(2000, "A", "ok", 1600, ("dp",)),
        (2000, "B"): ProblemRow(2000, "B", "trees", 1600, ("dp", "trees")),
        (2000, "C"): ProblemRow(2000, "C", "easy", 1200, ("dp",)),
        (2000, "D"): ProblemRow(2000, "D", "unrated", None, ("dp",)),
        (2100, "A"): ProblemRow(2100, "A", "live", 1600, ("dp",)),
        (2000, "E"): ProblemRow(2000, "E", "solved", 1600, ("dp",)),
    }
    return problems, {(2000, "E")}, Profile(1400, {"dp": 1400.0}), {2100}


def item(contest, idx, tag="dp"):
    return PlanItem(contest_id=contest, idx=idx, tag=tag, reason="r")


def errors_for(items, constraints=None):
    problems, solved, profile, running = plan_setup()
    return plan_errors(items, constraints or PlanConstraints(), problems, solved, profile, running)


def test_valid_plan_has_no_errors():
    assert errors_for([item(2000, "A")]) == []


def test_band_edges_are_inclusive():
    problems, solved, _, running = plan_setup()
    profile = Profile(1400, {"dp": 1300.0})           # band 1400-1600: 1600 is on the edge
    assert plan_errors([item(2000, "A")], PlanConstraints(), problems, solved, profile, running) == []


def test_each_rule_reports_its_own_line():
    errs = errors_for([
        item(2000, "C"),                 # 1200: below the band
        item(2000, "E"),                 # solved
        item(2100, "A"),                 # contest still running
        item(9999, "Z"),                 # doesn't exist
        item(2000, "A", tag="greedy"),   # tag not on the problem
        item(2000, "D"),                 # unrated
    ])
    assert errs == [
        "item 1 (2000C): rating 1200 is outside the dp band 1500-1700",
        "item 2 (2000E): already solved",
        "item 3 (2100A): contest 2100 is still running",
        "item 4 (9999Z): no such problem in the problemset",
        "item 5 (2000A): 'greedy' is not one of its tags ['dp']",
        "item 6 (2000D): unrated, so it can't be matched to a skill band",
    ]


def test_avoided_tags_duplicates_and_hours():
    errs = errors_for([item(2000, "A"), item(2000, "B"), item(2000, "A")],
                      PlanConstraints(hours=2, avoid_tags=["trees"]))
    assert errs == [
        "3 problems is too many for 2 hours: at most 2 (about 60 min each)",
        "item 2 (2000B): has avoided tag(s) ['trees']",
        "item 3 (2000A): listed twice",
    ]


def test_untried_tag_uses_the_cf_rating():
    problems, solved, _, running = plan_setup()
    profile = Profile(1400, {})                       # never tried dp -> skill = rating 1400
    assert plan_errors([item(2000, "A")], PlanConstraints(), problems, solved, profile, running) == []


def test_special_marker_is_not_a_practice_tag():
    problems = {(1952, "F"): ProblemRow(1952, "F", "fools", 1600, ("*special", "dp"))}
    errs = plan_errors([item(1952, "F", tag="*special")], PlanConstraints(), problems,
                       set(), Profile(1400, {}), set())
    assert errs == ["item 1 (1952F): '*special' is not one of its tags ['dp']"]


def test_save_refuses_an_invalid_plan_without_touching_the_db(monkeypatch):
    monkeypatch.setattr(tools, "check_plan", lambda ctx, items, c: ["item 1 (2000C): too easy"])
    ctx = ToolContext("me", session=None, now=TODAY)          # session=None: any DB use crashes
    with pytest.raises(ToolError, match="plan not saved.*too easy"):
        save_plan(ctx, SavePlanInput(items=[item(2000, "C")]))
    assert ctx.saved_plan_id is None


def test_save_only_once_per_run():
    ctx = ToolContext("me", session=None, now=TODAY, saved_plan_id=7)
    with pytest.raises(ToolError, match="already saved in this run"):
        save_plan(ctx, SavePlanInput(items=[item(2000, "A")]))


# ---------- the user's fixed constraints (decision #32) ----------

def fake_plan_data(monkeypatch):
    """tools' DB loaders return plan_setup() data, so no database is needed."""
    problems, solved, profile, running = plan_setup()
    monkeypatch.setattr(tools, "load_problems", lambda session: list(problems.values()))
    monkeypatch.setattr(tools, "load_solved", lambda session, handle: solved)
    monkeypatch.setattr(tools, "build_profile", lambda session, handle, now: profile)
    return running


def test_user_hours_win_and_avoided_tags_from_both_count():
    user = PlanConstraints(hours=4, avoid_tags=["geometry"])
    model = PlanConstraints(hours=10, avoid_tags=["trees", "geometry"])
    assert effective_constraints(user, model) == PlanConstraints(
        hours=4, avoid_tags=["geometry", "trees"])


def test_model_constraints_still_apply_when_the_user_set_none():
    model = PlanConstraints(hours=3, avoid_tags=["trees"])
    assert effective_constraints(PlanConstraints(), model) == model


def test_validate_enforces_user_constraints_the_model_dropped(monkeypatch):
    running = fake_plan_data(monkeypatch)
    ctx = ToolContext("me", session=None, now=TODAY, running_contests=running,
                      user_constraints=PlanConstraints(hours=1, avoid_tags=["trees"]))
    plan = ValidatePlanInput(items=[item(2000, "A"), item(2000, "B")])   # model sent no constraints
    assert validate_plan(ctx, plan).errors == [
        "2 problems is too many for 1 hours: at most 1 (about 60 min each)",
        "item 2 (2000B): has avoided tag(s) ['trees']",
    ]


class FakeSession:
    """Just enough of a Session for save_plan: records what was added."""
    def __init__(self):
        self.added = []

    def execute(self, statement):
        pass

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        self.added[-1].id = 42          # Postgres would assign the id here

    def commit(self):
        pass

    def rollback(self):
        pass


def test_save_stores_the_enforced_constraints_not_the_models(monkeypatch):
    monkeypatch.setattr(tools, "check_plan", lambda ctx, items, c: [])
    session = FakeSession()
    ctx = ToolContext("me", session=session, now=TODAY,
                      user_constraints=PlanConstraints(hours=4, avoid_tags=["geometry"]))
    out = save_plan(ctx, SavePlanInput(items=[item(2000, "A")]))   # model: hours=None
    assert out.plan_id == 42 and ctx.saved_plan_id == 42
    assert session.added[0].constraints == {"hours": 4, "avoid_tags": ["geometry"]}


def test_find_problems_always_skips_the_users_avoided_tags(monkeypatch):
    running = fake_plan_data(monkeypatch)
    seen = {}

    def fake_select(*args, excluded_tags, **kwargs):
        seen["excluded"] = excluded_tags
        return []

    monkeypatch.setattr(tools, "select_candidates", fake_select)
    ctx = ToolContext("me", session=None, now=TODAY, running_contests=running,
                      user_constraints=PlanConstraints(avoid_tags=["trees"]))
    find_problems(ctx, FindProblemsInput(tag="dp"))          # the model avoided nothing
    assert seen["excluded"] == frozenset({"trees"})


def test_unknown_user_tag_is_rejected_up_front(monkeypatch):
    fake_plan_data(monkeypatch)
    with pytest.raises(ToolError, match=r"unknown tag\(s\) \['geomtry'\]"):
        user_constraints(None, 4, ["geomtry"])
    assert user_constraints(None, 4, ["trees"]) == PlanConstraints(hours=4, avoid_tags=["trees"])


def test_memory_store_keeps_the_plan_out_of_the_database(monkeypatch):
    monkeypatch.setattr(tools, "check_plan", lambda ctx, items, c: [])
    store = MemoryPlanStore()
    ctx = ToolContext("me", session=None, now=TODAY, plan_store=store)   # session=None: no DB
    out = save_plan(ctx, SavePlanInput(items=[item(2000, "A")],
                                       constraints=PlanConstraints(hours=2)))
    assert out.plan_id == 1 and ctx.saved_plan_id == 1
    constraints, items = store.plans[1]
    assert constraints.hours == 2 and [(i.contest_id, i.idx) for i in items] == [(2000, "A")]
