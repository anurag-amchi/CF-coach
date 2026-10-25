from datetime import date, datetime, timezone
from typing import get_args

import pytest

import cf_coach.agent.tools as tools
from cf_coach.agent.schemas import (
    TOOLS, FindProblemsInput, GetProgressSinceInput, PlanConstraints, PlanItem, SavePlanInput,
)
from cf_coach.agent.tools import (
    STRATEGIES, TOOL_FUNCTIONS, ToolContext, ToolError, all_tags, backlog_out, check_tags,
    first_solves_since, get_progress_since, plan_errors, problem_out, profile_out, progress_out,
    save_plan,
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
    monkeypatch.setattr(tools, "check_plan", lambda ctx, args: ["item 1 (2000C): too easy"])
    ctx = ToolContext("me", session=None, now=TODAY)          # session=None: any DB use crashes
    with pytest.raises(ToolError, match="plan not saved.*too easy"):
        save_plan(ctx, SavePlanInput(items=[item(2000, "C")]))
    assert ctx.saved_plan_id is None


def test_save_only_once_per_run():
    ctx = ToolContext("me", session=None, now=TODAY, saved_plan_id=7)
    with pytest.raises(ToolError, match="already saved in this run"):
        save_plan(ctx, SavePlanInput(items=[item(2000, "A")]))
