from datetime import date, datetime, timezone
from typing import get_args

import pytest

import cf_coach.agent.tools as tools
from cf_coach.agent.schemas import (
    TOOLS, FindProblemsInput, GetProgressSinceInput, GetUpsolveBacklogInput, PlanConstraints,
    PlanItem, SavePlanInput, TagSkillOut, ValidatePlanInput,
)
from cf_coach.agent.tools import (
    STRATEGIES, TOOL_FUNCTIONS, MemoryPlanStore, Offer, SavedConstraints, SavedItem, ToolContext,
    ToolError, all_tags, backlog_out, check_tags, effective_constraints, find_problems,
    first_solves_since, get_progress_since, get_upsolve_backlog, item_facts, plan_errors,
    plan_size, plan_summary, plan_tag, problem_id, problem_out, profile_out, progress_out,
    save_plan, saved_items, user_constraints, validate_plan, wanted_count, week_ladder, week_quota, levels_text,
)
from cf_coach.analytics.attempts import SubmissionRow
from cf_coach.analytics.selection import Candidate, ProblemRow, Profile
from cf_coach.analytics.signals import BacklogItem, TagSignals
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
    assert [b.id for b in out.items] == ["2000A", "2000B"]
    assert out.items[0].tags == ["dp"]
    assert out.items[0].last_tried.isoformat() == "2026-10-25"   # a date, not a datetime


def test_problem_out_rounds_and_keeps_matched_tag():
    p = ProblemRow(2000, "B", "Name", 1600, ("dp", "greedy"))
    out = problem_out(Candidate(p, "dp", 1415.6, 1.2345))
    assert (out.id, out.matched_tag) == ("2000B", "dp")
    assert (out.tag_skill, out.score) == (1416, 1.23)
    assert out.gap == 200                 # 1600 - 1415 rounded to 1400


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
    assert [s.id for s in out.solved] == ["2004A", "2003A"]
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
    """Skill 1400 on dp: the normal dp band is 1400-1600 (harder 1500-1700)."""
    problems = {
        (2000, "A"): ProblemRow(2000, "A", "ok", 1600, ("dp",)),
        (2000, "F"): ProblemRow(2000, "F", "level", 1400, ("dp",)),
        (2000, "G"): ProblemRow(2000, "G", "level+1", 1500, ("dp",)),
        (2000, "B"): ProblemRow(2000, "B", "trees", 1600, ("dp", "trees")),
        (2000, "C"): ProblemRow(2000, "C", "easy", 1200, ("dp",)),
        (2000, "D"): ProblemRow(2000, "D", "unrated", None, ("dp",)),
        (2100, "A"): ProblemRow(2100, "A", "live", 1600, ("dp",)),
        (2000, "E"): ProblemRow(2000, "E", "solved", 1600, ("dp",)),
    }
    return problems, {(2000, "E")}, Profile(1400, {"dp": 1400.0}), {2100}


def offered_as_dp(problems):
    """As if find_problems had returned every one of them, matched to dp."""
    return {problem_id(c, i): Offer(c, i, "dp") for (c, i) in problems}


def item(pid, reason=None):
    """Each id gets its own reason by default: equal reasons are a plan error (#40)."""
    return PlanItem(id=pid, reason=reason or f"why {pid}")


def errors_for(items, constraints=None, offered=None):
    problems, solved, profile, running = plan_setup()
    offered = offered_as_dp(problems) if offered is None else offered
    return plan_errors(items, constraints or PlanConstraints(), offered, problems, solved,
                       profile, running)


def test_valid_plan_has_no_errors():
    assert errors_for([item("2000A")]) == []


def test_ids_forgive_case_and_spaces():
    assert errors_for([item(" 2000a ")]) == []


def test_band_edges_are_inclusive():
    problems, solved, _, running = plan_setup()
    for skill in (1350.0, 1550.0):     # rounds to 1400 (1600 = top edge), 1600 (bottom edge)
        profile = Profile(1400, {"dp": skill})
        assert plan_errors([item("2000A")], PlanConstraints(), offered_as_dp(problems),
                           problems, solved, profile, running) == []


def test_difficulty_moves_the_band():
    errs = errors_for([item("2000F"), item("2000A")])    # normal: 1400 and 1600 both fit
    assert errs == []
    problems, solved, profile, running = plan_setup()
    errs = plan_errors([item("2000F"), item("2000A")], PlanConstraints(),
                       offered_as_dp(problems), problems, solved, profile, running, "harder")
    assert errs == ["item 1 (2000F): rating 1400 is outside the harder dp band 1500-1700"]


def test_saved_plan_is_a_ladder_whatever_the_models_order():
    problems, _, profile, _ = plan_setup()
    out = saved_items([item("2000A"), item("2000G"), item("2000F")],   # hardest first
                      offered_as_dp(problems), problems, profile)
    assert [(s.id, s.gap) for s in out] == [("2000F", 0), ("2000G", 100), ("2000A", 200)]


def test_each_rule_reports_its_own_line():
    offered = offered_as_dp(plan_setup()[0]) | {"9999Z": Offer(9999, "Z", None)}
    errs = errors_for([
        item("2000C"),       # 1200: below the band
        item("2000E"),       # solved
        item("2100A"),       # contest still running
        item("9999Z"),       # the backlog returned it, but CF's problemset doesn't have it
        item("1234A"),       # no tool returned it
        item("2000D"),       # unrated
    ], offered=offered)
    assert errs == [
        "item 1 (2000C): rating 1200 is outside the normal dp band 1400-1600",
        "item 2 (2000E): already solved",
        "item 3 (2100A): contest 2100 is still running",
        "item 4 (9999Z): no such problem in the problemset",
        "item 5 (1234A): no tool returned this id in this conversation; "
        "plan only ids from find_problems or get_upsolve_backlog",
        "item 6 (2000D): unrated, so it can't be matched to a skill band",
    ]


def test_avoided_tags_duplicates_and_hours():
    errs = errors_for([item("2000A"), item("2000B"), item("2000a")],   # same id, other case
                      PlanConstraints(hours=2, avoid_tags=["trees"]))
    assert errs == [
        "3 problems is too many for 2 hours: at most 2 (about 60 min each)",
        "item 2 (2000B): has avoided tag(s) ['trees']",
        "item 3 (2000A): listed twice",
    ]


def test_untried_tag_uses_the_cf_rating():
    problems, solved, _, running = plan_setup()
    profile = Profile(1400, {})                       # never tried dp -> skill = rating 1400
    assert plan_errors([item("2000A")], PlanConstraints(), offered_as_dp(problems), problems,
                       solved, profile, running) == []


# ---------- backlog problems: no matched tag, so the code picks one ----------

BACKLOG = {
    (2200, "A"): ProblemRow(2200, "A", "two tags", 1600, ("greedy", "dp", "*special")),
    (2200, "B"): ProblemRow(2200, "B", "too easy", 1200, ("greedy", "dp")),
}
# greedy 1450 rounds to 1500: normal bands dp 1400-1600, greedy 1500-1700
BACKLOG_PROFILE = Profile(1400, {"dp": 1400.0, "greedy": 1450.0})
FROM_BACKLOG = {problem_id(c, i): Offer(c, i, None) for (c, i) in BACKLOG}


def test_backlog_problem_practises_its_weakest_tag_that_fits():
    p = BACKLOG[(2200, "A")]
    assert plan_tag(FROM_BACKLOG["2200A"], p, BACKLOG_PROFILE, "normal") == "dp"   # not *special
    assert saved_items([item("2200A")], FROM_BACKLOG, BACKLOG, BACKLOG_PROFILE) == [
        SavedItem(id="2200A", contest_id=2200, idx="A", tag="dp", rating=1600, gap=200,
                  facts="dp skill 1400 (level 1400) · 1600 = level +200 · in your upsolve backlog",
                  reason="why 2200A")]


def test_weakest_tag_must_still_fit_the_band():
    p = ProblemRow(2200, "C", "x", 1700, ("dp", "greedy"))    # above dp's 1600, inside greedy's
    assert plan_tag(Offer(2200, "C", None), p, BACKLOG_PROFILE, "normal") == "greedy"


def test_backlog_problem_outside_every_band_is_an_error():
    errs = plan_errors([item("2200B")], PlanConstraints(), FROM_BACKLOG, BACKLOG, set(),
                       BACKLOG_PROFILE, set())
    assert errs == ["item 1 (2200B): rating 1200 is outside the normal band of every one "
                    "of its tags ['greedy', 'dp']"]


def test_find_problems_tag_is_never_replaced_by_the_backlog():
    ctx = ToolContext("me", session=None, now=TODAY)
    ctx.offer(2000, "A", "dp")
    ctx.offer(2000, "A", None)            # the backlog returns it too, later
    ctx.offer(2000, "B", None)
    ctx.offer(2000, "B", "greedy")        # find_problems returns it later: its tag is used
    assert ctx.offered == {"2000A": Offer(2000, "A", "dp"), "2000B": Offer(2000, "B", "greedy")}


def test_ids_are_never_parsed_back():
    """Contest 921 has index '01': '92101' only works because the tool kept the split."""
    problems = {(921, "01"): ProblemRow(921, "01", "Labyrinth-1", 3200, ("dp",))}
    offered = {problem_id(921, "01"): Offer(921, "01", "dp")}
    profile = Profile(1400, {"dp": 3000.0})
    assert plan_errors([item("92101")], PlanConstraints(), offered, problems, set(),
                       profile, set()) == []
    [saved] = saved_items([item("92101")], offered, problems, profile)
    assert (saved.contest_id, saved.idx) == (921, "01")


def test_tools_remember_only_what_they_returned(monkeypatch):
    running = fake_plan_data(monkeypatch)
    problems, *_ = plan_setup()
    monkeypatch.setattr(tools, "select_candidates", lambda *a, **k: [
        Candidate(problems[(2000, "A")], "dp", 1400.0, 1.0)])
    monkeypatch.setattr(tools, "load_rows", lambda session, handle: [])
    monkeypatch.setattr(tools, "upsolve_backlog", lambda rows: [
        BacklogItem(2000, i, 1600, ("dp",), NOW) for i in "BCD"])
    ctx = ToolContext("me", session=None, now=TODAY, running_contests=running)
    find_problems(ctx, FindProblemsInput(tag="dp"))
    get_upsolve_backlog(ctx, GetUpsolveBacklogInput(limit=2))       # 2000D is not shown
    assert ctx.offered == {"2000A": Offer(2000, "A", "dp"),
                           "2000B": Offer(2000, "B", None), "2000C": Offer(2000, "C", None)}


def test_save_refuses_an_invalid_plan_without_touching_the_db(monkeypatch):
    monkeypatch.setattr(tools, "check_plan",
                        lambda ctx, items, c: (["item 1 (2000C): too easy"], []))
    ctx = ToolContext("me", session=None, now=TODAY)          # session=None: any DB use crashes
    with pytest.raises(ToolError, match="plan not saved.*too easy"):
        save_plan(ctx, SavePlanInput(items=[item("2000C")]))
    assert ctx.saved_plan_id is None


def test_save_only_once_per_run():
    ctx = ToolContext("me", session=None, now=TODAY, saved_plan_id=7)
    with pytest.raises(ToolError, match="already saved in this run"):
        save_plan(ctx, SavePlanInput(items=[item("2000A")]))


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
                      user_constraints=PlanConstraints(hours=1, avoid_tags=["trees"]),
                      offered=offered_as_dp(plan_setup()[0]))
    plan = ValidatePlanInput(items=[item("2000A"), item("2000B")])   # model sent no constraints
    assert validate_plan(ctx, plan).errors == [
        "2 problems is too many for 1 hours: at most 1 (about 60 min each)",
        "item 2 (2000B): has avoided tag(s) ['trees']",
    ]
    assert ctx.validated_ok is False


def test_validated_ok_follows_the_latest_validation(monkeypatch):
    running = fake_plan_data(monkeypatch)
    ctx = ToolContext("me", session=None, now=TODAY, running_contests=running,
                      offered=offered_as_dp(plan_setup()[0]))
    assert validate_plan(ctx, ValidatePlanInput(items=[item("2000A")])).ok
    assert ctx.validated_ok is True
    assert not validate_plan(ctx, ValidatePlanInput(items=[item("2000C")])).ok   # 1200: too easy
    assert ctx.validated_ok is False


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


def test_save_stores_the_enforced_constraints_and_the_worked_out_items(monkeypatch):
    running = fake_plan_data(monkeypatch)
    session = FakeSession()
    ctx = ToolContext("me", session=session, now=TODAY, running_contests=running,
                      user_constraints=PlanConstraints(hours=4, avoid_tags=["geometry"]),
                      offered=offered_as_dp(plan_setup()[0]))
    out = save_plan(ctx, SavePlanInput(items=[item("2000a")]))   # model: hours=None
    assert out.plan_id == 42 and ctx.saved_plan_id == 42
    assert session.added[0].constraints == {"hours": 4, "avoid_tags": ["geometry"],
                                            "difficulty": "normal"}
    # the JSON the plans row gets: the code's id spelling, tag, rating, gap and facts;
    # the model's reason
    assert session.added[0].items == [
        {"id": "2000A", "contest_id": 2000, "idx": "A", "tag": "dp", "rating": 1600, "gap": 200,
         "facts": "dp skill 1400 (level 1400) · 1600 = level +200", "reason": "why 2000a"}]
    constraints, items = ctx.saved_plan               # kept for the loop's summary
    assert constraints.hours == 4 and [i.id for i in items] == ["2000A"]


def test_find_problems_always_skips_the_users_avoided_tags(monkeypatch):
    running = fake_plan_data(monkeypatch)
    seen = {}

    def fake_select(*args, excluded_tags, difficulty, **kwargs):
        seen["excluded"], seen["difficulty"] = excluded_tags, difficulty
        return []

    monkeypatch.setattr(tools, "select_candidates", fake_select)
    ctx = ToolContext("me", session=None, now=TODAY, running_contests=running,
                      user_constraints=PlanConstraints(avoid_tags=["trees"]), difficulty="harder")
    out = find_problems(ctx, FindProblemsInput(tag="dp"))    # the model avoided nothing
    assert seen["excluded"] == frozenset({"trees"})
    assert seen["difficulty"] == "harder" == out.difficulty  # from ctx, not from the model


def test_unknown_user_tag_is_rejected_up_front(monkeypatch):
    fake_plan_data(monkeypatch)
    with pytest.raises(ToolError, match=r"unknown tag\(s\) \['geomtry'\]"):
        user_constraints(None, 4, ["geomtry"])
    assert user_constraints(None, 4, ["trees"]) == PlanConstraints(hours=4, avoid_tags=["trees"])


def test_memory_store_keeps_the_plan_out_of_the_database(monkeypatch):
    running = fake_plan_data(monkeypatch)
    store = MemoryPlanStore()
    ctx = ToolContext("me", session=None, now=TODAY, running_contests=running,   # no DB
                      plan_store=store, offered=offered_as_dp(plan_setup()[0]))
    out = save_plan(ctx, SavePlanInput(items=[item("2000A")],
                                       constraints=PlanConstraints(hours=2)))
    assert out.plan_id == 1 and ctx.saved_plan_id == 1
    constraints, items = store.plans[1]
    assert constraints.hours == 2 and [(i.contest_id, i.idx) for i in items] == [(2000, "A")]


# ---------- count and focus tags from the request (decision #38) ----------

def test_count_is_exact():
    problems, solved, profile, running = plan_setup()
    errs = plan_errors([item("2000A"), item("2000F")], PlanConstraints(), offered_as_dp(problems),
                       problems, solved, profile, running, count=3)
    assert errs == ["the plan has 2 problems; it must have exactly 3 (the user asked for 3)"]


def test_hours_cap_the_count_with_one_error_not_two():
    problems, solved, profile, running = plan_setup()
    five = [item(i) for i in ("2000A", "2000F", "2000G", "2000B", "2100A")]
    errs = plan_errors(five, PlanConstraints(hours=3), offered_as_dp(problems), problems,
                       solved, profile, running, count=5)
    assert errs[0] == ("the plan has 5 problems; it must have exactly 3 "
                       "(the user asked for 5, and 3 hours allow at most 3)")
    assert not any("too many" in e for e in errs)


def test_wanted_count():
    assert wanted_count(None, 4) is None
    assert wanted_count(3, None) == 3
    assert wanted_count(5, 3) == 3 and wanted_count(2, 3) == 2


def test_items_must_practise_a_focus_tag():
    problems, solved, profile, running = plan_setup()
    errs = plan_errors([item("2000A")], PlanConstraints(), offered_as_dp(problems), problems,
                       solved, profile, running, focus=["greedy"])
    assert errs == ["item 1 (2000A): has none of the tags the user asked for ['greedy']"]


def test_equal_reasons_are_an_error():
    errs = errors_for([item("2000A", "Good DP practice."), item("2000F", "fine"),
                       item("2000G", "  good dp   PRACTICE. ")])     # same, apart from case/spaces
    assert errs == ["item 3 (2000G): same reason as item 1; give each problem its own reason"]


# ---------- facts written by code, summary written by code (decision #40) ----------

def test_facts_for_a_tried_tag():
    profile = Profile(1400, {"dp": 1415.4}, {"dp": TagSignals(attempts=26, given_up=3)})
    assert item_facts("dp", 1500, profile, from_backlog=False) == (
        "dp skill 1415 (level 1400) · 26 tries · 12% abandoned · 1500 = level +100")


def test_facts_for_an_untried_tag_and_a_backlog_problem():
    profile = Profile(1450, {})                      # rating 1450 rounds to level 1500
    assert item_facts("geometry", 1500, profile, from_backlog=True) == (
        "geometry untried (level 1500, from your rating) · 1500 = level +0 · in your upsolve backlog")


def saved(pid, tag, rating, gap):
    return SavedItem(id=pid, contest_id=1, idx="A", tag=tag, rating=rating, gap=gap,
                     facts="f", reason="r")


def test_plan_summary_is_short_plain_text():
    items = [saved("1A", "dp", 1400, 0), saved("2A", "greedy", 1600, 0), saved("3A", "dp", 1500, 100)]
    text = plan_summary(SavedConstraints(hours=4, avoid_tags=["geometry"]), items)
    assert text == ("Saved 3 problems, easiest first, rated 1400-1600. Tags: dp x2, greedy x1. "
                    "Difficulty: normal (+0 to +200 around each tag's level). Hours: 4. "
                    "Avoiding: geometry.")
    assert "|" not in text and "\n" not in text
    one = plan_summary(SavedConstraints(difficulty="harder"), [saved("1A", "dp", 1500, 100)])
    assert one == ("Saved 1 problem, easiest first, rated 1500-1500. Tags: dp x1. "
                   "Difficulty: harder (+100 to +300 around each tag's level).")


# ---------- the week's tags when the request names none (decision #43) ----------

def tags_out(*rows):
    """(tag, attempts) pairs, already weakest first like get_tag_profile."""
    return [TagSkillOut(tag=t, skill=1400 + i, attempts=a, abandon_rate=0.0)
            for i, (t, a) in enumerate(rows)]


PROFILE_TAGS = tags_out(("dfs and similar", 3), ("geometry", 7), ("flows", 1), ("strings", 32),
                        ("trees", 5), ("combinatorics", 17), ("dp", 26))


def test_week_quota_takes_two_from_the_weakest_measured_tag_and_one_from_the_next():
    assert week_quota(PROFILE_TAGS, 5) == {"geometry": 2, "strings": 1, "trees": 1,
                                           "combinatorics": 1}       # dfs (3), flows (1) skipped


def test_week_quota_small_sizes_avoided_tags_and_too_few_tags():
    assert week_quota(PROFILE_TAGS, 1) == {"geometry": 1}
    assert week_quota(PROFILE_TAGS, 2) == {"geometry": 2}
    assert week_quota(PROFILE_TAGS, 3, avoid=["geometry"]) == {"strings": 2, "trees": 1}
    assert week_quota(tags_out(("dp", 9), ("greedy", 6)), 4) == {}   # needs 3 measured tags
    assert week_quota(PROFILE_TAGS, 0) == {}


def test_plan_size():
    assert plan_size(5, None) == 5 and plan_size(5, 3) == 3
    assert plan_size(None, 4) == 4 and plan_size(None, None) is None


QUOTA_PROBLEMS = {
    (2400, "A"): ProblemRow(2400, "A", "g1", 1400, ("geometry",)),
    (2400, "B"): ProblemRow(2400, "B", "g2", 1500, ("geometry",)),
    (2400, "C"): ProblemRow(2400, "C", "s1", 1500, ("strings",)),
    (2400, "D"): ProblemRow(2400, "D", "d1", 1500, ("dp",)),
}
QUOTA_OFFERS = {"2400A": Offer(2400, "A", "geometry"), "2400B": Offer(2400, "B", "geometry"),
                "2400C": Offer(2400, "C", "strings"), "2400D": Offer(2400, "D", "dp")}
QUOTA_PROFILE = Profile(1400, {"geometry": 1400.0, "strings": 1400.0, "dp": 1400.0})


def quota_errors(ids):
    return plan_errors([item(i) for i in ids], PlanConstraints(), QUOTA_OFFERS, QUOTA_PROBLEMS,
                       set(), QUOTA_PROFILE, set(), quota={"geometry": 2, "strings": 1})


def test_a_plan_that_meets_the_quota_is_valid():
    assert quota_errors(["2400A", "2400B", "2400C"]) == []


def test_quota_errors_name_the_tag_and_the_counts():
    assert quota_errors(["2400A", "2400C", "2400D"]) == [
        "item 3 (2400D): has none of this week's tags ['geometry', 'strings']",
        "this week needs exactly 2 geometry problem(s); the plan has 1",
    ]


# ---------- the week's levels: a ladder from easy to hard (decision #44) ----------

def test_week_ladder_spreads_the_levels_and_the_weakest_tag_climbs():
    quota = {"geometry": 2, "strings": 1, "trees": 1, "combinatorics": 1}
    assert week_ladder(quota) == {"geometry": [0, 200], "strings": [0], "trees": [100],
                                  "combinatorics": [100]}
    assert week_ladder({"geometry": 2, "strings": 1, "trees": 1}) == {
        "geometry": [0, 200], "strings": [0], "trees": [100]}
    assert week_ladder({"geometry": 2, "strings": 1}) == {"geometry": [0, 200], "strings": [100]}


def test_week_ladder_small_weeks_other_difficulties_and_empty():
    assert week_ladder({"geometry": 2}) == {"geometry": [0, 100]}
    assert week_ladder({"geometry": 1}) == {"geometry": [0]}
    assert week_ladder({"geometry": 2, "strings": 1}, "harder") == {"geometry": [100, 300],
                                                                    "strings": [200]}
    assert week_ladder({"geometry": 2, "strings": 1}, "easier") == {"geometry": [-100, 100],
                                                                    "strings": [0]}
    assert week_ladder({}) == {}


def test_levels_text():
    assert levels_text([0]) == "+0"
    assert levels_text([200, 0]) == "+0 and +200"
    assert levels_text([0, 100, 200]) == "+0, +100 and +200"


def ladder_errors(ids, gaps):
    return plan_errors([item(i) for i in ids], PlanConstraints(), QUOTA_OFFERS, QUOTA_PROBLEMS,
                       set(), QUOTA_PROFILE, set(), quota={"geometry": 2, "strings": 1},
                       gaps=gaps)


def test_a_plan_at_the_week_levels_is_valid():
    assert ladder_errors(["2400A", "2400B", "2400C"],
                         {"geometry": [0, 100], "strings": [100]}) == []


def test_level_errors_name_the_levels_and_the_ratings():
    assert ladder_errors(["2400A", "2400B", "2400C"],
                         {"geometry": [0, 200], "strings": [100]}) == [
        "this week's geometry problem(s) must be at level +0 and +200 (rated 1400, 1600); "
        "the plan has +0 and +100"]


def test_a_wrong_count_reports_only_the_count_not_the_levels():
    assert ladder_errors(["2400A", "2400C"], {"geometry": [0, 200], "strings": [100]}) == [
        "this week needs exactly 2 geometry problem(s); the plan has 1"]


def test_focus_re_matches_a_problem_to_the_asked_tag():
    """find_problems matched it to dfs; the user asked for dp, which also fits its band."""
    p = ProblemRow(2300, "A", "x", 1500, ("dfs and similar", "dp"))
    profile = Profile(1400, {"dp": 1400.0, "dfs and similar": 1400.0})
    assert plan_tag(Offer(2300, "A", "dfs and similar"), p, profile, "normal", ["dp"]) == "dp"
    assert plan_tag(Offer(2300, "A", "dfs and similar"), p, profile, "normal") == "dfs and similar"
