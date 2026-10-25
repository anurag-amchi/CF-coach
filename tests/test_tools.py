from datetime import datetime
from typing import get_args

import pytest

import cf_coach.agent.tools as tools
from cf_coach.agent.schemas import TOOLS, FindProblemsInput
from cf_coach.agent.tools import (
    STRATEGIES, TOOL_FUNCTIONS, ToolContext, ToolError, all_tags, backlog_out, check_tags,
    problem_out, profile_out,
)
from cf_coach.analytics.selection import Candidate, ProblemRow
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
