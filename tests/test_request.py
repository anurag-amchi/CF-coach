import json

import pytest

from cf_coach.agent.llm import ScriptedLLM, scripted_reply
from cf_coach.agent.request import (
    DEFAULT_COUNT, EMPTY, PlanRequest, check_reply, merge_flags, parse_request, plan_limits,
    read_request, request_schema,
)
from cf_coach.agent.tools import ToolError

TAGS = {"dp", "greedy", "geometry", "graphs"}


def reply_json(**fields):
    """A model reply: EMPTY's fields, with some changed."""
    return scripted_reply(text=json.dumps(EMPTY.model_dump() | fields))


def test_a_good_reply_is_used_as_it_is():
    llm = ScriptedLLM([reply_json(count=3, focus_tags=["dp"], difficulty="harder")])
    r = parse_request(llm, "Challenge me: 3 hard dp problems.", TAGS)
    assert r.request == PlanRequest(intent="plan", count=3, hours=None, focus_tags=["dp"],
                                    avoid_tags=[], difficulty="harder")
    assert r.warnings == []
    assert len(llm.seen) == 1


def test_the_schema_is_sent_and_lists_only_real_tags():
    llm = ScriptedLLM([reply_json()])
    parse_request(llm, "plan my week", TAGS)
    schema = llm.schemas[0]
    assert schema["properties"]["focus_tags"]["items"]["enum"] == sorted(TAGS)
    assert schema["properties"]["avoid_tags"]["items"]["enum"] == sorted(TAGS)


def test_every_field_is_required_and_nothing_extra():
    """Strict structured output needs this: null / [] instead of a missing field."""
    schema = request_schema(TAGS)
    assert sorted(schema["required"]) == sorted(PlanRequest.model_fields)
    assert schema["additionalProperties"] is False


def test_an_unknown_tag_gets_one_retry_with_the_error():
    llm = ScriptedLLM([reply_json(focus_tags=["dynamic programming"]),
                       reply_json(focus_tags=["dp"])])
    r = parse_request(llm, "3 dp problems", TAGS)
    assert r.request.focus_tags == ["dp"] and r.warnings == []
    retry = llm.seen[1][-1]["content"]
    assert retry.startswith("That JSON is not valid: unknown tag(s) ['dynamic programming']")


def test_out_of_range_numbers_are_rejected():
    llm = ScriptedLLM([reply_json(count=50), reply_json(count=20)])
    assert parse_request(llm, "50 problems", TAGS).request.count == 20
    assert "count" in llm.seen[1][-1]["content"]


def test_two_bad_replies_give_the_empty_request_and_a_warning():
    llm = ScriptedLLM([scripted_reply(text="not json"), scripted_reply(text="{}")])
    r = parse_request(llm, "plan", TAGS)
    assert r.request == EMPTY
    assert r.warnings[0].startswith("could not read the request (")
    assert r.warnings[0].endswith("planning without it")


def test_a_failed_call_never_blocks_the_run():
    llm = ScriptedLLM([ConnectionError("Groq is down")])
    r = parse_request(llm, "plan", TAGS)
    assert r.request == EMPTY
    assert "ConnectionError: Groq is down" in r.warnings[0]


def test_check_reply_needs_every_field():
    try:
        check_reply('{"intent": "plan"}', TAGS)
    except ValueError as e:
        assert "count" in str(e) and "Field required" in str(e)
    else:
        raise AssertionError("a reply with missing fields was accepted")


def test_flags_win_and_avoided_tags_from_both_count():
    text = PlanRequest(intent="plan", count=3, hours=10, focus_tags=["dp"],
                       avoid_tags=["graphs", "geometry"], difficulty="easier")
    merged = merge_flags(text, hours=4, avoid_tags=["geometry"], difficulty="harder")
    assert (merged.hours, merged.difficulty) == (4, "harder")
    assert merged.avoid_tags == ["geometry", "graphs"]
    assert (merged.count, merged.focus_tags) == (3, ["dp"])        # no flag: text kept


def test_no_flags_keep_the_text():
    text = PlanRequest(intent="plan", count=None, hours=6, focus_tags=[], avoid_tags=["dp"],
                       difficulty="harder")
    assert merge_flags(text, hours=None, avoid_tags=[], difficulty=None) == text


# ---------- read_request / plan_limits (decision #38) ----------

def plan(**fields):
    return EMPTY.model_copy(update=fields)


def test_no_count_and_no_hours_means_the_default_week():
    assert plan_limits(plan())["count"] == DEFAULT_COUNT
    assert plan_limits(plan(hours=4))["count"] is None          # the hours cap still applies
    assert plan_limits(plan(intent="question"))["count"] is None
    assert plan_limits(plan(count=3, hours=2))["count"] == 3    # capped later by the hours


def test_limits_become_context_fields():
    limits = plan_limits(plan(count=3, focus_tags=["dp"], avoid_tags=["geometry"], hours=6,
                              difficulty="harder"))
    assert limits["user_constraints"].model_dump() == {"hours": 6, "avoid_tags": ["geometry"]}
    assert (limits["difficulty"], limits["focus_tags"]) == ("harder", ["dp"])
    assert plan_limits(plan())["difficulty"] == "normal"


def fake_problems(monkeypatch):
    import cf_coach.analytics.selection as selection
    from cf_coach.analytics.selection import ProblemRow
    monkeypatch.setattr(selection, "load_problems", lambda session: [
        ProblemRow(1, "A", "x", 1500, tuple(TAGS))])


def test_read_request_merges_the_flags(monkeypatch):
    fake_problems(monkeypatch)
    llm = ScriptedLLM([reply_json(count=3, hours=10, difficulty="easier")])
    request, parsed = read_request(llm, None, "3 easy ones, 10 hours", hours=4,
                                   avoid_tags=["geometry"], difficulty=None)
    assert (request.count, request.hours, request.difficulty) == (3, 4, "easier")
    assert request.avoid_tags == ["geometry"] and parsed.warnings == []


def test_a_typo_in_a_flag_fails_before_any_llm_call(monkeypatch):
    fake_problems(monkeypatch)
    llm = ScriptedLLM([])                          # any call would raise "ran out of replies"
    with pytest.raises(ToolError, match=r"unknown tag\(s\) \['geomtry'\]"):
        read_request(llm, None, "plan", None, ["geomtry"], None)
    assert llm.seen == []
