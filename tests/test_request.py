import json

from cf_coach.agent.llm import ScriptedLLM, scripted_reply
from cf_coach.agent.request import (
    EMPTY, PlanRequest, check_reply, merge_flags, parse_request, request_schema,
)

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
