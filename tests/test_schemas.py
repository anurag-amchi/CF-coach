import json
from datetime import date

import pytest

from cf_coach.agent.schemas import (
    TOOL_SCHEMAS, TOOLS, FindProblemsInput, GetProgressSinceInput, GetTagProfileInput,
    SavePlanInput, ToolArgsError, ValidatePlanInput, parse_args,
)

ITEM = {"id": "2000B", "reason": "Weakest tag."}


def test_one_schema_per_tool_and_no_handle_argument():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names == list(TOOLS)
    assert len(names) == 6
    for s in TOOL_SCHEMAS:
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert "handle" not in params["properties"]


def test_defaults_are_filled_in():
    args = parse_args(FindProblemsInput, '{"tag": "dp"}')
    assert args == FindProblemsInput(tag="dp", strategy="weakness_first", avoid_tags=[], count=10)


def test_empty_arguments_are_allowed_for_a_tool_without_fields():
    assert parse_args(GetTagProfileInput, "") == GetTagProfileInput()
    assert parse_args(GetTagProfileInput, "{}") == GetTagProfileInput()


def test_every_problem_is_reported_in_one_message():
    with pytest.raises(ToolArgsError) as e:
        parse_args(FindProblemsInput, '{"count": 50, "handle": "tourist"}')
    msg = str(e.value)
    assert "count: Input should be less than or equal to 20" in msg
    assert "handle: Extra inputs are not permitted" in msg


def test_unknown_strategy_is_rejected():
    with pytest.raises(ToolArgsError, match="strategy"):
        parse_args(FindProblemsInput, '{"strategy": "easiest"}')


def test_bad_json_is_an_args_error():
    with pytest.raises(ToolArgsError, match="arguments: Invalid JSON"):
        parse_args(FindProblemsInput, '{"count": 5')


def test_date_is_parsed_from_iso_string():
    assert parse_args(GetProgressSinceInput, '{"since": "2026-10-18"}').since == date(2026, 10, 18)
    with pytest.raises(ToolArgsError, match="since"):
        parse_args(GetProgressSinceInput, '{"since": "last week"}')


def test_nested_plan_errors_name_the_item():
    raw = json.dumps({"items": [ITEM, {"id": "B"}]})
    with pytest.raises(ToolArgsError) as e:
        parse_args(ValidatePlanInput, raw)
    msg = str(e.value)
    assert "items.1.id: String should have at least 2 characters" in msg
    assert "items.1.reason: Field required" in msg
    assert "items.0" not in msg


def test_plan_item_is_just_id_and_reason():
    """Groq checks arguments against this schema; fewer required fields = fewer rejected calls."""
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "validate_plan")
    item = schema["function"]["parameters"]["$defs"]["PlanItem"]
    assert sorted(item["properties"]) == ["id", "reason"]
    assert sorted(item["required"]) == ["id", "reason"]


def test_old_item_format_is_rejected():
    raw = json.dumps({"items": [{"contest_id": 2000, "idx": "B", "tag": "dp", "reason": "r"}]})
    with pytest.raises(ToolArgsError) as e:
        parse_args(ValidatePlanInput, raw)
    msg = str(e.value)
    assert "items.0.id: Field required" in msg
    assert "items.0.contest_id: Extra inputs are not permitted" in msg


def test_plan_needs_at_least_one_item():
    with pytest.raises(ToolArgsError, match="items"):
        parse_args(ValidatePlanInput, '{"items": []}')


def test_save_plan_takes_the_same_arguments_as_validate_plan():
    raw = json.dumps({"items": [ITEM], "constraints": {"hours": 5, "avoid_tags": ["geometry"]}})
    saved = parse_args(SavePlanInput, raw)
    assert saved.constraints.avoid_tags == ["geometry"]
    assert saved.model_dump() == parse_args(ValidatePlanInput, raw).model_dump()
