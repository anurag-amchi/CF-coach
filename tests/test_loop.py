import json
from datetime import datetime, timezone

from cf_coach.agent.llm import LLMReply, ScriptedLLM, ToolCall, ToolCallRejected, scripted_reply
from cf_coach.agent.loop import run_agent, run_tool, user_message
from cf_coach.agent.schemas import PlanConstraints, SavePlanOut, TagProfileOut, TOOL_SCHEMAS
from cf_coach.agent.tools import ToolContext, ToolError

NOW = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)


def fake_profile(ctx, args):
    return TagProfileOut(rating=1400, tags=[])


def fake_save(ctx, args):
    ctx.saved_plan_id = 42
    return SavePlanOut(plan_id=42, items=len(args.items))


def fake_impossible(ctx, args):
    raise ToolError("unknown tag(s) ['DP']")


def fake_crash(ctx, args):
    raise ConnectionError("database is down")


FAKE_TOOLS = {"get_tag_profile": fake_profile, "save_plan": fake_save,
              "find_problems": fake_impossible, "get_upsolve_backlog": fake_crash}
PLAN = json.dumps({"items": [{"contest_id": 2000, "idx": "B", "tag": "dp", "reason": "r"}]})


def call(name, args="{}", id="c1"):
    return ToolCall(id, name, args)


def run(replies, max_steps=10):
    llm = ScriptedLLM(replies)
    ctx = ToolContext("me", session=None, now=NOW)
    return run_agent(llm, ctx, "plan my week", max_steps=max_steps, tools=FAKE_TOOLS), llm


def test_tool_result_goes_back_paired_with_its_call():
    result, llm = run([scripted_reply(calls=[call("get_tag_profile")]), scripted_reply("done")])
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = result.messages[3]
    assert tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"]) == {"rating": 1400, "tags": []}
    assert (result.outcome, result.final_text, result.steps) == ("no_plan", "done", 2)
    assert llm.seen[1][-1] == tool_msg            # the 2nd LLM call saw the result


def test_system_prompt_has_todays_date():
    result, _ = run([scripted_reply("hi")])
    assert "Today is 2026-10-25" in result.messages[0]["content"]


def test_saving_a_plan_sets_outcome_and_plan_id():
    result, _ = run([scripted_reply(calls=[call("save_plan", PLAN)]), scripted_reply("saved!")])
    assert (result.outcome, result.plan_id) == ("saved", 42)


def test_fixable_errors_go_back_to_the_model_and_the_run_continues():
    result, _ = run([
        scripted_reply(calls=[call("save_plan", '{"items": []}', id="a"),     # bad args
                              call("find_problems", '{"tag": "DP"}', id="b"),  # ToolError
                              call("no_such_tool", id="c")]),                  # unknown
        scripted_reply("sorry"),
    ])
    errors = [json.loads(m["content"])["error"] for m in result.messages if m["role"] == "tool"]
    assert errors[0].startswith("bad arguments for save_plan: items:")
    assert errors[1] == "unknown tag(s) ['DP']"
    assert errors[2].startswith("unknown tool 'no_such_tool'")
    assert result.outcome == "no_plan"


def test_rejected_tool_call_is_reported_and_counts_as_a_step():
    result, llm = run([ToolCallRejected("missing properties: 'tag'"), scripted_reply("fixed")])
    note = result.messages[2]
    assert note["role"] == "user" and "missing properties: 'tag'" in note["content"]
    assert (result.outcome, result.steps) == ("no_plan", 2)
    assert llm.seen[1][-1] == note


def test_max_steps_stops_a_model_that_never_finishes():
    replies = [scripted_reply(calls=[call("get_tag_profile", id=f"c{i}")]) for i in range(5)]
    result, _ = run(replies, max_steps=3)
    assert (result.outcome, result.steps, result.final_text) == ("max_steps", 3, None)


def test_unexpected_tool_error_stops_the_run_without_raising():
    result, _ = run([scripted_reply(calls=[call("get_upsolve_backlog")])])
    assert result.outcome == "error"
    assert result.error == "ConnectionError: database is down"


def test_llm_failure_is_an_error_outcome():
    result, _ = run([RuntimeError("Groq is down")])
    assert (result.outcome, result.error, result.steps) == ("error", "RuntimeError: Groq is down", 0)


def test_tokens_are_summed_over_steps():
    first = scripted_reply(calls=[call("get_tag_profile")])
    replies = [LLMReply(None, first.tool_calls, 100, 10, first.message),
               LLMReply("done", [], 150, 20, {"role": "assistant", "content": "done"})]
    result, _ = run(replies)
    assert (result.input_tokens, result.output_tokens) == (250, 30)


def test_run_tool_on_real_schema_names():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert set(FAKE_TOOLS) <= names
    ctx = ToolContext("me", session=None, now=NOW)
    assert json.loads(run_tool("get_tag_profile", "", ctx, FAKE_TOOLS))["rating"] == 1400


def test_no_fixed_constraints_leaves_the_request_alone():
    assert user_message("plan my week", PlanConstraints()) == "plan my week"


def test_fixed_constraints_are_added_to_the_users_message():
    fixed = PlanConstraints(hours=4, avoid_tags=["geometry", "dfs and similar"])
    llm = ScriptedLLM([scripted_reply("ok")])
    ctx = ToolContext("me", session=None, now=NOW, user_constraints=fixed)
    result = run_agent(llm, ctx, "3 dp problems", tools=FAKE_TOOLS)
    assert result.messages[1]["content"] == (
        "3 dp problems\n\nFixed constraints: hours: 4 (at most 4 problems); "
        "avoid tags: geometry, dfs and similar")
