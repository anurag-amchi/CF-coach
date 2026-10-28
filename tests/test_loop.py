import json
from datetime import datetime, timezone

from cf_coach.agent.llm import LLMReply, ScriptedLLM, ToolCall, ToolCallRejected, scripted_reply
from cf_coach.agent.loop import (
    PROFILE_CALL_ID, READ_ONLY_TOOLS, SAVE_NUDGE, SEARCH_CALL_ID, allowed_tools, run_agent,
    run_tool, user_message,
)
from cf_coach.agent.schemas import (
    FindProblemsOut, PlanConstraints, ProblemOut, SavePlanOut, TagProfileOut, TagSkillOut,
    TOOL_SCHEMAS, ValidationOut,
)
from cf_coach.agent.tools import (
    SavedConstraints, SavedItem, ToolContext, ToolError, plan_summary,
)

NOW = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)


def fake_profile(ctx, args):
    return TagProfileOut(rating=1400, tags=[])


SAVED_ITEM = SavedItem(id="2000B", contest_id=2000, idx="B", tag="dp", rating=1500, gap=100,
                       facts="f", reason="r")


def fake_save(ctx, args):
    ctx.saved_plan_id = 42
    ctx.saved_plan = (SavedConstraints(), [SAVED_ITEM])
    return SavePlanOut(plan_id=42, items=len(args.items))


def fake_impossible(ctx, args):
    raise ToolError("unknown tag(s) ['DP']")


def fake_crash(ctx, args):
    raise ConnectionError("database is down")


FAKE_TOOLS = {"get_tag_profile": fake_profile, "save_plan": fake_save,
              "find_problems": fake_impossible, "get_upsolve_backlog": fake_crash}
PLAN = json.dumps({"items": [{"id": "2000B", "reason": "r"}]})


def call(name, args="{}", id="c1"):
    return ToolCall(id, name, args)


def run(replies, max_steps=10, start_with_profile=False, intent="plan"):
    """Most tests switch the code-side profile call off, so the messages are only the model's."""
    llm = ScriptedLLM(replies)
    ctx = ToolContext("me", session=None, now=NOW, intent=intent)
    return run_agent(llm, ctx, "plan my week", max_steps=max_steps, tools=FAKE_TOOLS,
                     start_with_profile=start_with_profile), llm


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


def test_saving_a_plan_ends_the_run_with_a_code_written_summary():
    result, llm = run([scripted_reply(calls=[call("save_plan", PLAN)]), scripted_reply("| a table |")])
    assert (result.outcome, result.plan_id) == ("saved", 42)
    assert result.steps == 1 and len(llm.seen) == 1          # no LLM call after the save
    assert result.final_text == plan_summary(SavedConstraints(), [SAVED_ITEM])
    assert result.messages[-1]["role"] == "tool"              # the save result is the last message


def fake_validate_ok(ctx, args):
    ctx.validated_ok = True
    return ValidationOut(ok=True, errors=[])


def run_with_validate(replies):
    llm = ScriptedLLM(replies)
    ctx = ToolContext("me", session=None, now=NOW)
    tools = {**FAKE_TOOLS, "validate_plan": fake_validate_ok}
    return run_agent(llm, ctx, "plan my week", tools=tools, start_with_profile=False), llm


def test_save_written_as_text_after_an_ok_validation_gets_one_nudge():
    result, llm = run_with_validate([
        scripted_reply(calls=[call("validate_plan", PLAN)]),
        scripted_reply(PLAN),                                  # the slip: the save as text
        scripted_reply(calls=[call("save_plan", PLAN, id="c2")]),
    ])
    assert (result.outcome, result.plan_id, result.nudges, result.steps) == ("saved", 42, 1, 3)
    assert {"role": "user", "content": SAVE_NUDGE} in result.messages
    assert llm.seen[2][-1] == {"role": "user", "content": SAVE_NUDGE}   # the 3rd call saw it


def test_the_nudge_is_sent_only_once():
    result, _ = run_with_validate([
        scripted_reply(calls=[call("validate_plan", PLAN)]),
        scripted_reply(PLAN),
        scripted_reply("I won't save it."),
    ])
    assert (result.outcome, result.nudges, result.final_text) == ("no_plan", 1, "I won't save it.")


def test_a_refusal_without_an_ok_validation_is_not_nudged():
    result, _ = run_with_validate([scripted_reply("No problems are rated that high for you.")])
    assert (result.outcome, result.nudges, result.steps) == ("no_plan", 0, 1)


# ---------- code picks the week's tags and searches them first (decision #43) ----------

def week_profile(ctx, args):
    rows = [("dfs and similar", 3), ("geometry", 7), ("strings", 32), ("trees", 5), ("dp", 26)]
    return TagProfileOut(rating=1400, tags=[
        TagSkillOut(tag=t, skill=1400, attempts=a, abandon_rate=0.0) for t, a in rows])


def search_returning(per_tag):
    """A fake find_problems that returns `per_tag` problems for any tag."""
    def search(ctx, args):
        return FindProblemsOut(difficulty="normal", problems=[
            ProblemOut(id=f"{n}{args.tag[0].upper()}", name="x", rating=1400 + 100 * n,
                       tags=[args.tag], matched_tag=args.tag, tag_skill=1400, gap=100 * n,
                       score=1.0)
            for n in range(min(per_tag, args.count))])
    return search


def run_week(per_tag, count=4, focus=()):
    llm = ScriptedLLM([scripted_reply("ok")])
    ctx = ToolContext("me", session=None, now=NOW, count=count, focus_tags=list(focus))
    tools = {**FAKE_TOOLS, "get_tag_profile": week_profile, "find_problems": search_returning(per_tag)}
    return run_agent(llm, ctx, "plan my week", tools=tools), ctx, llm


def test_a_week_with_no_tags_gets_its_tags_and_searches_from_code():
    result, ctx, llm = run_week(per_tag=3, count=4)
    assert ctx.tag_quota == {"geometry": 2, "strings": 1, "trees": 1}   # dfs: only 3 attempts
    assert ctx.tag_gaps == {"geometry": [0, 200], "strings": [0], "trees": [100]}
    assert "this week's tags and levels (your weakest with 5+ attempts; their problems are " \
           "loaded; pick each by its gap field): geometry warm-up (gap +0) and stretch " \
           "(gap +200), strings warm-up (gap +0), trees step-up (gap +100)" \
           in result.messages[1]["content"]
    searches = result.messages[4]["tool_calls"]
    assert [(c["id"], json.loads(c["function"]["arguments"])) for c in searches] == [
        (f"{SEARCH_CALL_ID}-1", {"tag": "geometry", "count": 6}),   # 2 per level x 3 levels
        (f"{SEARCH_CALL_ID}-2", {"tag": "strings", "count": 6}),
        (f"{SEARCH_CALL_ID}-3", {"tag": "trees", "count": 6})]
    assert [m["role"] for m in result.messages[5:8]] == ["tool"] * 3
    assert result.steps == 1 and llm.seen[0][-1] == result.messages[7]   # no LLM step spent


def test_a_missing_level_drops_only_the_level_rule():
    result, ctx, _ = run_week(per_tag=2, count=4)       # gaps +0, +100 only; geometry needs +200
    assert ctx.tag_quota == {"geometry": 2, "strings": 1, "trees": 1}
    assert ctx.tag_gaps == {}
    assert "this week's tags (your weakest with 5+ attempts; their problems are loaded): " \
           "geometry x2, strings x1, trees x1" in result.messages[1]["content"]


def test_a_tag_with_too_few_problems_drops_the_quota():
    result, ctx, _ = run_week(per_tag=1, count=4)       # geometry needs 2, only 1 exists
    assert ctx.tag_quota == {} and ctx.tag_gaps == {}
    assert "this week's tags" not in result.messages[1]["content"]


def test_focus_tags_or_questions_get_no_week_tags():
    _, ctx, _ = run_week(per_tag=3, focus=["dp"])
    assert ctx.tag_quota == {}


def test_a_failed_save_does_not_end_the_run():
    def refuse(ctx, args):
        raise ToolError("plan not saved, it breaks these rules: ...")
    llm = ScriptedLLM([scripted_reply(calls=[call("save_plan", PLAN)]), scripted_reply("sorry")])
    ctx = ToolContext("me", session=None, now=NOW)
    result = run_agent(llm, ctx, "plan my week", tools={**FAKE_TOOLS, "save_plan": refuse},
                       start_with_profile=False)
    assert (result.outcome, result.steps, result.final_text) == ("no_plan", 2, "sorry")


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
    assert user_message("plan my week", PlanConstraints(), "harder") == (
        "plan my week\n\nFixed constraints: difficulty: harder")
    assert user_message("5 dp", PlanConstraints(hours=3), count=5, focus_tags=["dp"]) == (
        "5 dp\n\nFixed constraints: problems: exactly 3; focus tags: dp; "
        "hours: 3 (at most 3 problems)")


def test_fixed_constraints_are_added_to_the_users_message():
    fixed = PlanConstraints(hours=4, avoid_tags=["geometry", "dfs and similar"])
    llm = ScriptedLLM([scripted_reply("ok")])
    ctx = ToolContext("me", session=None, now=NOW, user_constraints=fixed)
    result = run_agent(llm, ctx, "3 dp problems", tools=FAKE_TOOLS)
    assert result.messages[1]["content"] == (
        "3 dp problems\n\nFixed constraints: hours: 4 (at most 4 problems); "
        "avoid tags: geometry, dfs and similar")


# ---------- profile first in code, read-only tools for questions (decision #39) ----------

def test_code_loads_the_profile_before_the_first_step():
    result, llm = run([scripted_reply("done")], start_with_profile=True)
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    call = result.messages[2]["tool_calls"][0]
    assert call["function"]["name"] == "get_tag_profile" and call["id"] == PROFILE_CALL_ID
    assert result.messages[3]["tool_call_id"] == PROFILE_CALL_ID
    assert json.loads(result.messages[3]["content"]) == {"rating": 1400, "tags": []}
    assert result.steps == 1                       # no LLM step was spent on it
    assert llm.seen[0][-1] == result.messages[3]   # the model's first call already sees it


def test_a_question_gets_only_read_only_tools():
    result, llm = run([scripted_reply(calls=[call("save_plan", PLAN)]), scripted_reply("ok")],
                      intent="question")
    assert "save_plan" not in llm.tool_names[0] and "get_tag_profile" in llm.tool_names[0]
    error = json.loads(result.messages[3]["content"])["error"]
    assert error.startswith("unknown tool 'save_plan'")       # and nothing was saved
    assert result.outcome == "no_plan" and result.plan_id is None
    assert "this is a question" in result.messages[1]["content"]


def test_allowed_tools():
    assert allowed_tools("plan", FAKE_TOOLS) == FAKE_TOOLS
    assert set(allowed_tools("question", FAKE_TOOLS)) == {
        "get_tag_profile", "find_problems", "get_upsolve_backlog"}
    assert set(READ_ONLY_TOOLS) < set(TOOL_SCHEMAS_NAMES)


TOOL_SCHEMAS_NAMES = [s["function"]["name"] for s in TOOL_SCHEMAS]
