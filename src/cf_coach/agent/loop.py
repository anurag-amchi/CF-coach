"""The agent loop: model -> tool calls -> results -> model ... until it answers or hits max_steps.

    messages = [system, user]
    repeat up to MAX_STEPS times:
        reply = llm.chat(messages, TOOL_SCHEMAS)
        append reply.message
        no tool calls?  -> done (final answer)
        else run each tool, append one {"role": "tool"} message per call
        a plan was saved?  -> done; code writes the summary (plan_summary), no LLM call
    stopped with text after an ok validate_plan but no save? -> one SAVE_NUDGE, go on

What goes back to the model as a tool result:
- the tool's output, as JSON
- or {"error": ...} when the model can fix it: bad arguments (ToolArgsError), an
  impossible request (ToolError), or a tool name that doesn't exist
If Groq itself rejects a tool call (ToolCallRejected: arguments didn't match the schema),
the error goes back as a user message, and the loop goes on.
Anything else (a bug, the database down) stops the run with outcome "error": the model
can't fix it, and retrying would only burn tokens.

run_agent() never raises and never writes to the database. It returns a RunResult;
save_run() writes that to agent_runs. Every run is logged, including failed ones.

Run on real data:  uv run python -m cf_coach.agent.loop TitanCoder1258 "5 hours, no geometry"
Design reasons: docs/decisions.md #30, #32 (fixed constraints in the user message),
#40 (the run ends at save_plan), #41 (the save nudge), #43 (code picks the week's tags).
"""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from cf_coach.agent.llm import LLM, ToolCall, ToolCallRejected, assistant_message
from cf_coach.agent.schemas import (
    TOOL_SCHEMAS, TOOLS, PlanConstraints, TagProfileOut, ToolArgsError, parse_args,
)
from cf_coach.agent.tools import (
    MIN_ATTEMPTS, MINUTES_PER_PROBLEM, TOOL_FUNCTIONS, ToolContext, ToolError, level_roles,
    plan_size, plan_summary, wanted_count, week_ladder, week_quota,
)
from cf_coach.analytics.selection import DIFFICULTY_GAPS
from cf_coach.db.models import AgentRun

MAX_STEPS = 10     # LLM calls per run. A normal run needs about 5-8.

SYSTEM_PROMPT = """You are a Codeforces practice coach. Today is {today}.
Build a practice plan for this week from the user's own data, using only the tools.

Rules:
- The first tool result is the user's tag profile, already loaded for you. Use
  get_upsolve_backlog and get_progress_since when they help.
- A plan item is {{"id": "<problem id exactly as a tool returned it>", "reason": "..."}}.
  Only plan ids that find_problems or get_upsolve_backlog returned. Never invent ids.
- find_problems lists problems in ladder order (one per level, easiest first). For N
  problems from one tag, take its first N. validate_plan checks the rating band; the
  saved plan is sorted easiest first by code.
- "Fixed constraints" after the message were read from the request and the user's
  settings (number of problems, hours, focus tags, avoided tags, difficulty). Plan exactly
  within them; validate_plan always enforces them. For focus tags, call find_problems
  with each one. If the note lists this week's tags and levels, their problems are
  already loaded: from each tag take one problem per listed level (a problem's level is
  its gap field), and don't search again.
- Call validate_plan before save_plan. If it returns errors, fix all of them and validate again.
- Each `reason` is one short sentence in words only: what the problem trains and its role
  in the week (warm-up, step-up or stretch, as the note names it). Write no digits and no
  gap, level, rating or count: code prints those numbers next to the reason. Every reason
  must be different. Example: "A warm-up on string hashing before the stretch problem."
- When validate_plan returns ok, call the save_plan tool with the same items. Saving ends
  the run: code then shows the plan to the user.
- If the message is a question (the note says so), answer it briefly in plain sentences
  (no tables) from the tools' data; you have only the read-only tools then."""

READ_ONLY_TOOLS = ("get_tag_profile", "get_upsolve_backlog", "find_problems",
                   "get_progress_since")
# Sent once when the model stops after an ok validate_plan without saving: gpt-oss
# sometimes writes the save_plan arguments as text instead of calling the tool (#41).
SAVE_NUDGE = ("validate_plan returned ok, but no plan was saved. Call the save_plan tool "
              "now with the same items: a tool call, not text.")
PROFILE_CALL_ID = "code-get_tag_profile"     # the call code makes before the first step
SEARCH_CALL_ID = "code-find_problems"        # + "-1", "-2"...: the week's searches (#43)
SEARCH_PER_LEVEL = 2                         # problems per level in each week search (#44)
MAX_WEEK_SEARCHES = 10                       # tags tried for the week, joined or skipped (#49)


def allowed_tools(intent: str, tools: dict[str, Callable]) -> dict[str, Callable]:
    """A question gets only the read-only tools: it can't validate or save a plan."""
    if intent == "plan":
        return tools
    return {name: fn for name, fn in tools.items() if name in READ_ONLY_TOOLS}


@dataclass
class RunResult:
    outcome: str                   # saved / no_plan / max_steps / error
    final_text: str | None
    messages: list[dict]
    steps: int                     # LLM calls made
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    plan_id: int | None = None
    error: str | None = None
    model: str = ""
    nudges: int = 0                # SAVE_NUDGE messages sent (0 or 1)


def user_message(request: str, fixed: PlanConstraints, difficulty: str = "normal",
                 count: int | None = None, focus_tags: list[str] = (),
                 intent: str = "plan", week_tags: dict[str, int] | None = None,
                 week_gaps: dict[str, list[int]] | None = None) -> str:
    """The request, plus the user's fixed constraints, so the model plans inside them."""
    notes = []
    if intent == "question":
        notes.append("this is a question: answer it, don't build a plan")
    if (want := wanted_count(count, fixed.hours)) is not None:
        notes.append(f"problems: exactly {want}")
    if focus_tags:
        notes.append(f"focus tags: {', '.join(focus_tags)}")
    if week_tags and week_gaps:
        notes.append(f"this week's tags and levels (your weakest with {MIN_ATTEMPTS}+ "
                     "attempts; their problems are loaded; pick each by its gap field): "
                     + ", ".join(f"{t} {level_roles(week_gaps[t], difficulty)}"
                                 for t in week_tags))
    elif week_tags:
        notes.append(f"this week's tags (your weakest with {MIN_ATTEMPTS}+ attempts; their "
                     "problems are loaded): "
                     + ", ".join(f"{t} x{n}" for t, n in week_tags.items()))
    if difficulty != "normal":
        notes.append(f"difficulty: {difficulty}")
    if fixed.hours is not None:
        most = fixed.hours * 60 // MINUTES_PER_PROBLEM
        notes.append(f"hours: {fixed.hours} (at most {most} problems)")
    if fixed.avoid_tags:
        notes.append(f"avoid tags: {', '.join(fixed.avoid_tags)}")
    if not notes:
        return request
    return f"{request}\n\nFixed constraints: {'; '.join(notes)}"


def run_tool(name: str, raw_args: str, ctx: ToolContext,
             tools: dict[str, Callable]) -> str:
    """One tool call -> the JSON text sent back to the model.

    Errors the model can fix become {"error": ...}. Any other exception is raised.
    """
    if name not in tools:
        return json.dumps({"error": f"unknown tool '{name}'. Tools: {', '.join(tools)}"})
    try:
        args = parse_args(TOOLS[name][1], raw_args)
        return tools[name](ctx, args).model_dump_json()
    except ToolArgsError as e:
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
    except ToolError as e:
        return json.dumps({"error": str(e)})


def pick_week(ctx: ToolContext, profile_json: str,
              tools: dict[str, Callable]) -> tuple[list[ToolCall], list[str]]:
    """For a plan request with no focus tags: choose the week's tags (week_quota) and the
    level of each problem (week_ladder), searching each tag first (#43, #44, #49).

    Measured tags are searched weakest first, SEARCH_PER_LEVEL problems per level of the
    band. A tag joins the week only if its search has a problem at EVERY level; otherwise
    it is skipped (its offers are forgotten) and the next weakest is tried, at most
    MAX_WEEK_SEARCHES searches. So whatever level week_ladder gives a tag, it exists.

    Sets ctx.tag_quota, ctx.tag_gaps and ctx.week_skipped. Returns the calls and results
    of the tags that joined, for the messages. If the profile can't be read or too few tags
    fill the band, no rule is set and the model plans as before (with what was found).
    """
    try:
        tags = TagProfileOut.model_validate_json(profile_json).tags
    except ValueError:                            # the profile call returned an error
        return [], []
    size = plan_size(ctx.count, ctx.user_constraints.hours) or 0
    avoid = ctx.user_constraints.avoid_tags
    measured = [t for t in tags if t.attempts >= MIN_ATTEMPTS and t.tag not in avoid]
    need = len(week_quota(measured, size))        # how many tags the week uses
    if need == 0:
        return [], []
    lo, hi = DIFFICULTY_GAPS[ctx.difficulty]
    levels = set(range(lo, hi + 1, 100))
    per_tag = SEARCH_PER_LEVEL * len(levels)
    joined, calls, results = [], [], []
    for t in measured[:MAX_WEEK_SEARCHES]:
        call = ToolCall(f"{SEARCH_CALL_ID}-{len(calls) + 1}", "find_problems",
                        json.dumps({"tag": t.tag, "count": per_tag}))
        before = set(ctx.offered)
        content = run_tool(call.name, call.arguments, ctx, tools)
        found = json.loads(content).get("problems", [])
        if levels <= {p["gap"] for p in found}:
            joined.append(t)
            calls.append(call)
            results.append(content)
            if len(joined) == need:
                break
        else:                       # can't fill the band: not shown, not plannable
            for pid in set(ctx.offered) - before:
                del ctx.offered[pid]
            ctx.week_skipped.append(t.tag)
    if len(joined) < need:          # too few tags fill the band: no week rules
        return calls, results
    ctx.tag_quota = week_quota(joined, size)
    ctx.tag_gaps = week_ladder(ctx.tag_quota, ctx.difficulty)
    return calls, results


def run_agent(llm: LLM, ctx: ToolContext, request: str, max_steps: int = MAX_STEPS,
              tools: dict[str, Callable] = TOOL_FUNCTIONS,
              on_message: Callable[[dict], None] = lambda m: None,
              start_with_profile: bool = True) -> RunResult:
    """Run the loop for one user request. `on_message` sees every new message (CLI, SSE).

    start_with_profile: code calls get_tag_profile before the first model step and adds
    the call + result to the messages, so every run starts from the user's data (this was
    a prompt rule) and saves one LLM step.
    """
    started = time.monotonic()
    tools = allowed_tools(ctx.intent, tools)
    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in tools]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=ctx.now.date())},
        {"role": "user", "content": user_message(request, ctx.user_constraints, ctx.difficulty,
                                                 ctx.count, ctx.focus_tags, ctx.intent)},
    ]
    result = RunResult(outcome="max_steps", final_text=None, messages=messages, steps=0,
                       model=llm.model)

    def add(msg: dict) -> None:
        messages.append(msg)
        on_message(msg)

    try:
        if start_with_profile:
            call = ToolCall(PROFILE_CALL_ID, "get_tag_profile", "{}")
            profile_json = run_tool(call.name, call.arguments, ctx, tools)
            searches, found = [], []
            if ctx.intent == "plan" and not ctx.focus_tags:      # code picks the week's tags
                searches, found = pick_week(ctx, profile_json, tools)
                if ctx.tag_quota:
                    messages[1]["content"] = user_message(
                        request, ctx.user_constraints, ctx.difficulty, ctx.count,
                        ctx.focus_tags, ctx.intent, ctx.tag_quota, ctx.tag_gaps)
            add(assistant_message(None, [call]))
            add({"role": "tool", "tool_call_id": call.id, "content": profile_json})
            if searches:
                add(assistant_message(None, searches))
                for c, content in zip(searches, found):
                    add({"role": "tool", "tool_call_id": c.id, "content": content})
        while result.steps < max_steps:
            try:
                reply = llm.chat(messages, schemas)
            except ToolCallRejected as e:            # the model can fix it: tell it, go on
                result.steps += 1
                add({"role": "user", "content": f"Your last tool call was rejected and not "
                                                f"run: {e}. Fix the arguments and call again."})
                continue
            result.steps += 1
            result.input_tokens += reply.input_tokens
            result.output_tokens += reply.output_tokens
            add(reply.message)
            if not reply.tool_calls:
                if ctx.validated_ok and ctx.saved_plan_id is None and result.nudges == 0:
                    result.nudges += 1               # a valid plan, but stopped before saving
                    add({"role": "user", "content": SAVE_NUDGE})
                    continue
                result.final_text = reply.text
                result.outcome = "saved" if ctx.saved_plan_id is not None else "no_plan"
                break
            for call in reply.tool_calls:
                content = run_tool(call.name, call.arguments, ctx, tools)
                add({"role": "tool", "tool_call_id": call.id, "content": content})
            if ctx.saved_plan_id is not None:    # saved: code writes the summary, no more steps
                result.outcome = "saved"
                result.final_text = plan_summary(*ctx.saved_plan) if ctx.saved_plan else None
                break
    except Exception as e:                       # a bug or an outage: stop, but still log
        result.outcome = "error"
        result.error = f"{type(e).__name__}: {e}"
        if ctx.session is not None:
            ctx.session.rollback()               # so save_run can use the session

    result.plan_id = ctx.saved_plan_id
    result.latency_ms = round((time.monotonic() - started) * 1000)
    return result


def save_run(session: Session, handle: str, result: RunResult) -> int:
    """Write one agent_runs row. Returns its id."""
    steps = list(result.messages)
    if result.error:
        steps.append({"role": "error", "content": result.error})
    run = AgentRun(handle=handle, plan_id=result.plan_id, model=result.model, steps=steps,
                   input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                   latency_ms=result.latency_ms, outcome=result.outcome)
    session.add(run)
    session.commit()
    return run.id


# ---------- Manual check: one real run ----------

if __name__ == "__main__":
    from cf_coach.agent.llm import GroqLLM
    from cf_coach.db.session import get_engine

    handle, request = sys.argv[1], sys.argv[2]

    def show(msg):
        if msg["role"] == "user":
            print(f"   !! {msg['content'][:150]}")
        if msg["role"] == "tool":
            print(f"   <- {msg['content'][:150]}")
        for c in msg.get("tool_calls", []):
            print(f"-> {c['function']['name']}({c['function']['arguments']})")

    with Session(get_engine()) as session:
        ctx = ToolContext(handle, session, datetime.now(timezone.utc))
        result = run_agent(GroqLLM(), ctx, request, on_message=show)
        run_id = save_run(session, handle, result)

    print(f"\n{result.final_text}\n")
    print(f"outcome={result.outcome} plan_id={result.plan_id} steps={result.steps} "
          f"tokens={result.input_tokens} in / {result.output_tokens} out "
          f"latency={result.latency_ms} ms  -> agent_runs id {run_id}")
    if result.error:
        print("error:", result.error)
