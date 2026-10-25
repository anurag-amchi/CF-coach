"""The agent loop: model -> tool calls -> results -> model ... until it answers or hits max_steps.

    messages = [system, user]
    repeat up to MAX_STEPS times:
        reply = llm.chat(messages, TOOL_SCHEMAS)
        append reply.message
        no tool calls?  -> done (final answer)
        else run each tool, append one {"role": "tool"} message per call

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
Design reasons: docs/decisions.md #30, #32 (fixed constraints in the user message).
"""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from cf_coach.agent.llm import LLM, ToolCallRejected
from cf_coach.agent.schemas import (
    TOOL_SCHEMAS, TOOLS, PlanConstraints, ToolArgsError, parse_args,
)
from cf_coach.agent.tools import MINUTES_PER_PROBLEM, TOOL_FUNCTIONS, ToolContext, ToolError
from cf_coach.db.models import AgentRun

MAX_STEPS = 10     # LLM calls per run. A normal run needs about 5-8.

SYSTEM_PROMPT = """You are a Codeforces practice coach. Today is {today}.
Build a practice plan for this week from the user's own data, using only the tools.

Rules:
- Start with get_tag_profile. Use get_upsolve_backlog and get_progress_since when they help.
- Only plan problems that a tool returned. Never invent problem ids.
- Every problem must be rated 100-300 above the user's skill on the tag you give it.
- Read the user's constraints (hours, tags to avoid) from their message and pass them in
  `constraints`. If they give no hours, plan 5 problems.
- "Fixed constraints" after the message come from the user's settings: plan within them.
  validate_plan always enforces them.
- Call validate_plan before save_plan. If it returns errors, fix all of them and validate again.
- Each `reason` is one sentence that cites the user's data (skill, abandon rate, backlog).
- After save_plan succeeds, reply with a short summary of the plan and call no more tools.
- If the message is not about practice planning, answer briefly without tools."""


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


def user_message(request: str, fixed: PlanConstraints) -> str:
    """The request, plus the user's fixed constraints, so the model plans inside them."""
    notes = []
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


def run_agent(llm: LLM, ctx: ToolContext, request: str, max_steps: int = MAX_STEPS,
              tools: dict[str, Callable] = TOOL_FUNCTIONS,
              on_message: Callable[[dict], None] = lambda m: None) -> RunResult:
    """Run the loop for one user request. `on_message` sees every new message (CLI, SSE)."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=ctx.now.date())},
        {"role": "user", "content": user_message(request, ctx.user_constraints)},
    ]
    result = RunResult(outcome="max_steps", final_text=None, messages=messages, steps=0,
                       model=llm.model)

    def add(msg: dict) -> None:
        messages.append(msg)
        on_message(msg)

    try:
        while result.steps < max_steps:
            try:
                reply = llm.chat(messages, TOOL_SCHEMAS)
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
                result.final_text = reply.text
                result.outcome = "saved" if ctx.saved_plan_id is not None else "no_plan"
                break
            for call in reply.tool_calls:
                content = run_tool(call.name, call.arguments, ctx, tools)
                add({"role": "tool", "tool_call_id": call.id, "content": content})
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
