"""Pydantic models for the agent's 6 tools: what the model may send, and what it gets back.

Each tool has one input class, and that class does two jobs:
  1. tool_schema() turns it into the JSON schema the model is shown (TOOL_SCHEMAS).
  2. parse_args() checks the model's raw JSON arguments against it.
The schema the model sees and the check we run come from the same class, so they can't
disagree.

The user's handle is NOT a tool argument. The loop passes it in, so the model can only
read and plan for the user it is running for.

Outputs are models too: a tool returns one, and the loop sends model_dump_json() back.

Run (prints the schemas, then tries some arguments):  uv run python -m cf_coach.agent.schemas
Add --live to send the schemas to the real model once.
Design reasons: docs/decisions.md #25.
"""

import json
import sys
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ---------- Inputs (what the model sends) ----------

class ToolInput(BaseModel):
    # An unknown argument is an error, not silently dropped, so the model hears about it.
    model_config = ConfigDict(extra="forbid")


class GetTagProfileInput(ToolInput):
    """No arguments: the profile is always the current user's."""


class GetUpsolveBacklogInput(ToolInput):
    limit: int = Field(10, ge=1, le=50, description="How many problems to return, newest first.")


class FindProblemsInput(ToolInput):
    tag: str | None = Field(None, description="Only problems with this tag, e.g. 'dp'. "
                                              "Leave it out to search every tag.")
    strategy: Literal["weakness_first", "contest_prep"] = Field(
        "weakness_first",
        description="weakness_first: the user's weakest tags first. "
                    "contest_prep: recent contest problems, in tags where the user is slow.")
    avoid_tags: list[str] = Field(default_factory=list,
                                    description="Skip problems that have any of these tags.")
    count: int = Field(10, ge=1, le=20)


class GetProgressSinceInput(ToolInput):
    since: date = Field(description="An ISO date, e.g. '2026-10-18'.")


class PlanItem(ToolInput):
    # Only 2 fields: the tag and the contest/index split come from the tool that offered
    # the id (ToolContext.offered), so the model can't get them wrong. Decision #35.
    id: str = Field(min_length=2, max_length=12,
                    description="The problem id exactly as a tool returned it, e.g. '2242D'.")
    reason: str = Field(min_length=1, max_length=300,
                        description="One sentence, shown to the user.")


class PlanConstraints(ToolInput):
    hours: int | None = Field(None, ge=1, le=40, description="Practice hours this week.")
    avoid_tags: list[str] = Field(default_factory=list,
                                  description="Tags the user asked not to practise.")


class ValidatePlanInput(ToolInput):
    items: list[PlanItem] = Field(min_length=1, max_length=30)
    constraints: PlanConstraints = Field(default_factory=PlanConstraints)


class SavePlanInput(ValidatePlanInput):
    """Same fields as validate_plan. save_plan runs the validation again before it writes."""


# ---------- Outputs (what the tools send back) ----------

class TagSkillOut(BaseModel):
    tag: str
    skill: int                     # rounded: decimals are noise and cost tokens
    attempts: int
    abandon_rate: float


class TagProfileOut(BaseModel):
    rating: int
    tags: list[TagSkillOut]


class BacklogItemOut(BaseModel):
    id: str                        # e.g. "2242D": what a plan item names
    rating: int | None
    tags: list[str]
    last_tried: date


class BacklogOut(BaseModel):
    total: int                     # the full backlog size, even when `items` is cut to `limit`
    items: list[BacklogItemOut]


class ProblemOut(BaseModel):
    id: str
    name: str
    rating: int
    tags: list[str]
    matched_tag: str               # the tag whose band it fell in
    tag_skill: int
    gap: int                       # rating above the rounded tag skill: its ladder level
    score: float


class FindProblemsOut(BaseModel):
    difficulty: str                # the band used: easier / normal / harder
    problems: list[ProblemOut]     # in ladder order: +0, +100, +200, +0, ...


class SolvedOut(BaseModel):
    id: str                        # same id format as the other tools
    rating: int | None
    tags: list[str]
    solved_on: date


class TagChangeOut(BaseModel):
    tag: str
    before: int
    after: int


class ProgressOut(BaseModel):
    since: date
    solved_total: int              # every problem first solved since then
    solved: list[SolvedOut]        # the newest ones, cut to a maximum
    skill_changes: list[TagChangeOut]   # only tags that moved, biggest move first


class ValidationOut(BaseModel):
    ok: bool
    errors: list[str]              # one line per broken rule; empty when ok


class SavePlanOut(BaseModel):
    plan_id: int
    items: int


# ---------- Schemas for the model, and argument checking ----------

# name -> (description shown to the model, input class)
TOOLS: dict[str, tuple[str, type[ToolInput]]] = {
    "get_tag_profile": (
        "The user's CF rating and their skill estimate (Elo-style), attempt count and "
        "abandon rate for every tag they have tried.", GetTagProfileInput),
    "get_upsolve_backlog": (
        "Contest problems the user submitted to and never solved, newest first.",
        GetUpsolveBacklogInput),
    "find_problems": (
        "Unsolved problems at the user's level on a tag (skill rounded to 100; the band "
        "comes from the requested difficulty), in ladder order: one per level, easiest "
        "first, then repeating. Taking the first N gives a plan that climbs evenly. Never "
        "returns problems from contests that are still running.",
        FindProblemsInput),
    "get_progress_since": (
        "Problems the user solved since a date, and how their tag skills changed.",
        GetProgressSinceInput),
    "validate_plan": (
        "Check a weekly plan against the rules (every id was returned by find_problems or "
        "get_upsolve_backlog in this conversation, unsolved, in the skill band, not from a "
        "running contest, no avoided tags). Returns every broken rule.",
        ValidatePlanInput),
    "save_plan": (
        "Save the plan for the user. Call only after validate_plan returned ok=true.",
        SavePlanInput),
}


def tool_schema(name: str, description: str, model: type[ToolInput]) -> dict:
    """One tool in the OpenAI format: parameters = the class's JSON schema."""
    params = model.model_json_schema()
    params.pop("title", None)
    params.pop("description", None)    # the class docstring is for us, not the model
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": params}}


TOOL_SCHEMAS = [tool_schema(name, desc, model) for name, (desc, model) in TOOLS.items()]


class ToolArgsError(ValueError):
    """The model's arguments are bad. The message is written for the model to read."""


def parse_args[T: ToolInput](model: type[T], raw: str) -> T:
    """Raw JSON string from the model -> a checked input object, or ToolArgsError.

    One error line per problem, e.g. "items.0.reason: Field required".
    """
    try:
        return model.model_validate_json(raw.strip() or "{}")
    except ValidationError as e:
        lines = []
        for err in e.errors():
            where = ".".join(str(part) for part in err["loc"]) or "arguments"
            lines.append(f"{where}: {err['msg']}")
        raise ToolArgsError("; ".join(lines)) from None


# ---------- Manual check ----------

if __name__ == "__main__":
    print(json.dumps(TOOL_SCHEMAS, indent=1), "\n")

    tries = [
        ("find_problems", '{"tag": "dp", "count": 5}'),
        ("find_problems", '{"count": 50, "handle": "tourist"}'),
        ("find_problems", '{"strategy": "easiest"}'),
        ("get_progress_since", '{"since": "last week"}'),
        ("validate_plan", '{"items": [{"id": "B"}]}'),
        ("validate_plan", '{"items": [{"contest_id": 2000, "idx": "B", "reason": "x"}]}'),
        ("get_tag_profile", ""),
        ("get_upsolve_backlog", '{"limit": 5'),
    ]
    for name, raw in tries:
        try:
            print(f"OK    {name}({raw}) -> {parse_args(TOOLS[name][1], raw)!r}")
        except ToolArgsError as e:
            print(f"ERROR {name}({raw}) -> {e}")

    if "--live" in sys.argv:
        from cf_coach.agent.llm import GroqLLM
        reply = GroqLLM().chat([{"role": "user", "content":
            "Find me 3 dp problems for practice, but nothing with geometry."}], TOOL_SCHEMAS)
        print(f"\nlive: {reply.input_tokens} tokens in, text={reply.text!r}")
        for c in reply.tool_calls:
            print(f"  {c.name}({c.arguments}) -> {parse_args(TOOLS[c.name][1], c.arguments)!r}")
