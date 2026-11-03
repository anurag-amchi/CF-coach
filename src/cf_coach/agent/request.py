"""The request parser: one LLM call that turns the user's text into a PlanRequest.

    "Challenge me: 3 hard dp problems."
    -> PlanRequest(intent="plan", count=3, hours=None, focus_tags=["dp"], avoid_tags=[],
                   difficulty="harder")

Why a separate step: limits that exist only in the text ("4 hours", "3 problems", "hard")
were prompt rules the model could ignore (#33). Once they are fields, code enforces them:
read_request() parses and merges the flags, plan_limits() turns the result into
ToolContext fields (count, focus_tags, difficulty, hours, avoided tags), and
validate_plan / find_problems apply those.

The call uses strict structured output: the provider only lets the model produce JSON
that fits the schema, and the two tag lists may only contain CF's real tags (an enum
built from the problemset). Pydantic and a tag check run again on the reply anyway: the
schema is a request to the provider, the check is our rule.

A bad reply gets one retry with the error. If it still fails, or the call itself fails,
parse_request returns EMPTY plus a warning, so a run is never blocked by the parser.

Run (parses the eval requests live):  uv run python -m cf_coach.agent.request
Design reasons: docs/decisions.md #37, #38.
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cf_coach.agent.llm import LLM
from cf_coach.analytics.selection import Difficulty


class PlanRequest(BaseModel):
    """What the user asked for. Every field is required (strict mode needs that):
    null / [] means "the user didn't say"."""
    model_config = ConfigDict(extra="forbid")

    intent: Literal["plan", "question"]
    count: int | None = Field(ge=1, le=20, description="How many problems they asked for.")
    hours: int | None = Field(ge=1, le=40, description="Practice hours they mentioned.")
    focus_tags: list[str] = Field(description="Tags they want to practise.")
    avoid_tags: list[str] = Field(description="Tags they don't want.")
    difficulty: Difficulty | None


EMPTY = PlanRequest(intent="plan", count=None, hours=None, focus_tags=[], avoid_tags=[],
                    difficulty=None)

SYSTEM_PROMPT = """You read one message sent to a Codeforces practice coach and fill in a JSON object.
Fill in only what the user said; use null or [] for anything they did not say.
- intent: "question" only if they ask something about their data or chat, and do not want
  problems (e.g. "What is my weakest tag?"). Anything about practising is "plan", including
  a message that only states limits (e.g. "I have 4 hours this week. No geometry please.").
- count: how many problems they asked for.
- hours: how many hours they want to practise.
- focus_tags: topics they want to practise. avoid_tags: topics they don't want.
  Use Codeforces tag names only ("DP" -> "dp", "graph" -> "graphs").
  Words about difficulty ("hard", "easy") are not tags.
- difficulty: "harder" for hard / challenge / above my level; "easier" for easy / warm-up /
  rusty / revision; "normal" for "my level"; null if they don't say."""

MAX_ATTEMPTS = 2


@dataclass
class ParseResult:
    request: PlanRequest
    warnings: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def request_schema(known_tags: set[str]) -> dict:
    """PlanRequest's JSON schema, with the tag lists limited to CF's real tags."""
    schema = PlanRequest.model_json_schema()
    tags = sorted(known_tags)
    for name in ("focus_tags", "avoid_tags"):
        schema["properties"][name]["items"] = {"type": "string", "enum": tags}
    return schema


def check_reply(text: str | None, known_tags: set[str]) -> PlanRequest:
    """Reply text -> PlanRequest, or ValueError saying what is wrong.

    (Pydantic's ValidationError is a ValueError, so one except catches both.)
    """
    request = PlanRequest.model_validate_json(text or "")
    unknown = [t for t in request.focus_tags + request.avoid_tags if t not in known_tags]
    if unknown:
        raise ValueError(f"unknown tag(s) {unknown}")
    return request


def parse_request(llm: LLM, text: str, known_tags: set[str]) -> ParseResult:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    schema = request_schema(known_tags)
    result = ParseResult(EMPTY)
    error = "no reply"
    for _ in range(MAX_ATTEMPTS):
        try:
            reply = llm.chat(messages, tools=[], response_schema=schema)
        except Exception as e:    # network / provider trouble: plan without the parse
            error = f"{type(e).__name__}: {e}"
            break
        result.input_tokens += reply.input_tokens
        result.output_tokens += reply.output_tokens
        try:
            result.request = check_reply(reply.text, known_tags)
            return result
        except ValueError as e:
            error = str(e)
            messages += [reply.message, {"role": "user", "content":
                         f"That JSON is not valid: {error}. Reply with the corrected JSON only."}]
    result.warnings.append(f"could not read the request ({error}); planning without it")
    return result


def merge_flags(request: PlanRequest, hours: int | None, avoid_tags: list[str],
                difficulty: Difficulty | None) -> PlanRequest:
    """Flags (or form fields) win over the text; avoided tags from both count."""
    return request.model_copy(update={
        "hours": hours if hours is not None else request.hours,
        "avoid_tags": avoid_tags + [t for t in request.avoid_tags if t not in avoid_tags],
        "difficulty": difficulty if difficulty is not None else request.difficulty,
    })


DEFAULT_COUNT = 5      # a week's plan when the user gives neither a count nor hours


def read_request(llm: LLM, session, text: str, hours: int | None, avoid_tags: list[str],
                 difficulty: Difficulty | None) -> tuple[PlanRequest, ParseResult]:
    """Parse the text, check the flags' tags, merge the flags in. Used by the CLI and evals."""
    from cf_coach.agent.tools import all_tags, check_tags
    from cf_coach.analytics.selection import load_problems
    known = all_tags(load_problems(session))
    check_tags(avoid_tags, known)                  # a typo in --avoid -> ToolError up front
    parsed = parse_request(llm, text, known)
    return merge_flags(parsed.request, hours, avoid_tags, difficulty), parsed


def plan_limits(request: PlanRequest) -> dict:
    """PlanRequest -> the ToolContext fields that enforce it:  ToolContext(..., **plan_limits(r))."""
    from cf_coach.agent.schemas import PlanConstraints
    count = request.count
    if count is None and request.hours is None and request.intent == "plan":
        count = DEFAULT_COUNT
    return {"intent": request.intent,
            "user_constraints": PlanConstraints(hours=request.hours,
                                                avoid_tags=list(request.avoid_tags)),
            "difficulty": request.difficulty or "normal",
            "count": count,
            "focus_tags": list(request.focus_tags)}


# ---------- Manual check: the eval requests, parsed by the real model ----------

if __name__ == "__main__":
    from sqlalchemy.orm import Session

    from cf_coach.agent.llm import GroqLLM
    from cf_coach.agent.tools import all_tags
    from cf_coach.analytics.selection import load_problems
    from cf_coach.db.session import get_engine
    from cf_coach.evals.cases import CASES

    with Session(get_engine()) as session:
        known = all_tags(load_problems(session))
    llm = GroqLLM()
    for n, case in enumerate(CASES):
        if n:
            time.sleep(8)      # ~1.3k tokens a call; stay under 8,000 tokens/min
        r = parse_request(llm, case.request, known)
        print(f"{case.name:22} {r.input_tokens:5} in {r.output_tokens:4} out  "
              f"{r.request.model_dump()}  {r.warnings or ''}")
