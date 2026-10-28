"""The web API contract: every endpoint, who may call it, and the JSON it takes and returns.

Written before any FastAPI code (slice 2 of decision #51). In slice 3 FastAPI builds its
routes from these models, so its OpenAPI schema (/openapi.json) is this file, and the
frontend's TypeScript types are generated from that schema. A test then checks that the
real routes match ENDPOINTS.

These are API models, separate from the tool models (agent/schemas.py) and the tables
(db/models.py): the frontend depends on this shape, so internal changes don't leak into it.

All paths start with /api: the Next.js app proxies /api/* to FastAPI, so the browser sees
one site and the session cookie is first-party.

Who may call (Auth):
  public    anyone, no cookie needed
  guest_ok  a guest (guest cookie) or a signed-in account
  account   a signed-in account only; a guest gets 401 sign_in_required

Every error is ErrorOut: {"error": {"code": ..., "message": ..., "retry_after_s": ...}}.

Print the table and an example event stream:  uv run python -m cf_coach.api.contract
Design reasons: docs/decisions.md #51, #52, #53 (Firebase auth).
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Difficulty = Literal["easier", "normal", "harder"]
Auth = Literal["public", "guest_ok", "account"]


class In(BaseModel):
    """A request body. An unknown field is a 422, not silently dropped."""
    model_config = ConfigDict(extra="forbid")


class Out(BaseModel):
    """A response body."""


# ---------- Errors ----------

ErrorCode = Literal[
    "validation",          # 422: the body or query is wrong
    "not_found",           # 404: no such handle / plan / run
    "unauthorized",        # 401: the ID token or session cookie is bad or expired
    "sign_in_required",    # 401: a guest called an account-only endpoint
    "email_not_verified",  # 403: a password account before its email link was opened
    "rate_limited",        # 429: too many tries; retry_after_s says when
    "guest_limit",         # 429: a guest already made today's plan
    "budget_exhausted",    # 503: today's token budget is spent
    "run_in_progress",     # 409: this account or guest already has a run going
    "cf_unavailable",      # 502: the Codeforces API failed after retries
]


class ErrorBody(Out):
    code: ErrorCode
    message: str                   # shown to the user as is
    retry_after_s: int | None = None


class ErrorOut(Out):
    error: ErrorBody


# ---------- Auth (slices 4a-4b, Firebase: decision #53) ----------
# The browser signs in with the Firebase JS SDK (email + password or Google); Firebase
# also sends the verify and reset emails. The API never sees a password. The browser
# posts the ID token Firebase gives it once; the API verifies it and sets an httpOnly
# session cookie, which every later request (and the SSE stream) carries.

class SessionIn(In):
    id_token: str = Field(min_length=100, max_length=4096)   # a Firebase ID token (a JWT)


class MeOut(Out):
    email: str
    email_verified: bool
    sign_in_provider: Literal["password", "google.com"]
    handle: str | None             # the CF handle this account plans for; None until set


class MeUpdateIn(In):
    handle: str = Field(min_length=3, max_length=24, pattern=r"^[A-Za-z0-9_.-]+$")


# ---------- Sync and profile (slices 3, 8) ----------

class SyncStatusOut(Out):
    handle: str
    state: Literal["never", "queued", "running", "done", "error"]
    synced_at: datetime | None
    new_submissions: int | None    # from the last finished sync
    error: str | None = None


class TagOut(Out):
    tag: str
    skill: int                     # the Elo estimate, rounded
    level: int                     # skill rounded to 100: where its plan problems start
    attempts: int
    solved: int
    abandon_rate: float            # 0.0 to 1.0
    avg_wrong_before_ac: float | None
    backlog: int                   # unsolved contest problems with this tag


class ProfileOut(Out):
    handle: str
    rating: int | None             # None for unrated users
    synced_at: datetime
    solved_total: int
    backlog_total: int
    median_solve_minutes: float | None
    tags: list[TagOut]             # weakest first, by level


class SolvedOut(Out):
    id: str                        # e.g. "2242D"
    rating: int | None
    tags: list[str]
    solved_on: date


class TagChangeOut(Out):
    tag: str
    before: int
    after: int


class ProgressOut(Out):
    since: date
    solved_total: int
    solved: list[SolvedOut]        # newest first, cut to 50
    skill_changes: list[TagChangeOut]


# ---------- Plans (slices 5, 7) ----------

class PlanItemOut(Out):
    id: str                        # the problem id, e.g. "2242D"; also the item's id
    name: str
    url: str                       # the problem on codeforces.com
    tag: str                       # the tag it practises
    rating: int
    gap: int                       # rating above the tag's level: its rung on the ladder
    facts: str                     # numbers written by code
    reason: str                    # one sentence written by the model
    solved: bool                   # worked out at read time from the latest sync


class PlanOut(Out):
    # A string, because a guest plan lives in Redis with a random id ("g_...");
    # a saved plan's id is its Postgres row id ("123").
    id: str
    handle: str
    created_at: datetime
    difficulty: Difficulty
    hours: int | None
    avoid_tags: list[str]
    summary: str                   # plan_summary(): written by code
    items: list[PlanItemOut]       # easiest first
    saved: bool                    # False = a guest plan, gone at expires_at
    expires_at: datetime | None


class ItemActionIn(In):
    # 0 tokens: code picks another offered problem on the same tag at the same rung
    # (swap), one rung down (easier) or one rung up (harder).
    action: Literal["swap", "easier", "harder"]


class PlansOut(Out):
    plans: list[PlanOut]           # newest first


# ---------- Agent runs (slices 6, 9) ----------

class RunIn(In):
    # plan: build a new plan (guests may, once a day)
    # adjust: change the current plan in words (account only, needs plan_id)
    # question: ask about your data, read-only tools (account only)
    kind: Literal["plan", "adjust", "question"]
    handle: str = Field(min_length=3, max_length=24, pattern=r"^[A-Za-z0-9_.-]+$")
    request: str = Field(min_length=1, max_length=500)
    difficulty: Difficulty | None = None      # None = read it from the request text
    hours: int | None = Field(None, ge=1, le=40)
    avoid_tags: list[str] = Field(default_factory=list, max_length=20)
    plan_id: str | None = None

    @model_validator(mode="after")
    def adjust_needs_a_plan(self):
        if self.kind == "adjust" and self.plan_id is None:
            raise ValueError("an adjust run needs plan_id")
        if self.kind != "adjust" and self.plan_id is not None:
            raise ValueError("only an adjust run takes plan_id")
        return self


class RunCreatedOut(Out):
    run_id: str
    events_url: str                # GET it as text/event-stream


# Server-sent events. Each is one frame: "event: <type>" + "data: <json>" (see sse()).

class StepEvent(Out):
    type: Literal["step"] = "step"
    step: int                      # 1, 2, ... one per tool call
    tool: str
    summary: str                   # e.g. "find_problems dp: 10 problems"
    ok: bool                       # False = the tool returned an error to the model


class PlanEvent(Out):
    type: Literal["plan"] = "plan"
    plan: PlanOut


class AnswerEvent(Out):
    type: Literal["answer"] = "answer"
    text: str                      # the reply to a question run


class DoneEvent(Out):
    type: Literal["done"] = "done"
    outcome: Literal["saved", "answered", "no_plan", "max_steps", "error"]
    steps: int
    input_tokens: int
    output_tokens: int
    latency_ms: int


class ErrorEvent(Out):
    type: Literal["error"] = "error"
    error: ErrorBody


RunEvent = Annotated[StepEvent | PlanEvent | AnswerEvent | DoneEvent | ErrorEvent,
                     Field(discriminator="type")]
RUN_EVENT = TypeAdapter(RunEvent)


def sse(event: StepEvent | PlanEvent | AnswerEvent | DoneEvent | ErrorEvent) -> str:
    """One server-sent-event frame. The browser's EventSource splits frames on a blank line."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


class RunOut(Out):
    """A finished run, for the "how this plan was made" trace."""
    id: str
    kind: Literal["plan", "adjust", "question"]
    created_at: datetime
    steps: list[StepEvent]
    done: DoneEvent


class BudgetOut(Out):
    day: date
    used_tokens: int               # by everyone today
    cap_tokens: int
    guest_plans_left: int | None   # None for accounts


# ---------- The endpoints ----------

@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    auth: Auth
    body: type[In] | None
    response: type[Out] | None     # None = 204 No Content (or a redirect)
    status: int
    slice: str
    note: str = ""


ENDPOINTS = [
    # auth: sign-in itself happens in the browser with Firebase; the session is an
    # httpOnly cookie, never in a body or in localStorage
    Endpoint("POST", "/api/auth/session", "public", SessionIn, MeOut, 200, "4a",
             "verifies the Firebase ID token, sets the session cookie; claims the guest plan"),
    Endpoint("POST", "/api/auth/logout", "account", None, None, 204, "4a",
             "clears the cookie, revokes the Firebase session"),
    Endpoint("GET", "/api/me", "account", None, MeOut, 200, "4a"),
    Endpoint("PATCH", "/api/me", "account", MeUpdateIn, MeOut, 200, "4a",
             "sets the CF handle"),
    # data
    Endpoint("POST", "/api/users/{handle}/sync", "public", None, SyncStatusOut, 202, "3/8",
             "queues a sync; one per handle at a time"),
    Endpoint("GET", "/api/users/{handle}/sync", "public", None, SyncStatusOut, 200, "3/8"),
    Endpoint("GET", "/api/users/{handle}/profile", "public", None, ProfileOut, 200, "3"),
    Endpoint("GET", "/api/users/{handle}/progress", "public", None, ProgressOut, 200, "3",
             "?since=YYYY-MM-DD"),
    # plans
    Endpoint("GET", "/api/plans/current", "guest_ok", None, PlanOut, 200, "5",
             "account: its active plan; guest: the plan in Redis"),
    Endpoint("POST", "/api/plans/current/items/{item_id}", "guest_ok", ItemActionIn, PlanOut,
             200, "7", "swap / easier / harder, 0 tokens"),
    Endpoint("GET", "/api/plans", "account", None, PlansOut, 200, "5", "saved plans"),
    # agent runs
    Endpoint("POST", "/api/runs", "guest_ok", RunIn, RunCreatedOut, 202, "6",
             "guests: kind=plan only, 1 per IP per day"),
    Endpoint("GET", "/api/runs/{run_id}/events", "guest_ok", None, None, 200, "6",
             "text/event-stream of RunEvent frames"),
    Endpoint("GET", "/api/runs/{run_id}", "guest_ok", None, RunOut, 200, "9", "the trace"),
    Endpoint("GET", "/api/budget", "public", None, BudgetOut, 200, "9"),
]


# ---------- Manual check ----------

if __name__ == "__main__":
    for e in ENDPOINTS:
        body = e.body.__name__ if e.body else "-"
        resp = e.response.__name__ if e.response else "-"
        print(f"{e.method:6} {e.path:38} {e.auth:9} {body:16} -> {e.status} {resp:14} "
              f"[{e.slice}] {e.note}")
    print(f"\n{len(ENDPOINTS)} endpoints\n\nexample event stream:")
    done = DoneEvent(outcome="saved", steps=2, input_tokens=6100, output_tokens=800,
                     latency_ms=9400)
    for ev in [StepEvent(step=1, tool="find_problems", summary="dp: 10 problems", ok=True),
               StepEvent(step=2, tool="validate_plan", summary="ok", ok=True), done]:
        print(sse(ev), end="")
