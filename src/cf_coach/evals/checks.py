"""Eval checks: pure functions over one finished run (a RunView). No LLM, no database.

A check returns None when it passes, or a short message saying what was wrong.

Plan checks (count, band, tags...) pass when no plan was saved: a case that needs a plan
also lists saved(). That way "never outside the band" can be checked on a run that
correctly refused to plan.

The eval has its own skill rounding (round_skill) on purpose: if it imported the app's
version, a bug there would pass its own test.
Design reasons: docs/decisions.md #34.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

REJECTED_PREFIX = "Your last tool call was rejected"


def round_skill(skill: float) -> int:
    """To the nearest 100, halves up, never below 800 (CF's easiest problems):
    1415 -> 1400, 1450 -> 1500, 407 -> 800."""
    return max(int((skill + 50) // 100 * 100), 800)


@dataclass(frozen=True)
class PlannedProblem:
    """One saved plan item plus the facts the checks need, looked up by the runner."""
    contest_id: int
    idx: str
    rating: int | None
    tags: tuple[str, ...]          # the problem's own tags
    tag: str                       # the tag the plan says it practises
    reason: str
    skill: float                   # the user's skill on `tag` when the run started
    solved: bool
    running: bool                  # from a contest that is still running
    facts: str = ""                # the code-written numbers line (#40), for reading the report

    @property
    def id(self) -> str:
        return f"{self.contest_id}{self.idx}"

    @property
    def gap(self) -> int | None:
        """Rating above the rounded skill: 1600 for skill 1415 -> +200."""
        return None if self.rating is None else self.rating - round_skill(self.skill)


@dataclass
class RunView:
    """Everything the checks look at for one run."""
    outcome: str                                   # saved / no_plan / max_steps / error
    final_text: str | None
    plan: list[PlannedProblem] | None              # None = no plan saved
    saved_hours: int | None = None                 # constraints.hours as stored with the plan
    tools_called: list[str] = field(default_factory=list)
    rejected_calls: int = 0
    weakest_tag: str | None = None                 # from tag_skills, for question cases


def trace(messages: list[dict]) -> tuple[list[str], int]:
    """Tool names in call order, and how many calls Groq rejected."""
    tools, rejected = [], 0
    for m in messages:
        if m["role"] == "assistant":
            tools += [c["function"]["name"] for c in m.get("tool_calls") or []]
        elif m["role"] == "user" and (m.get("content") or "").startswith(REJECTED_PREFIX):
            rejected += 1
    return tools, rejected


def step_log(messages: list[dict]) -> list[dict]:
    """The run after the system prompt, one entry per call / result / text, for the report.

    {"call": name, "args": "..."} · {"result": "..."} · {"text": "..."} (the model's words)
    · {"user": "..."} (the request, a rejection note or a nudge). Lets a report show *why*
    a run took many steps, e.g. what a failed validate_plan said.
    """
    log = []
    for m in messages[1:]:
        if m["role"] == "assistant":
            if m.get("content"):
                log.append({"text": m["content"]})
            log += [{"call": c["function"]["name"], "args": c["function"]["arguments"]}
                    for c in m.get("tool_calls") or []]
        elif m["role"] == "tool":
            log.append({"result": m["content"]})
        elif m["role"] == "user":
            log.append({"user": m["content"]})
    return log


@dataclass(frozen=True)
class Check:
    name: str
    fn: Callable[[RunView], str | None]

    def __call__(self, view: RunView) -> str | None:
        return self.fn(view)


def _on_plan(name: str, fn: Callable[[list[PlannedProblem]], str | None]) -> Check:
    """A check on the plan's items; passes when there is no plan (see the module doc)."""
    return Check(name, lambda v: None if v.plan is None else fn(v.plan))


# ---------- outcome ----------

def saved() -> Check:
    return Check("saved", lambda v: None if v.outcome == "saved" and v.plan is not None
                 else f"outcome {v.outcome}")


def no_plan() -> Check:
    return Check("no plan saved", lambda v: None if v.plan is None
                 else f"saved a plan of {len(v.plan)}")


# ---------- what is in the plan ----------

def count(n: int) -> Check:
    return _on_plan(f"{n} problems", lambda p: None if len(p) == n else f"got {len(p)}")


def at_most(n: int) -> Check:
    return _on_plan(f"at most {n} problems", lambda p: None if len(p) <= n else f"got {len(p)}")


def tags_in(*allowed: str) -> Check:
    def fn(plan):
        bad = [f"{x.id} [{x.tag}]" for x in plan if x.tag not in allowed]
        return f"other tags: {', '.join(bad)}" if bad else None
    return _on_plan(f"tags in {{{', '.join(allowed)}}}", fn)


def avoids(*tags: str) -> Check:
    def fn(plan):
        bad = [f"{x.id} {sorted(set(x.tags) & set(tags))}" for x in plan if set(x.tags) & set(tags)]
        return f"has avoided tags: {', '.join(bad)}" if bad else None
    return _on_plan(f"avoids {', '.join(tags)}", fn)


def band(lo: int, hi: int) -> Check:
    def fn(plan):
        bad = [f"{x.id} {x.gap:+d}" if x.gap is not None else f"{x.id} unrated"
               for x in plan if x.gap is None or not lo <= x.gap <= hi]
        return f"outside: {', '.join(bad)}" if bad else None
    return _on_plan(f"band {lo:+d}..{hi:+d}", fn)


def ladder(*gaps: int) -> Check:
    """The gaps, in plan order: ladder(0, 100, 200) = your level, +100, +200, easy first."""
    want = list(gaps)
    return _on_plan("ladder " + "/".join(f"{g:+d}" for g in gaps),
                    lambda p: None if [x.gap for x in p] == want
                    else f"gaps {[x.gap for x in p]}, want {want}")


def saved_hours(h: int) -> Check:
    return Check(f"saved hours = {h}", lambda v: None if v.plan is None or v.saved_hours == h
                 else f"saved hours = {v.saved_hours}")


# ---------- guardrails and quality (added to every case by the runner) ----------

def unsolved() -> Check:
    return _on_plan("unsolved", lambda p: (f"already solved: {', '.join(x.id for x in p if x.solved)}"
                                           if any(x.solved for x in p) else None))


def not_running() -> Check:
    return _on_plan("no running contest", lambda p: (
        f"running: {', '.join(x.id for x in p if x.running)}" if any(x.running for x in p) else None))


def distinct_reasons() -> Check:
    def fn(plan):
        reasons = [x.reason.strip().lower() for x in plan]
        return None if len(set(reasons)) == len(reasons) else "some reasons are the same"
    return _on_plan("distinct reasons", fn)


def no_numbers_in_reasons() -> Check:
    """The prompt says reasons have no numbers: code shows them in the facts line (#40).
    Digits inside the problem's own tag names ("2-sat") don't count."""
    def fn(plan):
        bad = []
        for x in plan:
            text = x.reason.lower()
            for t in x.tags:
                text = text.replace(t, "")
            if any(ch.isdigit() for ch in text):
                bad.append(f"{x.id} \"{x.reason}\"")
        return f"numbers in: {'; '.join(bad)}" if bad else None
    return _on_plan("no numbers in reasons", fn)


def no_table() -> Check:
    def fn(v):
        lines = (v.final_text or "").splitlines()
        return "summary has a table" if any(line.lstrip().startswith("|") for line in lines) else None
    return Check("summary has no table", fn)


def no_rejected_calls() -> Check:
    return Check("no rejected calls", lambda v: None if v.rejected_calls == 0
                 else f"{v.rejected_calls} rejected")


def starts_with_profile() -> Check:
    return Check("starts with get_tag_profile", lambda v: (
        None if v.tools_called[:1] == ["get_tag_profile"]
        else f"first call: {v.tools_called[0] if v.tools_called else 'none'}"))


def mentions_weakest_tag() -> Check:
    return Check("names the weakest tag", lambda v: (
        None if v.weakest_tag and v.weakest_tag in (v.final_text or "").lower()
        else f"'{v.weakest_tag}' not in the answer"))


UNIVERSAL = [starts_with_profile(), no_rejected_calls(), unsolved(), not_running(),
             distinct_reasons(), no_numbers_in_reasons(), no_table()]
