"""Run the eval cases against the real model and report which checks pass.

    uv run python -m cf_coach.evals.run TitanCoder1258 --label baseline
    uv run python -m cf_coach.evals.run TitanCoder1258 --only dp_upskill rusty --pause 0

- Uses the database as it is (no CF sync, no tag refresh), so every case sees the same data.
- Plans go to a MemoryPlanStore: the evals never replace your real active plan, and
  eval runs are not written to agent_runs (the JSON report holds them).
- Cases start --pause seconds apart (default 60): Groq's free tier allows 8,000 tokens a
  minute and one run uses ~8k, so without the pause the 429 waits distort the timings.
- Writes evals/results/<date>-<label>.json. Exit code 1 if any check failed.
Design reasons: docs/decisions.md #34.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from cf_coach.agent.llm import LLM, GroqLLM
from cf_coach.agent.loop import MAX_STEPS, run_agent
from cf_coach.agent.tools import MemoryPlanStore, ToolContext, user_constraints
from cf_coach.analytics.selection import (
    IGNORED_TAGS, build_profile, load_problems, load_solved, running_contest_ids,
)
from cf_coach.db.models import TagSkill
from cf_coach.db.session import get_engine
from cf_coach.evals.cases import CASES, Case
from cf_coach.evals.checks import UNIVERSAL, PlannedProblem, RunView, trace

RESULTS_DIR = Path(__file__).resolve().parents[3] / "evals" / "results"


@dataclass
class EvalData:
    """Loaded once per eval run and shared by every case."""
    problems: dict
    solved: set
    skills: dict[str, float]
    rating: int
    running: set[int]
    weakest_tag: str | None


def load_data(session: Session, handle: str, now: datetime) -> EvalData:
    profile = build_profile(session, handle, now)
    rows = session.scalars(select(TagSkill).where(TagSkill.handle == handle)).all()
    shown = [r for r in rows if r.tag not in IGNORED_TAGS]       # what get_tag_profile shows
    weakest = min(shown, key=lambda r: (r.skill, r.tag)).tag if shown else None
    return EvalData(
        problems={(p.contest_id, p.idx): p for p in load_problems(session)},
        solved=load_solved(session, handle),
        skills=profile.skills, rating=profile.rating,
        running=running_contest_ids(), weakest_tag=weakest)


def build_view(result, ctx: ToolContext, data: EvalData) -> RunView:
    tools_called, rejected = trace(result.messages)
    plan, hours = None, None
    if ctx.saved_plan_id is not None:
        constraints, items = ctx.plan_store.plans[ctx.saved_plan_id]
        hours = constraints.hours
        plan = []
        for it in items:
            p = data.problems.get((it.contest_id, it.idx))
            plan.append(PlannedProblem(
                contest_id=it.contest_id, idx=it.idx,
                rating=p.rating if p else None, tags=p.tags if p else (),
                tag=it.tag, reason=it.reason,
                skill=data.skills.get(it.tag, data.rating),
                solved=(it.contest_id, it.idx) in data.solved,
                running=it.contest_id in data.running))
    return RunView(outcome=result.outcome, final_text=result.final_text, plan=plan,
                   saved_hours=hours, tools_called=tools_called, rejected_calls=rejected,
                   weakest_tag=data.weakest_tag)


def run_case(case: Case, llm: LLM, session: Session, handle: str, now: datetime,
             data: EvalData, max_steps: int) -> dict:
    ctx = ToolContext(handle, session, now,
                      running_contests=set(data.running),
                      user_constraints=user_constraints(session, case.hours, case.avoid),
                      plan_store=MemoryPlanStore())
    result = run_agent(llm, ctx, case.request, max_steps=max_steps)
    view = build_view(result, ctx, data)
    checks = [{"check": c.name, "ok": (msg := c(view)) is None, "detail": msg}
              for c in case.checks + UNIVERSAL]
    return {
        "case": case.name, "request": case.request,
        "hours": case.hours, "avoid": case.avoid,
        "outcome": result.outcome, "error": result.error,
        "steps": result.steps, "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens, "latency_ms": result.latency_ms,
        "tools_called": view.tools_called, "rejected_calls": view.rejected_calls,
        "plan": None if view.plan is None else [
            {**asdict(x), "id": x.id, "gap": x.gap} for x in view.plan],
        "final_text": result.final_text,
        "checks": checks,
    }


def git_commit() -> str:
    """The commit the eval ran on, plus '+dirty' if there were uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summary(results: list[dict]) -> dict:
    checks = [c for r in results for c in r["checks"]]
    return {
        "cases": len(results),
        "cases_all_passed": sum(all(c["ok"] for c in r["checks"]) for r in results),
        "checks": len(checks),
        "checks_passed": sum(c["ok"] for c in checks),
        "rejected_calls": sum(r["rejected_calls"] for r in results),
        "avg_steps": round(sum(r["steps"] for r in results) / len(results), 1),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in results) / len(results)),
        "avg_latency_s": round(sum(r["latency_ms"] for r in results) / len(results) / 1000, 1),
    }


def print_case(r: dict) -> None:
    ok = sum(c["ok"] for c in r["checks"])
    print(f"{r['case']:<22} {r['outcome']:<9} {ok:>2}/{len(r['checks']):<2} checks  "
          f"{r['steps']} steps  {r['input_tokens']:>6,} tokens  {r['latency_ms'] / 1000:5.1f}s")
    for c in r["checks"]:
        if not c["ok"]:
            print(f"    FAIL {c['check']}: {c['detail']}")
    if r["error"]:
        print(f"    ERROR {r['error']}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):        # model text may not fit cp1252 (#32)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="cf_coach.evals.run")
    parser.add_argument("handle")
    parser.add_argument("--only", nargs="+", metavar="CASE", help="run just these cases")
    parser.add_argument("--label", default="run", help="goes in the report's file name")
    parser.add_argument("--pause", type=float, default=60, help="seconds between cases")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    cases = CASES
    if args.only:
        unknown = set(args.only) - {c.name for c in CASES}
        if unknown:
            parser.error(f"unknown case(s) {sorted(unknown)}; cases: {[c.name for c in CASES]}")
        cases = [c for c in CASES if c.name in args.only]

    now = datetime.now(timezone.utc)
    llm = GroqLLM(model=args.model)
    print(f"{len(cases)} cases, model {llm.model}, {args.pause:g}s apart\n")
    results = []
    with Session(get_engine()) as session:
        data = load_data(session, args.handle, now)
        for n, case in enumerate(cases):
            if n:
                time.sleep(args.pause)
            results.append(run_case(case, llm, session, args.handle, now, data, args.max_steps))
            print_case(results[-1])

    report = {"label": args.label, "started": now.isoformat(), "model": llm.model,
              "commit": git_commit(), "handle": args.handle,
              "summary": summary(results), "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{now.astimezone():%Y-%m-%d-%H%M}-{args.label}.json"   # local time
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    print(f"\n{s['cases_all_passed']}/{s['cases']} cases fully passed, "
          f"{s['checks_passed']}/{s['checks']} checks, {s['rejected_calls']} rejected calls, "
          f"avg {s['avg_steps']} steps / {s['avg_input_tokens']:,} tokens / {s['avg_latency_s']}s")
    print(f"report: {path}")
    return 0 if s["checks_passed"] == s["checks"] else 1


if __name__ == "__main__":
    sys.exit(main())
