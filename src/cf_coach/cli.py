"""Command line: one command from a handle to a saved plan.

    uv run python -m cf_coach plan TitanCoder1258 "4 hours this week, no geometry"

Steps: sync the user's new submissions -> refresh tag_skills -> parse the request
(count, hours, tags, difficulty; flags win) -> run the agent -> log the run to agent_runs
-> print the saved plan (read back from the plans table).

Options:  --hours N   practice hours this week (1-40); always enforced on the plan
          --avoid TAG a tag to keep out of the plan, e.g. --avoid geometry; repeat it for
                      more ("--avoid 'dfs and similar'" for tags with spaces)
          --difficulty easier|normal|harder   the rating band (default normal = level
                      to +200); always enforced
          --no-sync   skip the CF sync (the user must have been synced before)
          --max-steps N, --model NAME
          -v          also show every HTTP request and SDK retry (to see where time goes)
Exit code: 0 if the run ended normally (saved / no_plan), 1 otherwise.

Run as `python -m cf_coach`: Windows Smart App Control blocks the cf-coach.exe launcher.
Design reasons: docs/decisions.md #31, #32, #36, #38, #40.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

OK_OUTCOMES = {"saved", "no_plan"}


def short(text: str, n: int = 110) -> str:
    return text if len(text) <= n else text[:n - 3] + "..."


def printer(started: float):
    """An on_message callback: one line per tool call / result / note, with elapsed time.

    An assistant line's time is when the model's reply arrived, so the gap before it is
    the LLM call; the gap before a result line is the tool.
    """
    def show(msg: dict) -> None:
        t = f"[{time.monotonic() - started:5.1f}s]"
        if msg["role"] == "assistant":
            for c in msg.get("tool_calls", []):
                print(f"{t} -> {c['function']['name']}({short(c['function']['arguments'], 90)})")
        elif msg["role"] == "tool":
            print(f"{t}    <- {short(msg['content'])}")
        elif msg["role"] == "user":
            print(f"{t}    !! {short(msg['content'])}")
    return show


def hours_arg(text: str) -> int:
    """argparse type for --hours: an int in the same range the plan schema allows."""
    n = int(text)                       # "abc" -> ValueError -> argparse prints a usage error
    if not 1 <= n <= 40:
        raise argparse.ArgumentTypeError("hours must be between 1 and 40")
    return n


def exit_code(outcome: str) -> int:
    return 0 if outcome in OK_OUTCOMES else 1


def print_plan(session, plan_id: int) -> None:
    """The plan as saved in Postgres, not as the model described it."""
    from cf_coach.db.models import Plan, Problem
    plan = session.get(Plan, plan_id)
    print(f"\nPlan #{plan.id}  (constraints: {plan.constraints})")
    for n, item in enumerate(plan.items, start=1):
        c, i = item["contest_id"], item["idx"]
        p = session.get(Problem, (c, i))
        if "facts" in item:                                     # plans since #40
            print(f"{n}. {c}{i}  {p.name}")
            print(f"   {item['facts']}")
            print(f"   why: {item['reason']}")
        else:
            gap = f" {item['gap']:+d}" if "gap" in item else ""  # plans before #36 have none
            print(f"{n}. {c}{i}  {p.name}  ({p.rating})  [{item['tag']}{gap}]")
            print(f"   {item['reason']}")
        print(f"   https://codeforces.com/problemset/problem/{c}/{i}")


def cmd_plan(args) -> int:
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from cf_coach.agent.llm import GroqLLM
    from cf_coach.agent.loop import run_agent, save_run
    from cf_coach.agent.request import plan_limits, read_request
    from cf_coach.agent.tools import ToolContext
    from cf_coach.analytics.selection import DIFFICULTY_GAPS
    from cf_coach.analytics.tag_profile import refresh_tag_skills
    from cf_coach.db.models import Problem
    from cf_coach.db.session import get_engine
    from cf_coach.sync.sync import sync_problemset, sync_user

    started = time.monotonic()
    now = datetime.now(timezone.utc)
    handle = args.handle

    if not args.no_sync:
        with Session(get_engine()) as session:
            have_problems = session.scalar(select(func.count()).select_from(Problem)) > 0
        if not have_problems:                       # first run on an empty database
            print(f"problemset: {sync_problemset()} problems")
        summary = sync_user(handle)
        handle = summary["handle"]                  # CF's spelling of the handle
        print(f"synced {handle}: {summary['fetched']} new submissions, {summary['total']} total")
    print(f"tag profile: {refresh_tag_skills(handle, now)} tags")

    with Session(get_engine()) as session:
        llm = GroqLLM(model=args.model)
        # unknown --avoid tag -> ToolError here, before any agent step
        request, parsed = read_request(llm, session, args.request, args.hours, args.avoid,
                                       args.difficulty)
        for warning in parsed.warnings:
            print(f"warning: {warning}")
        limits = plan_limits(request)
        lo, hi = DIFFICULTY_GAPS[limits["difficulty"]]
        print(f"request: {request.intent}, count={limits['count']}, hours={request.hours}, "
              f"focus={request.focus_tags}, avoid={request.avoid_tags}, "
              f"difficulty={limits['difficulty']} ({lo:+d} to {hi:+d} around each tag's level)"
              f"  [{parsed.input_tokens} tokens]")
        print(f"agent: {llm.model}, max {args.max_steps} steps\n")
        ctx = ToolContext(handle, session, now, **limits)
        result = run_agent(llm, ctx, args.request, max_steps=args.max_steps,
                           on_message=printer(started))
        result.input_tokens += parsed.input_tokens      # the run's cost includes the parse
        result.output_tokens += parsed.output_tokens
        run_id = save_run(session, handle, result)
        if result.final_text:
            print(f"\n{result.final_text}")
        if result.plan_id is not None:
            print_plan(session, result.plan_id)

    print(f"\noutcome={result.outcome}  steps={result.steps}  "
          f"tokens={result.input_tokens} in / {result.output_tokens} out  "
          f"agent {result.latency_ms / 1000:.1f}s, total {time.monotonic() - started:.1f}s  "
          f"(agent_runs id {run_id})")
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
    return exit_code(result.outcome)


def build_parser() -> argparse.ArgumentParser:
    from cf_coach.agent.loop import MAX_STEPS
    from cf_coach.analytics.selection import DIFFICULTY_GAPS
    parser = argparse.ArgumentParser(prog="cf_coach", description="Codeforces practice coach")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="sync, then build and save a plan for this week")
    plan.add_argument("handle")
    plan.add_argument("request", help='what you want, e.g. "4 hours, no geometry"')
    plan.add_argument("--hours", type=hours_arg, default=None,
                      help="practice hours this week (1-40), always enforced")
    plan.add_argument("--avoid", action="append", default=[], metavar="TAG",
                      help="a tag to keep out of the plan; repeat for more")
    plan.add_argument("--difficulty", choices=list(DIFFICULTY_GAPS), default=None,
                      help="easier = -100..+100, normal = +0..+200, harder = +100..+300 "
                           "around each tag's level (skill rounded to 100). Default: from "
                           "the request text, else normal")
    plan.add_argument("--no-sync", action="store_true")
    plan.add_argument("--max-steps", type=int, default=MAX_STEPS)
    plan.add_argument("--model", default=None, help="default: LLM_MODEL from .env")
    plan.add_argument("-v", "--verbose", action="store_true")
    plan.set_defaults(func=cmd_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Model text has characters like ' '. Piped to a file on Windows, Python would
    # write cp1252 and crash on them; UTF-8 can write anything.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="        %(name)s: %(message)s")
    if args.verbose:
        for name in ("httpx", "openai"):            # each HTTP request, and SDK retries
            logging.getLogger(name).setLevel(logging.INFO)

    from cf_coach.agent.tools import ToolError
    from cf_coach.sync.cf_client import CFError
    try:
        return args.func(args)
    except (CFError, LookupError, ToolError) as e:  # bad handle / never synced / bad --avoid tag
        print(f"error: {e}", file=sys.stderr)
        return 1
