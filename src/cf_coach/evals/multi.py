"""Run the any-user eval cases on many handles: ranks from newbie to LGM, small to huge
histories, and edge accounts (new, inactive, missing).

    uv run python -m cf_coach.evals.multi --label handles
    uv run python -m cf_coach.evals.multi --handles newbie lgm --cases default_week rusty
    uv run python -m cf_coach.evals.multi --max-tokens 80000

- Handles come from evals/handles.json: [{"label": ..., "handle": ..., "why": ...}].
- Each handle is synced from CF and its tag profile refreshed first (CF calls only, no
  LLM tokens), so the cases see that user's real data. A handle that fails to sync is
  reported with the error and skipped.
- Runs cases.ANY_USER_CASES (checks that hold for every user) + checks.UNIVERSAL.
- --max-tokens stops before the next case once the run has used that many tokens, so a
  long run can't burn Groq's daily cap (200k on the free tier). Unrun cases are listed.
- Writes evals/results/<date>-<label>.json: per handle, a data snapshot + the case results.
Design reasons: docs/decisions.md #47.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cf_coach.agent.llm import GroqLLM
from cf_coach.agent.loop import MAX_STEPS
from cf_coach.agent.tools import MIN_ATTEMPTS
from cf_coach.analytics.selection import IGNORED_TAGS
from cf_coach.analytics.tag_profile import refresh_tag_skills
from cf_coach.db.models import Submission, TagSkill, User
from cf_coach.db.session import get_engine
from cf_coach.evals.cases import ANY_USER_CASES
from cf_coach.evals.run import RESULTS_DIR, git_commit, load_data, print_case, run_case, summary
from cf_coach.sync.sync import sync_user

HANDLES_FILE = Path(__file__).resolve().parents[3] / "evals" / "handles.json"


def snapshot(session: Session, handle: str, data) -> dict:
    """What this user's data looks like: explains why a case went the way it did."""
    tags = session.scalars(select(TagSkill).where(TagSkill.handle == handle)).all()
    shown = [t for t in tags if t.tag not in IGNORED_TAGS]
    return {
        "rating": session.scalar(select(User.rating).where(User.handle == handle)),
        "submissions": session.scalar(select(func.count()).select_from(Submission)
                                      .where(Submission.handle == handle)),
        "solved": len(data.solved),
        "tags": len(shown),
        "measured_tags": sum(t.attempts >= MIN_ATTEMPTS for t in shown),   # week rules need these
        "weakest_tag": data.weakest_tag,
    }


def prepare(entry: dict, now: datetime) -> tuple[str | None, dict]:
    """Sync + refresh one handle. Returns (CF's spelling of the handle, info);
    the handle is None if the sync failed."""
    started = time.monotonic()
    try:
        s = sync_user(entry["handle"])
        tags = refresh_tag_skills(s["handle"], now)
    except Exception as e:                       # e.g. CFError: handle not found
        return None, {"error": f"{type(e).__name__}: {e}"}
    return s["handle"], {"fetched": s["fetched"], "tag_rows": tags,
                         "sync_s": round(time.monotonic() - started, 1)}


def tokens(r: dict) -> int:
    return r["input_tokens"] + r["output_tokens"]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="cf_coach.evals.multi")
    parser.add_argument("--handles", nargs="+", metavar="LABEL", help="only these labels")
    parser.add_argument("--cases", nargs="+", metavar="CASE", help="only these cases")
    parser.add_argument("--label", default="handles", help="goes in the report's file name")
    parser.add_argument("--pause", type=float, default=60, help="seconds between cases")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sync-only", action="store_true",
                        help="sync and print each handle's snapshot, run no cases (0 tokens)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    entries = json.loads(HANDLES_FILE.read_text(encoding="utf-8"))
    if args.handles:
        unknown = set(args.handles) - {e["label"] for e in entries}
        if unknown:
            parser.error(f"unknown label(s) {sorted(unknown)}; labels: {[e['label'] for e in entries]}")
        entries = [e for e in entries if e["label"] in args.handles]
    cases = ANY_USER_CASES
    if args.cases:
        unknown = set(args.cases) - {c.name for c in cases}
        if unknown:
            parser.error(f"unknown case(s) {sorted(unknown)}; cases: {[c.name for c in cases]}")
        cases = [c for c in cases if c.name in args.cases]

    now = datetime.now(timezone.utc)
    llm = GroqLLM(model=args.model)
    print(f"{len(entries)} handles x {0 if args.sync_only else len(cases)} cases, "
          f"model {llm.model}\n")
    used, ran_any, sections = 0, False, []
    for entry in entries:
        handle, info = prepare(entry, now)
        section = {**entry, **info, "snapshot": None, "results": [], "not_run": []}
        sections.append(section)
        if handle is None:
            print(f"== {entry['label']} ({entry['handle']}): sync failed: {info['error']}\n")
            continue
        with Session(get_engine()) as session:
            data = load_data(session, handle, now)
            section["snapshot"] = snap = snapshot(session, handle, data)
            print(f"== {entry['label']} ({handle}): rating {snap['rating']}, "
                  f"{snap['solved']} solved, {snap['measured_tags']}/{snap['tags']} tags "
                  f"measured, synced {info['fetched']} new in {info['sync_s']}s")
            for case in [] if args.sync_only else cases:
                if args.max_tokens is not None and used >= args.max_tokens:
                    section["not_run"].append(case.name)
                    continue
                if ran_any:
                    time.sleep(args.pause)
                ran_any = True
                r = run_case(case, llm, session, handle, now, data, args.max_steps)
                section["results"].append(r)
                used += tokens(r)
                print_case(r)
                if case.name == "default_week" and not r["week_rules"]["levels"]:
                    print("    note: week rules off (quota "
                          f"{r['week_rules']['quota'] or 'none'}, no levels)")
        if section["not_run"]:
            print(f"    not run (token cap): {section['not_run']}")
        if section["results"]:
            section["summary"] = summary(section["results"])
        print()

    if args.sync_only:
        return 0 if not any(s.get("error") for s in sections) else 1
    results = [r for s in sections for r in s["results"]]
    total = summary(results) if results else {}
    report = {"label": args.label, "started": now.isoformat(), "model": llm.model,
              "commit": git_commit(), "tokens": used, "summary": total, "handles": sections}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{now.astimezone():%Y-%m-%d-%H%M}-{args.label}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    failed_sync = [s["label"] for s in sections if s.get("error")]
    if total:
        print(f"{total['cases_all_passed']}/{total['cases']} cases fully passed, "
              f"{total['checks_passed']}/{total['checks']} checks, {used:,} tokens")
    if failed_sync:
        print(f"sync failed: {failed_sync}")
    print(f"report: {path}")
    ok = bool(total) and total["checks_passed"] == total["checks"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
