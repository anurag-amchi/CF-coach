"""Client for the public Codeforces API.

Every request goes through call_cf(), which handles three things:
  1. Rate limiting  - at most one request every MIN_INTERVAL seconds.
  2. Retries        - temporary failures are retried with exponential backoff.
  3. Error checking - Codeforces wraps every reply in {"status": ..., "result": ...}.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://codeforces.com/api/"
MIN_INTERVAL = 2.0   # seconds between requests (Codeforces allows about 1 per 2 s)
MAX_RETRIES = 4      # retries after the first attempt, so up to 5 requests in total
BACKOFF_BASE = 2.0   # waits 2, 4, 8, 16 seconds between retries
TIMEOUT = 30.0       # seconds; problemset.problems is a large reply


class CFError(Exception):
    """Codeforces refused the request, or kept failing after all retries."""


# ---------- Rate limiting ----------
# In-memory version: only works inside one Python process.
# Gets replaced by a Redis token bucket so several workers can share the limit.

_last_call_time = 0.0


def _wait_for_slot():
    global _last_call_time
    elapsed = time.monotonic() - _last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_call_time = time.monotonic()


# ---------- The one function every request goes through ----------

def call_cf(method, params=None):
    """Call a Codeforces API method and return its "result" field."""
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            wait = BACKOFF_BASE * 2 ** (attempt - 1)
            logger.warning("%s failed (%s); retry %d in %.0fs", method, last_error, attempt, wait)
            time.sleep(wait)

        _wait_for_slot()

        try:
            resp = httpx.get(BASE_URL + method, params=params, timeout=TIMEOUT)
        except httpx.TransportError as e:
            # Timeout, connection reset, DNS failure... worth retrying.
            last_error = f"network error: {e!r}"
            continue

        # Codeforces sends JSON even for most errors, e.g. a bad handle gives
        # HTTP 400 with {"status": "FAILED", "comment": "handle: User ... not found"}.
        try:
            data = resp.json()
        except ValueError:
            data = None

        if data is not None and data.get("status") == "OK":
            return data["result"]

        comment = data.get("comment", "") if data else ""

        # Temporary problems: server errors or "Call limit exceeded" -> retry.
        if resp.status_code >= 500 or resp.status_code == 429 or "limit exceeded" in comment.lower():
            last_error = f"HTTP {resp.status_code} {comment}".strip()
            continue

        # Permanent problems (bad handle, bad parameters): retrying won't help.
        raise CFError(f"{method}: HTTP {resp.status_code} {comment}".strip())

    raise CFError(f"{method}: gave up after {MAX_RETRIES} retries ({last_error})")


# ---------- One thin wrapper per endpoint the project uses ----------

def get_user_info(handles):
    """Profile info (rating, rank...) for a list of handles."""
    return call_cf("user.info", {"handles": ";".join(handles)})


def get_user_submissions(handle, start=1, count=None):
    """Submissions, newest first. start is 1-based; count=None means all of them."""
    params = {"handle": handle, "from": start}
    if count is not None:
        params["count"] = count
    return call_cf("user.status", params)


def get_rating_history(handle):
    """One entry per rated contest the user took part in."""
    return call_cf("user.rating", {"handle": handle})


def get_problemset():
    """All problems with tags and ratings: {"problems": [...], "problemStatistics": [...]}."""
    return call_cf("problemset.problems")


def get_contests():
    """All contests with their phase (BEFORE, CODING, FINISHED...). Used by the running-contest guardrail."""
    return call_cf("contest.list", {"gym": "false"})


# ---------- Manual check: uv run python -m cf_coach.sync.cf_client ----------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    HANDLE = "TitanCoder1258"

    info = get_user_info([HANDLE])[0]
    print("user.info:", info["handle"], info.get("rating"), info.get("rank"))

    subs = get_user_submissions(HANDLE, count=3)
    print("user.status:", [(s["id"], s["problem"]["contestId"], s["problem"]["index"], s.get("verdict")) for s in subs])

    history = get_rating_history(HANDLE)
    print("user.rating:", len(history), "rated contests, latest:", history[-1]["newRating"] if history else None)

    problemset = get_problemset()
    print("problemset.problems:", len(problemset["problems"]), "problems")

    contests = get_contests()
    running = [c["id"] for c in contests if c["phase"] == "CODING"]
    print("contest.list:", len(contests), "contests, running now:", running)

    try:
        get_user_info(["this_handle_does_not_exist_12345"])
    except CFError as e:
        print("bad handle ->", e)
