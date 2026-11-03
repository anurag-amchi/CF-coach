"""The FastAPI app, without Postgres: the pure profile builder, the error shape, and
that every built route matches its row in the contract's ENDPOINTS table."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from cf_coach.analytics.attempts import SubmissionRow
from cf_coach.analytics.selection import round_skill
from cf_coach.api import sync_job
from cf_coach.api.app import app
from cf_coach.api.common import get_redis, get_session
from cf_coach.api.contract import ENDPOINTS
from cf_coach.api.sync_job import lock_key, run_sync, start_sync
from cf_coach.api.users import profile_out, progress_out
from cf_coach.sync.cf_client import CFError

NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)


def sub(verdict, contest, idx, rating, tags, days_ago=30, ptype="PRACTICE", rel_min=None):
    return SubmissionRow(contest, idx, verdict, NOW - timedelta(days=days_ago), rating, tags,
                         ptype, None if rel_min is None else rel_min * 60)


ROWS = [
    # 2000A dp: one WA, then AC in practice
    sub("WRONG_ANSWER", 2000, "A", 1500, ("dp",), days_ago=30),
    sub("OK", 2000, "A", 1500, ("dp",), days_ago=29),
    # 2000B dp+greedy: a WA in the contest, never solved -> given up + backlog
    sub("WRONG_ANSWER", 2000, "B", 1600, ("dp", "greedy"), ptype="CONTESTANT", rel_min=50),
    # 2001A greedy: AC in the contest after 20 minutes
    sub("OK", 2001, "A", 1200, ("greedy",), ptype="CONTESTANT", rel_min=20),
    # 2002A from a special round: given up, but never in the backlog (#57)
    sub("WRONG_ANSWER", 2002, "A", 1000, ("*special", "implementation"),
        ptype="CONTESTANT", rel_min=10),
]


# ---------- profile_out (pure) ----------

def test_profile_numbers():
    p = profile_out("someone", None, NOW, ROWS, NOW)
    assert p.rating is None                        # unrated stays None in the response
    assert p.solved_total == 2                     # 2000A, 2001A
    assert p.backlog_total == 1                    # 2000B; the special 2002A is left out
    assert p.median_solve_minutes == 20.0          # only 2001A was solved in a contest

    by_tag = {t.tag: t for t in p.tags}
    assert set(by_tag) == {"dp", "greedy", "implementation"}   # no "*special"
    dp = by_tag["dp"]
    assert (dp.attempts, dp.solved, dp.abandon_rate, dp.avg_wrong_before_ac, dp.backlog) \
        == (2, 1, 0.5, 1.0, 1)
    impl = by_tag["implementation"]
    assert (impl.solved, impl.abandon_rate, impl.avg_wrong_before_ac, impl.backlog) \
        == (0, 1.0, None, 0)


def test_tags_are_weakest_first_with_their_level():
    p = profile_out("someone", 1400, NOW, ROWS, NOW)
    skills = [t.skill for t in p.tags]
    assert skills == sorted(skills)
    assert all(t.level == round_skill(t.skill) for t in p.tags)


def test_no_submissions_gives_an_empty_profile():
    p = profile_out("newbie", None, NOW, [], NOW)
    assert (p.solved_total, p.backlog_total, p.median_solve_minutes, p.tags) == (0, 0, None, [])


# ---------- the error shape, through the real app ----------

class NoUsers:
    """Stands in for the DB session: no handle has been synced (find_user finds nothing)."""
    def scalar(self, stmt):
        return None


class BrokenDb:
    def scalar(self, stmt):
        raise RuntimeError("connection refused")


class FakeRedis:
    """The 4 Redis calls the sync code makes, in a dict. Expiry (ex=) is ignored."""
    def __init__(self):
        self.data = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None                    # what real Redis returns when NX refuses
        self.data[key] = value
        return True

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self.data.pop(key, None)


@pytest.fixture
def client():
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_unsynced_handle_is_404_not_found(client):
    app.dependency_overrides[get_session] = lambda: NoUsers()
    r = client.get("/api/users/nobody_here/profile")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "not_found",
                                  "message": "nobody_here has not been synced yet",
                                  "retry_after_s": None}}


def test_bad_handle_is_422_validation(client):
    r = client.get("/api/users/no!/profile")       # too short and has "!"
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation"
    assert r.json()["error"]["message"].startswith("handle:")


def test_unknown_path_and_wrong_method_are_not_found(client):
    for r in (client.get("/api/nothing"), client.post("/api/users/TitanCoder1258/profile")):
        assert r.status_code in (404, 405)
        assert r.json()["error"]["code"] == "not_found"


def test_a_crash_is_500_internal_without_details(client):
    app.dependency_overrides[get_session] = lambda: BrokenDb()
    r = client.get("/api/users/TitanCoder1258/profile")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal"
    assert "connection refused" not in r.text      # internals stay in the server log


# ---------- progress ----------

def test_progress_counts_only_first_solves_since_the_date():
    rows = ROWS + [
        sub("OK", 2000, "A", 1500, ("dp",), days_ago=2),       # re-solve: not new
        sub("OK", 2003, "C", 1700, ("dp",), days_ago=3),       # new
    ]
    p = progress_out(rows, (NOW - timedelta(days=7)).date(), NOW, 1400)
    assert p.solved_total == 1
    assert [(s.id, s.solved_on) for s in p.solved] == [("2003C", (NOW - timedelta(days=3)).date())]
    assert [c.tag for c in p.skill_changes] == ["dp"]           # only dp moved
    assert p.skill_changes[0].after > p.skill_changes[0].before


def test_progress_since_in_the_future_is_422(client):
    app.dependency_overrides[get_session] = lambda: NoUsers()
    r = client.get("/api/users/TitanCoder1258/progress?since=2999-01-01")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation"
    r = client.get("/api/users/TitanCoder1258/progress?since=2026-13-01")   # not a date
    assert r.status_code == 422


def test_progress_of_unsynced_handle_is_404(client):
    app.dependency_overrides[get_session] = lambda: NoUsers()
    assert client.get("/api/users/nobody_here/progress").status_code == 404


# ---------- sync ----------

def test_only_one_sync_per_handle_at_a_time():
    r = FakeRedis()
    assert start_sync(r, "TitanCoder1258") is True
    assert start_sync(r, "titancoder1258") is False     # same handle, other spelling
    assert start_sync(r, "tourist") is True             # other handles aren't blocked


def test_post_sync_runs_the_job_once_and_reports_done(client, monkeypatch):
    r = FakeRedis()
    ran = []
    monkeypatch.setattr(sync_job, "sync_steps", lambda h: ran.append(h) or 12)
    app.dependency_overrides[get_session] = lambda: NoUsers()
    app.dependency_overrides[get_redis] = lambda: r

    resp = client.post("/api/users/TitanCoder1258/sync")
    assert resp.status_code == 202
    assert resp.json()["state"] == "queued"     # the reply is sent before the job runs
    assert ran == ["TitanCoder1258"]             # TestClient runs background tasks after it
    status = client.get("/api/users/TitanCoder1258/sync").json()
    assert (status["state"], status["new_submissions"], status["error"]) == ("done", 12, None)
    assert r.get(lock_key("TitanCoder1258")) is None    # lock freed: a new sync may start


def test_a_never_synced_handle_reports_never(client):
    app.dependency_overrides[get_session] = lambda: NoUsers()
    app.dependency_overrides[get_redis] = lambda: FakeRedis()
    assert client.get("/api/users/nobody_here/sync").json()["state"] == "never"


def raises(e):
    def steps(handle):
        raise e
    return steps


def test_unknown_cf_handle_is_an_error_state_in_plain_words(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(sync_job, "sync_steps", raises(
        CFError("user.info: HTTP 400 handles: User with handle zzq not found")))
    start_sync(r, "zzq")
    run_sync(r, "zzq")
    saved = json.loads(r.get(sync_job.status_key("zzq")))
    assert saved == {"state": "error", "new_submissions": None,
                     "error": "Codeforces has no user called zzq"}
    assert r.get(lock_key("zzq")) is None


def test_a_crash_in_the_job_still_frees_the_lock(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(sync_job, "sync_steps", raises(RuntimeError("db down")))
    start_sync(r, "TitanCoder1258")
    run_sync(r, "TitanCoder1258")
    saved = json.loads(r.get(sync_job.status_key("TitanCoder1258")))
    assert saved["state"] == "error" and "db down" not in saved["error"]
    assert start_sync(r, "TitanCoder1258") is True


# ---------- routes vs the contract ----------

def api_routes():
    return [r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/")]


def test_every_built_route_matches_its_contract_row():
    table = {(e.method, e.path): e for e in ENDPOINTS}
    for route in api_routes():
        for method in route.methods:
            e = table.get((method, route.path))
            assert e is not None, f"{method} {route.path} is not in ENDPOINTS"
            assert (route.status_code or 200) == e.status, route.path
            assert route.response_model is e.response, route.path


def test_openapi_shows_the_error_shape():
    schema = app.openapi()
    assert "/api/users/{handle}/profile" in schema["paths"]
    assert "ErrorOut" in schema["components"]["schemas"]
