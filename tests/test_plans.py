"""GET /api/plans/current and GET /api/plans (slice 5), with the account and DB faked."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from cf_coach.api import plans
from cf_coach.api.app import app
from cf_coach.api.auth import SignedIn, optional_account
from cf_coach.api.common import get_session
from cf_coach.api.firebase import Identity
from cf_coach.db.models import Account, Plan

WHEN = datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc)


def item(cid, idx, rating, gap, tag="dp"):
    return {"id": f"{cid}{idx}", "contest_id": cid, "idx": idx, "tag": tag, "rating": rating,
            "gap": gap, "facts": f"{tag} skill 1415 (level 1400) · {rating} = level +{gap}",
            "reason": f"why {cid}{idx}"}


NEW = {"hours": 5, "avoid_tags": ["geometry"], "difficulty": "normal"}
# A plan saved on 25 Oct, before items had id / rating / gap / facts
OLD_ITEMS = [{"idx": "D", "tag": "dp", "reason": "...", "contest_id": 2242}]


def row(plan_id, items, constraints=NEW, status="active"):
    return Plan(id=plan_id, handle="TitanCoder1258", created_at=WHEN, constraints=constraints,
                items=items, status=status)


def me(handle="TitanCoder1258"):
    who = Identity(uid="u1", email="u1@example.com", email_verified=True,
                   provider="password", auth_time=0)
    return SignedIn(who, Account(uid="u1", email="u1@example.com", handle=handle))


# ---------- pure ----------

def test_read_plan_rejects_the_old_format():
    assert plans.read_plan({"hours": 3, "avoid_tags": []}, OLD_ITEMS) is None
    constraints, items = plans.read_plan(NEW, [item(2000, "B", 1400, 0)])
    assert constraints.difficulty == "normal" and items[0].gap == 0


def test_old_constraints_without_difficulty_read_as_normal():
    constraints, _ = plans.read_plan({"hours": None, "avoid_tags": []}, [item(1, "A", 800, 0)])
    assert constraints.difficulty == "normal"


def test_plan_out_adds_name_url_and_solved_and_keeps_the_order():
    c, items = plans.read_plan(NEW, [item(2000, "B", 1400, 0), item(921, "01", 1600, 200)])
    out = plans.plan_out(7, "TitanCoder1258", WHEN, c, items,
                         names={(2000, "B"): "Easy One"}, solved={(921, "01")})
    assert out.id == "7" and out.saved is True and out.expires_at is None
    assert [i.id for i in out.items] == ["2000B", "92101"]
    assert out.items[0].name == "Easy One"
    assert out.items[1].name == "92101"                 # no problems row: the id
    assert out.items[1].url == "https://codeforces.com/problemset/problem/921/01"
    assert [i.solved for i in out.items] == [False, True]
    assert out.summary.startswith("Saved 2 problems, easiest first, rated 1400-1600.")
    assert out.hours == 5 and out.avoid_tags == ["geometry"]


# ---------- routes ----------

@pytest.fixture
def client(monkeypatch):
    seen = {}

    def load_plans(session, handle, active_only, limit):
        seen.update(handle=handle, active_only=active_only, limit=limit)
        rows = seen.get("rows", [])
        return [r for r in rows if r.status == "active"][:limit] if active_only else rows[:limit]

    monkeypatch.setattr(plans, "load_plans", load_plans)
    monkeypatch.setattr(plans, "load_names", lambda session, keys: {})
    monkeypatch.setattr(plans, "load_solved", lambda session, handle: set())
    app.dependency_overrides[get_session] = lambda: None
    yield TestClient(app), seen
    app.dependency_overrides.clear()


def signed_in(account):
    app.dependency_overrides[optional_account] = lambda: account


def test_a_guest_has_no_plan_yet(client):
    c, _ = client
    signed_in(None)
    r = c.get("/api/plans/current")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_the_plan_list_needs_an_account(client):
    c, _ = client
    signed_in(None)
    r = c.get("/api/plans")
    assert r.status_code == 401 and r.json()["error"]["code"] == "sign_in_required"


def test_a_bad_cookie_is_401_not_the_guest_view(client):
    c, _ = client
    app.dependency_overrides.pop(optional_account, None)
    from cf_coach.api.firebase import AuthFailed, get_firebase

    class Rejects:
        def verify_session_cookie(self, cookie):
            raise AuthFailed("bad")
    app.dependency_overrides[get_firebase] = lambda: Rejects()
    c.cookies.set("cfc_session", "forged")
    r = c.get("/api/plans/current")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_no_handle_set_says_so(client):
    c, _ = client
    signed_in(me(handle=None))
    r = c.get("/api/plans/current")
    assert r.status_code == 404
    assert r.json()["error"]["message"] == "set your Codeforces handle first"


def test_current_is_the_active_plan(client):
    c, seen = client
    signed_in(me())
    seen["rows"] = [row(9, [item(2000, "B", 1400, 0)])]
    r = c.get("/api/plans/current")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "9" and body["items"][0]["facts"].startswith("dp skill 1415")
    assert seen["active_only"] is True and seen["limit"] == 1


def test_an_old_format_active_plan_counts_as_no_plan(client):
    c, seen = client
    signed_in(me())
    seen["rows"] = [row(8, OLD_ITEMS, constraints={"hours": 3, "avoid_tags": []})]
    assert c.get("/api/plans/current").status_code == 404


def test_the_list_is_newest_first_and_skips_old_plans(client):
    c, seen = client
    signed_in(me())
    seen["rows"] = [row(10, [item(1, "A", 800, 0)]),
                    row(9, [item(2, "A", 900, 100)], status="replaced"),
                    row(8, OLD_ITEMS, status="replaced")]
    r = c.get("/api/plans")
    assert r.status_code == 200
    assert [p["id"] for p in r.json()["plans"]] == ["10", "9"]
    assert seen["limit"] == plans.MAX_PLANS_SHOWN
