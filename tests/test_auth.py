"""Sign-in, /me and logout through the real app, with Firebase and the DB faked."""

import time

import pytest
from fastapi.testclient import TestClient

from cf_coach.api.app import app
from cf_coach.api.auth import COOKIE
from cf_coach.api.common import get_session
from cf_coach.api.firebase import AuthFailed, Identity, get_firebase


def who(uid="u1", provider="password", verified=True, signed_in_s_ago=10):
    return Identity(uid=uid, email=f"{uid}@example.com", email_verified=verified,
                    provider=provider, auth_time=int(time.time()) - signed_in_s_ago)


def token(name):
    return name.ljust(120, "x")      # the contract wants a JWT-sized string


class FakeFirebase:
    """Knows a fixed set of ID tokens; a cookie is "cookie-<uid>" until revoked."""
    def __init__(self, tokens):
        self.tokens = tokens                       # id token -> Identity
        self.sessions = {}                         # cookie -> Identity
        self.revoked = set()

    def verify_id_token(self, id_token):
        if id_token not in self.tokens:
            raise AuthFailed("bad token")
        return self.tokens[id_token]

    def create_session_cookie(self, id_token, expires_in):
        cookie = "cookie-" + self.tokens[id_token].uid
        self.sessions[cookie] = self.tokens[id_token]
        return cookie

    def verify_session_cookie(self, cookie):
        found = self.sessions.get(cookie)
        if found is None or found.uid in self.revoked:
            raise AuthFailed("bad cookie")
        return found

    def revoke(self, uid):
        self.revoked.add(uid)


class FakeDb:
    """The session calls the auth routes make: accounts in a dict, no synced CF users."""
    def __init__(self):
        self.accounts = {}

    def get(self, model, key):
        return self.accounts.get(key)

    def add(self, obj):
        self.accounts[obj.uid] = obj

    def commit(self):
        pass

    def scalar(self, stmt):                        # find_user: nobody synced
        return None


TOKENS = {
    token("ok"): who(),
    token("google"): who(uid="g1", provider="google.com"),
    token("unverified"): who(uid="u2", verified=False),
    token("old"): who(uid="u3", signed_in_s_ago=3600),
    token("anon"): who(uid="u4", provider="anonymous"),
}


@pytest.fixture
def env():
    fb, db = FakeFirebase(TOKENS), FakeDb()
    app.dependency_overrides[get_firebase] = lambda: fb
    app.dependency_overrides[get_session] = lambda: db
    # https: a Secure cookie is only stored and sent back over HTTPS
    client = TestClient(app, base_url="https://testserver", raise_server_exceptions=False)
    yield client, fb, db
    app.dependency_overrides.clear()


def sign_in(client, name="ok"):
    return client.post("/api/auth/session", json={"id_token": token(name)})


def test_sign_in_sets_a_locked_down_cookie_and_creates_the_account(env):
    client, fb, db = env
    r = sign_in(client)
    assert r.status_code == 200
    assert r.json() == {"email": "u1@example.com", "email_verified": True,
                        "sign_in_provider": "password", "handle": None}
    set_cookie = r.headers["set-cookie"]
    assert set_cookie.startswith(f"{COOKIE}=cookie-u1;")
    for flag in ("HttpOnly", "Secure", "SameSite=lax", "Path=/", "Max-Age=432000"):
        assert flag in set_cookie                  # 432000 s = 5 days
    assert list(db.accounts) == ["u1"]


def test_signing_in_again_reuses_the_account(env):
    client, fb, db = env
    sign_in(client)
    sign_in(client)
    assert list(db.accounts) == ["u1"]


def test_me_and_linking_a_handle(env):
    client, fb, db = env
    sign_in(client, "google")
    assert client.get("/api/me").json()["sign_in_provider"] == "google.com"
    r = client.patch("/api/me", json={"handle": "TitanCoder1258"})
    assert r.status_code == 200 and r.json()["handle"] == "TitanCoder1258"
    assert client.get("/api/me").json()["handle"] == "TitanCoder1258"
    assert client.patch("/api/me", json={"handle": "no!"}).status_code == 422


@pytest.mark.parametrize("name, status, code", [
    ("unverified", 403, "email_not_verified"),
    ("old", 401, "unauthorized"),             # signed in an hour ago: sign in again
    ("anon", 401, "unauthorized"),            # a method we didn't turn on
    ("forged", 401, "unauthorized"),          # Firebase doesn't know this token
])
def test_a_refused_sign_in_gets_no_cookie_and_no_account(env, name, status, code):
    client, fb, db = env
    r = sign_in(client, name)
    assert (r.status_code, r.json()["error"]["code"]) == (status, code)
    assert "set-cookie" not in r.headers
    assert db.accounts == {}


def test_me_without_a_cookie_is_sign_in_required(env):
    client, fb, db = env
    r = client.get("/api/me")
    assert (r.status_code, r.json()["error"]["code"]) == (401, "sign_in_required")


def test_logout_ends_the_session_even_for_a_copied_cookie(env):
    client, fb, db = env
    sign_in(client)
    stolen = client.cookies[COOKIE]
    r = client.post("/api/auth/logout")
    assert r.status_code == 204
    assert COOKIE not in client.cookies           # the browser drops it
    client.cookies.set(COOKIE, stolen)            # someone kept a copy
    r = client.get("/api/me")
    assert (r.status_code, r.json()["error"]["code"]) == (401, "unauthorized")


def test_logout_without_a_session_still_succeeds(env):
    client, fb, db = env
    assert client.post("/api/auth/logout").status_code == 204
