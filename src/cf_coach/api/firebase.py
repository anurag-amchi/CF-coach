"""The only file that talks to Firebase (decision #53). Four calls:

  verify_id_token        the browser's sign-in proof (a JWT signed by Google, valid 1 h)
  create_session_cookie  swaps that proof for a longer session (we choose: 5 days)
  verify_session_cookie  checks the cookie on every signed-in request
  revoke                 ends every session of a user (logout)

Each failure Firebase can raise for a bad, expired or revoked token, or a disabled user,
becomes one AuthFailed, so the routes (and the fake in the tests) handle a single error.

Needs GOOGLE_APPLICATION_CREDENTIALS in .env: the path to the service-account key file,
kept outside the repo. The key lets this server create session cookies for the project.
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import auth

# Allowed clock difference between this machine and Google's when checking a token's
# issued-at / expiry times. A laptop clock a few seconds off would otherwise fail
# fresh tokens with "used too early".
CLOCK_SKEW_S = 10


class AuthFailed(Exception):
    """The token or cookie is bad, expired or revoked, or its user is disabled."""


@dataclass(frozen=True)
class Identity:
    """Who a token or session cookie belongs to: the claims the routes use."""
    uid: str
    email: str
    email_verified: bool
    provider: str          # "password" or "google.com" (the two methods turned on)
    auth_time: int         # when the user actually signed in (Unix seconds)


def identity(claims: dict) -> Identity:
    """Pick the fields we use out of a decoded token or cookie (a dict of claims)."""
    return Identity(uid=claims["uid"], email=claims.get("email", ""),
                    email_verified=bool(claims.get("email_verified")),
                    provider=claims["firebase"]["sign_in_provider"],
                    auth_time=int(claims["auth_time"]))


class FirebaseAuth:
    def __init__(self):
        load_dotenv()
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            raise KeyError("GOOGLE_APPLICATION_CREDENTIALS")   # fail early, by name
        # No arguments: the SDK reads the key file named by GOOGLE_APPLICATION_CREDENTIALS
        self.app = firebase_admin.initialize_app()

    def verify_id_token(self, id_token: str) -> Identity:
        try:
            # check_revoked: also refuse a token issued before the user's last logout
            claims = auth.verify_id_token(id_token, app=self.app, check_revoked=True,
                                          clock_skew_seconds=CLOCK_SKEW_S)
        except (auth.InvalidIdTokenError, auth.UserDisabledError, ValueError) as e:
            raise AuthFailed(str(e)) from e
        return identity(claims)

    def create_session_cookie(self, id_token: str, expires_in: timedelta) -> str:
        try:
            return auth.create_session_cookie(id_token, expires_in, app=self.app)
        except (auth.InvalidIdTokenError, ValueError) as e:
            raise AuthFailed(str(e)) from e

    def verify_session_cookie(self, cookie: str) -> Identity:
        try:
            # check_revoked asks Firebase whether the user logged out (or was disabled)
            # after this cookie was made: one extra call to Google per request.
            claims = auth.verify_session_cookie(cookie, check_revoked=True, app=self.app,
                                                clock_skew_seconds=CLOCK_SKEW_S)
        except (auth.InvalidSessionCookieError, auth.UserDisabledError, ValueError) as e:
            raise AuthFailed(str(e)) from e
        return identity(claims)

    def revoke(self, uid: str) -> None:
        """Firebase can't end one session; this ends all of them (every device)."""
        auth.revoke_refresh_tokens(uid, app=self.app)


_firebase = None


def get_firebase() -> FirebaseAuth:
    """Made on first use, like get_engine / get_redis: initialize_app may run only once."""
    global _firebase
    if _firebase is None:
        _firebase = FirebaseAuth()
    return _firebase
