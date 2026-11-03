"""Sign-in, sign-out and the signed-in account (slice 4a, decision #53, #60).

The flow:
  1. The browser signs in with the Firebase JS SDK (email + password, or Google) and
     gets an ID token: a JWT signed by Google, valid for 1 hour.
  2. POST /api/auth/session {id_token}: we verify it, check the email is verified, make
     a 5-day Firebase session cookie, create the accounts row on the first sign-in, and
     send the cookie as httpOnly + Secure + SameSite=Lax.
  3. Every later request carries the cookie by itself (the browser adds it, also on the
     SSE stream). Routes that need an account take `me: CurrentAccount`.
  4. POST /api/auth/logout: revoke the user's sessions in Firebase, delete the cookie.

The API never sees a password. Firebase is reached only through api/firebase.py, so the
tests swap in a fake (FakeFirebase) and need no network.
"""

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Response
from sqlalchemy.orm import Session

from cf_coach.api.common import ERRORS, ApiError, DbSession, find_user
from cf_coach.api.contract import ErrorOut, MeOut, MeUpdateIn, SessionIn
from cf_coach.api.firebase import AuthFailed, FirebaseAuth, Identity, get_firebase
from cf_coach.db.models import Account

router = APIRouter(prefix="/api", tags=["auth"])

COOKIE = "cfc_session"
SESSION_LENGTH = timedelta(days=5)     # Firebase allows 5 minutes to 2 weeks
# A session may only be made from a sign-in that just happened. An ID token is valid for
# an hour and is refreshed silently, so without this a stolen token could start new
# sessions long after the user signed in. (Firebase's docs suggest the same 5 minutes.)
RECENT_SIGN_IN_S = 5 * 60
PROVIDERS = {"password", "google.com"}

AUTH_ERRORS = {**ERRORS, 401: {"model": ErrorOut}, 403: {"model": ErrorOut}}

Firebase = Annotated[FirebaseAuth, Depends(get_firebase)]
SessionCookie = Annotated[str | None, Cookie(alias=COOKIE)]


@dataclass
class SignedIn:
    identity: Identity     # from the session cookie
    account: Account       # our row for it


def check_sign_in(who: Identity, now_s: float) -> None:
    """The rules a fresh sign-in must pass before it gets a session. Raises ApiError."""
    if who.provider not in PROVIDERS:
        raise ApiError(401, "unauthorized", "this sign-in method isn't supported")
    if who.provider == "password" and not who.email_verified:
        # Google accounts arrive verified by Google; a password account must have
        # opened the link in the email Firebase sent it.
        raise ApiError(403, "email_not_verified",
                       "open the link in the email we sent you, then sign in again")
    if now_s - who.auth_time > RECENT_SIGN_IN_S:
        raise ApiError(401, "unauthorized", "please sign in again")


def save_account(session: Session, who: Identity) -> Account:
    """Create the account on its first sign-in; keep its email up to date after that."""
    account = session.get(Account, who.uid)
    if account is None:
        account = Account(uid=who.uid, email=who.email)
        session.add(account)
    else:
        account.email = who.email          # it can be changed in Firebase
    session.commit()
    return account


def me_out(who: Identity, account: Account) -> MeOut:
    return MeOut(email=who.email, email_verified=who.email_verified,
                 sign_in_provider=who.provider, handle=account.handle)


def current_account(session: DbSession, fb: Firebase, cookie: SessionCookie = None) -> SignedIn:
    """The dependency behind CurrentAccount: a signed-in account, or a 401.
    FastAPI gives it the same DbSession as the route (one session per request)."""
    if not cookie:
        raise ApiError(401, "sign_in_required", "sign in to do this")
    try:
        who = fb.verify_session_cookie(cookie)
    except AuthFailed:
        raise ApiError(401, "unauthorized", "your session has ended; please sign in again")
    account = session.get(Account, who.uid)
    if account is None:
        raise ApiError(401, "unauthorized", "no account for this session; please sign in again")
    return SignedIn(who, account)


CurrentAccount = Annotated[SignedIn, Depends(current_account)]


@router.post("/auth/session", responses=AUTH_ERRORS)
def post_session(body: SessionIn, response: Response, session: DbSession,
                 fb: Firebase) -> MeOut:
    try:
        who = fb.verify_id_token(body.id_token)
    except AuthFailed:
        raise ApiError(401, "unauthorized", "sign-in failed or expired; please sign in again")
    check_sign_in(who, time.time())
    try:
        cookie = fb.create_session_cookie(body.id_token, SESSION_LENGTH)
    except AuthFailed:
        raise ApiError(401, "unauthorized", "sign-in failed or expired; please sign in again")
    account = save_account(session, who)
    response.set_cookie(
        COOKIE, cookie, max_age=int(SESSION_LENGTH.total_seconds()), path="/",
        httponly=True,     # page scripts can't read it, so injected JS can't steal it
        secure=True,       # sent over HTTPS only (browsers treat localhost as secure)
        samesite="lax",    # not sent on cross-site POSTs (a CSRF defence)
    )
    # Slice 4c: claim the guest plan here.
    return me_out(who, account)


@router.post("/auth/logout", status_code=204, responses=ERRORS)
def post_logout(response: Response, fb: Firebase, cookie: SessionCookie = None) -> None:
    """Always succeeds: an expired or missing cookie means you are already signed out."""
    if cookie:
        try:
            fb.revoke(fb.verify_session_cookie(cookie).uid)
        except AuthFailed:
            pass                            # already ended: nothing to revoke
    response.delete_cookie(COOKIE, path="/", httponly=True, secure=True, samesite="lax")


@router.get("/me", responses=AUTH_ERRORS)
def get_me(me: CurrentAccount) -> MeOut:
    return me_out(me.identity, me.account)


@router.patch("/me", responses=AUTH_ERRORS)
def patch_me(body: MeUpdateIn, me: CurrentAccount, session: DbSession) -> MeOut:
    # No ownership check (decision #52): CF data is public, and an account only ever
    # plans for its own handle. Keep CF's spelling if this handle was synced already.
    user = find_user(session, body.handle)
    me.account.handle = user.handle if user else body.handle
    session.commit()
    return me_out(me.identity, me.account)
