"""The FastAPI app: the routes plus one error shape for every failure.

Every error the API sends is ErrorOut, {"error": {"code", "message", "retry_after_s"}},
whoever raised it:
  ApiError (our routes)            -> its own status and code
  RequestValidationError (FastAPI) -> 422 "validation"  (bad path, query or body)
  HTTPException (routing)          -> 404 "not_found"   (no such path, or wrong method)
  any other exception (a bug)      -> 500 "internal"    (traceback in the server log only)

Run:  uv run python -m cf_coach.api.app
      then http://127.0.0.1:8000/docs (try the routes) or /openapi.json (the schema)
Design reasons: docs/decisions.md #52, #58.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from cf_coach.api import auth, plans, users
from cf_coach.api.common import ApiError
from cf_coach.api.contract import ErrorBody, ErrorCode, ErrorOut

log = logging.getLogger("cf_coach.api")

app = FastAPI(title="CF Coach API", version="0.1.0")
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(plans.router)


def error_response(status: int, code: ErrorCode, message: str,
                   retry_after_s: int | None = None) -> JSONResponse:
    body = ErrorOut(error=ErrorBody(code=code, message=message, retry_after_s=retry_after_s))
    # Retry-After is the standard header for "try again in N seconds"
    headers = {"Retry-After": str(retry_after_s)} if retry_after_s is not None else None
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(exc.status, exc.code, exc.message, exc.retry_after_s)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Each error has a loc like ("path", "handle") and a msg; keep the field name and msg.
    parts = [f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
    return error_response(422, "validation", "; ".join(parts))


@app.exception_handler(HTTPException)
async def routing_error(request: Request, exc: HTTPException) -> JSONResponse:
    # Our routes raise ApiError, so this comes from routing: no route has this path (404)
    # or the path exists for another method (405).
    return error_response(exc.status_code, "not_found",
                          f"no such endpoint: {request.method} {request.url.path}")


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # The traceback is logged; the browser gets no internals (they can leak SQL, paths).
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return error_response(500, "internal", "something went wrong on our side")


if __name__ == "__main__":
    import uvicorn

    # reload=True restarts the server when a .py file changes (development only)
    uvicorn.run("cf_coach.api.app:app", host="127.0.0.1", port=8000, reload=True)
