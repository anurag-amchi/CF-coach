import json

import pytest
from pydantic import BaseModel, ValidationError

from cf_coach.api.contract import (
    ENDPOINTS, RUN_EVENT, DoneEvent, ErrorBody, ErrorEvent, ItemActionIn, RunIn, SessionIn,
    StepEvent, sse,
)


def test_each_method_and_path_appears_once():
    keys = [(e.method, e.path) for e in ENDPOINTS]
    assert len(keys) == len(set(keys))


def test_every_path_is_under_api():
    assert all(e.path.startswith("/api/") for e in ENDPOINTS)


def test_every_model_builds_a_json_schema():
    # FastAPI needs this for /openapi.json, and the TypeScript types come from that.
    models = {m for e in ENDPOINTS for m in (e.body, e.response) if m is not None}
    for m in models:
        assert issubclass(m, BaseModel)
        assert m.model_json_schema()["type"] == "object"


def test_bodies_are_only_on_post_and_patch():
    for e in ENDPOINTS:
        if e.body is not None:
            assert e.method in ("POST", "PATCH"), e.path


def test_a_request_body_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ItemActionIn.model_validate({"action": "swap", "rating": 3000})


def test_no_endpoint_takes_a_password():
    # Firebase handles passwords in the browser (decision #53): the API must never see one.
    for e in ENDPOINTS:
        if e.body is not None:
            assert "password" not in e.body.model_fields, e.path


def test_a_session_needs_a_token_shaped_string():
    SessionIn(id_token="x" * 900)              # real Firebase ID tokens are ~900-1200 chars
    with pytest.raises(ValidationError):
        SessionIn(id_token="")


def test_only_an_adjust_run_takes_a_plan_id():
    RunIn(kind="plan", handle="TitanCoder1258", request="plan my week")
    RunIn(kind="adjust", handle="TitanCoder1258", request="less dp", plan_id="12")
    with pytest.raises(ValidationError, match="needs plan_id"):
        RunIn(kind="adjust", handle="TitanCoder1258", request="less dp")
    with pytest.raises(ValidationError, match="only an adjust run"):
        RunIn(kind="plan", handle="TitanCoder1258", request="x", plan_id="12")


def test_an_event_frame_parses_back_to_the_same_event_by_its_type():
    ev = StepEvent(step=1, tool="find_problems", summary="dp: 10 problems", ok=True)
    frame = sse(ev)
    assert frame.startswith("event: step\ndata: {") and frame.endswith("\n\n")
    data = frame.split("data: ", 1)[1].strip()
    assert RUN_EVENT.validate_json(data) == ev


def test_the_type_field_picks_the_event_class():
    done = RUN_EVENT.validate_python({"type": "done", "outcome": "saved", "steps": 2,
                                      "input_tokens": 1, "output_tokens": 1, "latency_ms": 5})
    assert isinstance(done, DoneEvent)
    err = RUN_EVENT.validate_python(
        {"type": "error", "error": {"code": "budget_exhausted", "message": "try tomorrow"}})
    assert isinstance(err, ErrorEvent) and err.error == ErrorBody(
        code="budget_exhausted", message="try tomorrow")
    with pytest.raises(ValidationError):
        RUN_EVENT.validate_python({"type": "thinking", "text": "hmm"})


def test_an_error_code_must_be_a_known_one():
    with pytest.raises(ValidationError):
        ErrorBody(code="oops", message="x")
    assert json.loads(ErrorBody(code="rate_limited", message="wait",
                                retry_after_s=30).model_dump_json())["retry_after_s"] == 30
