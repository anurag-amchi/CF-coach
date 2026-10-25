from types import SimpleNamespace as NS

import pytest

from cf_coach.agent.llm import ScriptedLLM, ToolCall, parse_response, scripted_reply


def fake_response(content=None, tool_calls=None, prompt=10, completion=5):
    """Same shape as the openai SDK's response object, built from SimpleNamespace."""
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg)],
              usage=NS(prompt_tokens=prompt, completion_tokens=completion))


def test_parse_text_reply():
    r = parse_response(fake_response(content="hi"))
    assert r.text == "hi" and r.tool_calls == []
    assert (r.input_tokens, r.output_tokens) == (10, 5)
    assert r.message == {"role": "assistant", "content": "hi"}   # no tool_calls key


def test_parse_tool_call_keeps_raw_arguments_and_builds_message():
    call = NS(id="c1", function=NS(name="get_rating", arguments='{"handle": "tourist"}'))
    r = parse_response(fake_response(tool_calls=[call]))
    assert r.tool_calls == [ToolCall("c1", "get_rating", '{"handle": "tourist"}')]
    assert r.message["tool_calls"][0] == {
        "id": "c1", "type": "function",
        "function": {"name": "get_rating", "arguments": '{"handle": "tourist"}'}}


def test_empty_arguments_become_empty_object():
    call = NS(id="c1", function=NS(name="f", arguments=""))
    assert parse_response(fake_response(tool_calls=[call])).tool_calls[0].arguments == "{}"


def test_scripted_llm_replays_in_order_and_records_messages():
    llm = ScriptedLLM([scripted_reply("one"), scripted_reply("two")])
    assert llm.chat([{"role": "user", "content": "a"}], []).text == "one"
    assert llm.chat([{"role": "user", "content": "b"}], []).text == "two"
    assert [m[0]["content"] for m in llm.seen] == ["a", "b"]
    with pytest.raises(RuntimeError):
        llm.chat([], [])


def test_groq_tool_use_failed_becomes_tool_call_rejected():
    import httpx
    from openai import BadRequestError

    from cf_coach.agent.llm import GroqLLM, ToolCallRejected

    def create(**kwargs):
        response = httpx.Response(400, request=httpx.Request("POST", "https://groq.test"))
        raise BadRequestError("400", response=response, body={
            "message": "parameters for tool validate_plan did not match schema",
            "code": "tool_use_failed"})

    llm = GroqLLM.__new__(GroqLLM)          # skip __init__: no API key, no network
    llm.model = "test"
    llm._client = NS(chat=NS(completions=NS(create=create)))
    with pytest.raises(ToolCallRejected, match="did not match schema"):
        llm.chat([{"role": "user", "content": "hi"}], [])


def test_scripted_llm_raises_an_exception_reply():
    llm = ScriptedLLM([RuntimeError("down")])
    with pytest.raises(RuntimeError, match="down"):
        llm.chat([], [])
