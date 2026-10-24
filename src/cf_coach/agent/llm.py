"""The LLM interface: the only file that knows which provider and SDK we use.

The agent loop only sees:
    llm.chat(messages, tools) -> LLMReply(text, tool_calls, input_tokens, output_tokens, message)

- GroqLLM:     the real model (Groq's OpenAI-compatible API through the `openai` SDK).
- ScriptedLLM: a fake that returns pre-written replies, for tests and evals (no network).

Switching provider = one new class here; the loop and tools don't change.

Run (one real call):  uv run python -m cf_coach.agent.llm
Design reasons: docs/decisions.md #24.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class ToolCall:
    id: str            # pairs this call with its result (tool_call_id)
    name: str
    arguments: str     # raw JSON string from the model; the tool layer validates it


@dataclass(frozen=True)
class LLMReply:
    text: str | None
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    message: dict      # the assistant turn, ready to append to `messages`


class LLM(ABC):
    model: str

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply: ...


def assistant_message(text: str | None, tool_calls: list[ToolCall]) -> dict:
    """Build the assistant turn in OpenAI format (tool calls must be sent back as-is)."""
    msg: dict = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": c.arguments}}
            for c in tool_calls
        ]
    return msg


def parse_response(resp) -> LLMReply:
    """OpenAI-SDK response object -> LLMReply. Separate so it's testable without network."""
    msg = resp.choices[0].message
    calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}")
             for c in (msg.tool_calls or [])]
    return LLMReply(
        text=msg.content,
        tool_calls=calls,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        message=assistant_message(msg.content, calls),
    )


class GroqLLM(LLM):
    def __init__(self, model: str | None = None):
        from openai import OpenAI      # only the real model needs the SDK
        load_dotenv()
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        # The SDK retries 429 / 5xx / connection errors itself, with backoff.
        self._client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL,
                              max_retries=3, timeout=60)

    def chat(self, messages, tools):
        kwargs = {"tools": tools} if tools else {}
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, **kwargs)
        return parse_response(resp)


@dataclass
class ScriptedLLM(LLM):
    """Returns `replies` one by one and records every `messages` list it was sent."""
    replies: list[LLMReply]
    model: str = "scripted"
    seen: list[list[dict]] = field(default_factory=list)

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        if not self.replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


def scripted_reply(text: str | None = None, calls: list[ToolCall] = ()) -> LLMReply:
    """Shortcut for tests: a reply with zero token counts."""
    calls = list(calls)
    return LLMReply(text, calls, 0, 0, assistant_message(text, calls))


# ---------- Manual check: one real call with one tool ----------

if __name__ == "__main__":
    tools = [{"type": "function", "function": {
        "name": "get_rating",
        "description": "Get a Codeforces user's current rating.",
        "parameters": {"type": "object",
                       "properties": {"handle": {"type": "string"}},
                       "required": ["handle"]},
    }}]
    llm = GroqLLM()
    reply = llm.chat([{"role": "user", "content": "What is tourist's rating?"}], tools)
    print("model:      ", llm.model)
    print("text:       ", reply.text)
    print("tool_calls: ", reply.tool_calls)
    print("tokens:     ", reply.input_tokens, "in /", reply.output_tokens, "out")
    print("message:    ", reply.message)
