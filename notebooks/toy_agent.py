import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252, which can't print some model output
load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 5  # hard cap on model calls, so a confused model can't loop forever


# ---------- 1. The tools: plain Python functions with fake data ----------

def get_tag_profile():
    """Fake per-tag ratings. Later this will come from the Elo model."""
    return {"dp": 1250, "graphs": 1400, "greedy": 1550, "binary search": 1100}


def find_problems(tag, min_rating, max_rating):
    """Fake problem search. Later this will query Postgres."""
    fake_db = [
        {"id": "1850C", "tag": "binary search", "rating": 1200},
        {"id": "1873E", "tag": "binary search", "rating": 1300},
        {"id": "1901B", "tag": "dp", "rating": 1300},
        {"id": "1882C", "tag": "greedy", "rating": 1600},
        {"id": "1907D", "tag": "binary search", "rating": 1500},
    ]
    return [
        p for p in fake_db
        if p["tag"] == tag and min_rating <= p["rating"] <= max_rating
    ]


# Name the model uses -> the Python function we actually run.
TOOL_FUNCTIONS = {
    "get_tag_profile": get_tag_profile,
    "find_problems": find_problems,
}


# ---------- 2. The tool schemas: what the model is told about each tool ----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_tag_profile",
            "description": "Get the user's current skill rating for each Codeforces tag.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_problems",
            "description": "Find Codeforces problems with a given tag inside a rating range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Tag name, e.g. 'dp'"},
                    "min_rating": {"type": "integer"},
                    "max_rating": {"type": "integer"},
                },
                "required": ["tag", "min_rating", "max_rating"],
            },
        },
    },
]


# ---------- 3. Run one tool call safely ----------

def run_tool(name, arguments_json):
    if name not in TOOL_FUNCTIONS:
        return {"error": f"unknown tool {name}"}
    try:
        args = json.loads(arguments_json)  # the model sends arguments as a JSON string
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        # Send the error back to the model instead of crashing, so it can retry.
        return {"error": str(e)}


# ---------- 4. The agent loop ----------

def run_agent(user_question):
    messages = [
        {"role": "system", "content": "You are a Codeforces practice coach. Use the tools to look up data; never invent ratings or problems."},
        {"role": "user", "content": user_question},
    ]

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        # resp and msg are pydantic objects; model_dump_json turns them into readable JSON.
        # print(f"===== resp (step {step}) =====\n{resp.model_dump_json(indent=2)}")
        msg = resp.choices[0].message
        # print(f"===== msg (step {step}) =====\n{msg.model_dump_json(indent=2)}")
        print(f"\n--- step {step} (prompt_tokens={resp.usage.prompt_tokens}) ---")

        # No tool calls -> the model is giving its final answer.
        if not msg.tool_calls:
            print("FINAL ANSWER:\n" + msg.content)
            return msg.content

        # Record the model's turn (including which tools it asked for).
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in msg.tool_calls
            ],
        })

        # Run each requested tool and send the result back, matched by tool_call_id.
        for call in msg.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            print(f"TOOL {call.function.name}({call.function.arguments}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    print(f"\nStopped: hit MAX_STEPS={MAX_STEPS} without a final answer.")
    return None


if __name__ == "__main__":
    run_agent("Find me 2 problems for my weakest tag, slightly above my current rating there.")
