import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

resp = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
    json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Explain recursion in one sentence."}],
    },
    timeout=30,
)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))