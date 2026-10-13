import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env into environment variables

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",  # point the OpenAI SDK at Groq
)

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",  # check the Models page in the console; pick one that supports tool use
    messages=[{"role": "user", "content": "Explain recursion in one sentence."}],
)
print(resp.choices[0].message.content)

MODEL = "openai/gpt-oss-120b"

first = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "My favourite tag is dp. Remember it."}],
)
print(first.choices[0].message.content)

second = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "My favourite tag is dp. Remember it."},
              {"role": "assistant", "content": "Got it! Your favorite tag is **dp**. If you'd like to talk about anything related to that tag—questions, ideas, or anything else—just let me know!"},
              {"role": "user", "content": "What is my favourite tag?"},
            
              ],
)
print(second.choices[0].message.content)
print(second.usage.prompt_tokens)