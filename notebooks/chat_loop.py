import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env into environment variables

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",  # point the OpenAI SDK at Groq
)

messages=[]

questions=["My favourite DSA topic is DP. Remember this","Then I would say Graph, Tries, Binary search in decreasing order, one line reply only", "I hate Binary search, it has so many nuances with mid and left and right, one line reply only", "What is my favourite DSA topic? list them from most favourite to least.","Which is the topic I hate and why? one line reply only"]

for question in questions:
    messages.append({"role":"user","content":question})
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
    )
    reply=resp.choices[0].message.content
    print(reply)
    print(resp.usage.prompt_tokens)
    messages.append({"role":"assistant","content":reply})