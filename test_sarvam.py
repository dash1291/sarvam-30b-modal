#!/usr/bin/env python3
"""Quick test script for sarvam-30b running via vllm"""

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://89.169.122.229:8000").rstrip("/") + "/v1",
    api_key=os.environ.get("LLM_API_KEY", "dummy"),
)

# List available models
models = client.models.list()
print(models)
MODEL = models.data[0].id
print(f"Model: {MODEL}\n")

tests = [
    {
        "name": "Basic English",
        "messages": [{"role": "user", "content": "What is the capital of India?"}],
    },
    {
        "name": "Tamil",
        "messages": [{"role": "user", "content": "நீங்கள் எப்படி இருக்கீர்கள்?"}],
    },
    {
        "name": "Kannada",
        "messages": [{"role": "user", "content": "ನೀವು ಹೇಗಿದ್ದೀರಿ?"}],
    },
    {
        "name": "Hindi",
        "messages": [{"role": "user", "content": "भारत की राजधानी क्या है?"}],
    },
    {
        "name": "Code generation",
        "messages": [{"role": "user", "content": "Write a Python function to reverse a string."}],
    },
]

for test in tests:
    print(f"--- {test['name']} ---")
    print(f"Q: {test['messages'][0]['content']}")
    print("A: ", end="", flush=True)
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=test["messages"],
            max_tokens=200,
            temperature=0.7,
            stream=True,
            stream_options={"include_usage": True},
        )
        usage = None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                print(delta, end="", flush=True)
        print()
        if usage:
            print(f"   ({usage.prompt_tokens} prompt / {usage.completion_tokens} completion tokens)")
        print()
    except Exception as e:
        print(f"FAILED: {e}\n")
