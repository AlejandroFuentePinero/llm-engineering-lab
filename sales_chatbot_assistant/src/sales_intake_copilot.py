import os
import json
from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_MESSAGE = """
You are a B2B sales assistant. Your job is to:
1) Reply to the user naturally.
2) Ask at most ONE qualifying question if needed (keep it lightweight).
3) Also produce an INTERNAL HANDOFF NOTE for a human sales rep.

Output format MUST be exactly:

<assistant_reply>

---HANDOFF---
Use case:
Industry:
Company size:
Timeline:
Budget:
Authority:
Next step:
---END---

Rules:
- If you don't know a field, write "Unknown".
- Never invent facts.
- Keep the assistant reply short and helpful.
"""

HANDOFF_MARKER = "\n---HANDOFF---\n"
END_MARKER = "\n---END---"


def _visible_part(text: str) -> str:
    if HANDOFF_MARKER in text:
        return text.split(HANDOFF_MARKER, 1)[0].strip()
    return text.strip()


def _handoff_part(text: str) -> str:
    if HANDOFF_MARKER in text:
        handoff = text.split(HANDOFF_MARKER, 1)[1]
        return handoff.replace(END_MARKER, "").strip()
    return ""


def sales_assistant_stream(message: str = "", history=None):
    load_dotenv(override=True)
    client = OpenAI()

    history = history or []
    history = [{"role": h["role"], "content": h["content"]} for h in history]

    messages = (
        [{"role": "system", "content": SYSTEM_MESSAGE}]
        + history
        + [{"role": "user", "content": message}]
    )

    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        stream=True,
        temperature=0.3,
    )

    response = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            response += token
            yield _visible_part(response)

    print("\nINTERNAL HANDOFF:\n", _handoff_part(response))
