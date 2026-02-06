from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional
import subprocess


def _chat_complete(
    client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def build_messages(
    question: str,
    code: Optional[str],
    favourite_movie: str,
) -> List[Dict[str, str]]:
    system_prompt = f"""
You are a technical tutor and storyteller for data work (data engineering, data science, machine learning, and general software concepts).

You will receive questions about technical concepts and/or chunks of code.
Explain clearly for an intelligent user and competent coder who may be unfamiliar with the specific topic, jargon, or coding approach.

Analogy (make it the protagonist, not decoration):
- Use ONE central analogy from {favourite_movie} and carry it through the whole explanation.
- The analogy should be the backbone: each paragraph should advance the story AND teach a concept.
- After each story beat, add a short “translation” in plain technical terms (1 sentence max) so it stays precise.
- Do not name-drop lots of characters/places; keep references tight and coherent.

Markdown presentation (clean, not robotic):
- Start with one title (##).
- Use 2–3 short sections (###) max; no numbered sections.
- You may include EITHER one small table (max 5 rows) OR one short bullet list (max 4 bullets).
- Avoid nested bullet lists.

Content requirements:
- Define key terms simply (e.g., bias vs variance) but keep jargon minimal.
- If code is provided: explain what it does using the same analogy-first pattern, then give 2–3 gotchas.

Closing:
End with one short closing line in the vibe of a famous character from {favourite_movie}.

Length:
~180–320 words unless the user asks for more.

Output must be Markdown (no fenced code blocks).
""".strip()

    user_msg = (question or "").strip()
    if code and code.strip():
        user_msg += "\n\nCode to explain:\n" + code.strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


def tech_tutor(
    question: str,
    code: Optional[str] = None,
    favourite_movie: str = "Lord of the Rings",
    openai_model: str = "gpt-5-nano",
    ollama_model: str = "llama3.2",
    run_open_ai: bool = True,
    run_ollama: bool = True,
    ollama_base_url: str = "http://localhost:11434/v1",
    temperature: float = 1,
    show: bool = True,
    pull_ollama: bool = False,
):

    messages = build_messages(question, code, favourite_movie)

    results: Dict[str, str] = {}

    if run_open_ai:
        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")
        client = OpenAI(api_key=api_key)
        results["openai"] = _chat_complete(client, openai_model, messages, temperature)

    if run_ollama:
        if pull_ollama:
            subprocess.run(["ollama", "pull", ollama_model], check=True)
        client = OpenAI(base_url=ollama_base_url, api_key="ollama")
        results["ollama"] = _chat_complete(client, ollama_model, messages, temperature)

    if show:
        try:
            from IPython.display import Markdown, display  # type: ignore

            for k, v in results.items():
                print(f"\n[{k}]")
                display(Markdown(v))
        except Exception:
            for k, v in results.items():
                print(f"\n[{k}]")
                print(v)

    return results
