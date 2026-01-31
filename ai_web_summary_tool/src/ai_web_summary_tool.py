# Define path to helpers
import sys
from pathlib import Path

SRC_PATH = (Path.cwd().parent / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Imports
import os
import subprocess
from dotenv import load_dotenv
from scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI
from typing import List, Dict


# Helpers


def _prepare_website(url, max_chars):
    # Fetch content
    try:
        website = fetch_website_contents(url)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch website content from {url}: {e}") from e

    if not website or not website.strip():
        raise RuntimeError(f"No readable content returned from {url}.")

    # Simple cap to prevent huge prompts
    website = website.strip()
    if len(website) > max_chars:
        website = website[:max_chars] + "\n\n[TRUNCATED]"
    return website


def _prepare_messages(chat_personality, website):
    # Define system and user message for the call

    system_prompt = f"You are a {chat_personality} assistant that analyzes the contents of a website, and provides a short, {chat_personality}, easy-to-follow summary, ignoring text that might be navigation related. Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown."

    user_prompt_prefix = """
    Here are the contents of a website.
    Provide a short summary of this website.
    If it includes news or announcements, then summarize these too.
    """.strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt_prefix}\n\n{website}"},
    ]
    return messages


def _chat_complete(
    client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float
) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(f"Model request failed (model={model}): {e}") from e

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(f"Empty response returned (model={model}).")

    return content


# Main function


def web_summary_tool(
    url: str = "http://google.com",
    chat_personality: str = "snarky",
    openai_model: str = "gpt-4.1-mini",
    ollama_model: str = "llama3.2",
    max_chars: int = 25_000,
    show: bool = True,
    run_open_ai: bool = True,
    run_ollama: bool = True,
    ollama_base_url: str = "http://localhost:11434/v1",
    temperature: float = 0.2,
):
    """
    Fetches website content, summarizes it with OpenAI, and optionally displays the result.
    - max_chars: crude safety cap to avoid massive pages / token blowups.
    - show: if True, display Markdown; if False, return the string.
    """
    if not (run_open_ai or run_ollama):
        raise ValueError("At least one of run_open_ai or run_ollama must be True.")

    website = _prepare_website(url, max_chars)
    messages = _prepare_messages(chat_personality, website)
    results: Dict[str, str] = {}

    if run_open_ai:

        # Load and check the KEY
        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY")

        # Validate API key
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Add it to your .env file (and ensure the notebook kernel uses this repo's venv)."
            )
        if api_key.strip() != api_key:
            raise RuntimeError(
                "OPENAI_API_KEY has leading/trailing whitespace. Remove spaces/tabs in your .env."
            )
        if not api_key.startswith("sk-"):
            raise RuntimeError(
                "OPENAI_API_KEY doesn't look like a valid key (expected it to start with 'sk-')."
            )

        print(f"Calling OpenAI. Model {openai_model}\n")
        # Load OpenAI
        openai_client = OpenAI(api_key=api_key)
        results["openai"] = _chat_complete(
            client=openai_client,
            model=openai_model,
            messages=messages,
            temperature=temperature,
        )

    if run_ollama:
        print("Downloading Ollama model...\n")
        try:
            subprocess.run(
                ["ollama", "pull", ollama_model],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "Ollama CLI not found. Install Ollama and ensure `ollama` is on PATH."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to pull Ollama model {ollama_model}: {e.stderr}"
            ) from e

        print(f"Calling Ollama. Model {ollama_model}\n")

        ollama_client = OpenAI(base_url=ollama_base_url, api_key="ollama")
        results["ollama"] = _chat_complete(
            client=ollama_client,
            model=ollama_model,
            messages=messages,
            temperature=temperature,
        )

        # Display a nice Markdown summary
    if show:
        if "openai" in results:
            print("\n🤖 OpenAI response:")
            display(Markdown(results["openai"]))
        if "ollama" in results:
            print("\n\n🦙 Ollama response:")
            display(Markdown(results["ollama"]))

    return results
