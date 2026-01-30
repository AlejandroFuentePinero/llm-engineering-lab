# Define path to helpers
import sys
from pathlib import Path

SRC_PATH = (Path.cwd().parent / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Imports
import os
from dotenv import load_dotenv
from scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI


# Main function
def web_summary_tool(
    url: str = "http://google.com",
    system_prompt: str = "prompt",
    user_prompt_prefix: str = "prompt",
    model: str = "gpt-4.1-mini",
    max_chars: int = 25_000,
    show: bool = True,
):
    """
    Fetches website content, summarizes it with OpenAI, and optionally displays the result.
    - max_chars: crude safety cap to avoid massive pages / token blowups.
    - show: if True, display Markdown; if False, return the string.
    """

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

    # Load OpenAI
    client = OpenAI()

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

    # Define system and user message for the call

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt_prefix}{website}"},
    ]

    # Call OpenAI
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        summary = resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed: {e}") from e

    # Display a nice Markdown summary
    if show:
        display(Markdown(summary))
        return None
    return summary
