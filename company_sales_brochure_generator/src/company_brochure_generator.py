import os
import json
import time
from dotenv import load_dotenv
from urllib.parse import urljoin

from IPython.display import Markdown, display, update_display
from openai import OpenAI

from ai_web_summary_tool.src.scraper import fetch_website_links, fetch_website_contents


# SYSTEM PROMPTS
link_system_prompt = """
You are provided with a list of links found on a webpage.
Decide which links are most relevant to include in a brochure about the company,
such as links to an About page, Company page, Products/Solutions, Pricing, Customers, or Careers/Jobs pages.
Respond in JSON like:

{
  "links": [
    {"type": "about page", "url": "https://full.url/goes/here/about"},
    {"type": "careers page", "url": "https://another.full.url/careers"}
  ]
}
"""

brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""


# USER PROMPTS
def get_links_user_prompt(url: str) -> str:
    links = fetch_website_links(url)
    return (
        f"Here is the list of links on the website {url}.\n"
        "Please decide which of these are relevant web links for a brochure about the company.\n"
        "Respond with full https URLs in JSON format.\n"
        "Do not include Terms of Service, Privacy, or email links.\n\n"
        "Links (some might be relative):\n" + "\n".join(links)
    )


def get_brochure_user_prompt(
    company_name: str, pages_text: str, max_chars: int = 25_000
) -> str:
    prompt = (
        f"You are looking at a company called: {company_name}\n"
        "Here are the contents of its landing page and other relevant pages.\n"
        "Use this information to build a short brochure of the company in markdown (no code blocks).\n\n"
        + pages_text
    )
    return prompt[:max_chars]


# FETCH RELEVANT LINKS
def select_relevant_links(url: str, client: OpenAI, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content or "{}")

    cleaned = {"links": []}
    for item in data.get("links", []):
        link_type = (item.get("type") or "").strip() or "relevant page"
        link_url = (item.get("url") or "").strip()
        if not link_url:
            continue

        full_url = urljoin(url, link_url)

        # Minimal filter to avoid non-web links
        if full_url.startswith(("mailto:", "tel:")):
            continue

        cleaned["links"].append({"type": link_type, "url": full_url})

    return cleaned


# FETCH CONTENT TOO
def fetch_page_and_all_relevant_links(
    url: str, client: OpenAI, model: str, max_pages: int = 6
) -> str:
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url, client=client, model=model)

    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links["links"][:max_pages]:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result


# TYPEWRITER ILLUSION OUTPUT
def stream_markdown_typewriter(
    stream, min_chars_per_update: int = 40, min_seconds_per_update: float = 0.05
) -> str:
    response = ""
    buffer = ""
    last_update = time.time()

    handle = display(Markdown(""), display_id=True)

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue

        buffer += token
        now = time.time()

        if (
            len(buffer) >= min_chars_per_update
            or (now - last_update) >= min_seconds_per_update
        ):
            response += buffer
            buffer = ""
            update_display(Markdown(response), display_id=handle.display_id)
            last_update = now

    if buffer:
        response += buffer
        update_display(Markdown(response), display_id=handle.display_id)

    return response


def brochure_generator(
    company_name: str, url: str, model: str = "gpt-4.1-mini", max_pages: int = 6
) -> str:
    load_dotenv(override=True)
    client = OpenAI()

    pages_text = fetch_page_and_all_relevant_links(
        url, client=client, model=model, max_pages=max_pages
    )
    user_prompt = get_brochure_user_prompt(company_name, pages_text)

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    return stream_markdown_typewriter(stream)
