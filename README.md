## Web Summary Tool

This subproject is a small learning lab to practice a clean “fetch → prompt → summarise” flow with the OpenAI API.

### Problem

When I’m learning or researching, I often want a quick, structured summary of a webpage without manually copying the content into a prompt.

### Business applications

This tool uses an LLM API to generate concise summaries from long, unstructured text. Common uses include news/research briefings, report highlights, stakeholder updates, and first-draft writing support. Treat results as a draft—validate key facts and avoid sensitive inputs.

### What I built

A lightweight notebook-first tool that:

1) Fetches the readable text content from a target URL (via a helper in `src/`).
2) Builds a simple prompt (system prompt + user prompt prefix + website text).
3) Calls an OpenAI chat model to generate a summary.
4) Displays the result nicely in the notebook as Markdown.

The core entry point is `web_summary_tool(...)`.

### How to use

Open the notebook in `OpenAI-API-Call/Notebooks/`, run the setup cells, then call:

```python
web_summary_tool(
    url="https://example.com",
    system_prompt="You are a helpful assistant. Summarise the page clearly.",
    user_prompt_prefix="Summarise this webpage:\n\n",
    model="gpt-4.1-mini",
)
```

> Notes:
> The tool expects OPENAI_API_KEY to be available via your environment (loaded from .env).
> The page fetch logic lives in OpenAI-API-Call/src/ so the notebook stays focused on the workflow.