## Web Summary Tool

A small, deployable Python utility that turns a webpage URL into a concise Markdown summary using an LLM. It’s designed to be embedded into internal workflows where people need quick, repeatable briefs from unstructured web content.

### Business problem

Stakeholders often need to extract decision-relevant information from long webpages (announcements, reports, research posts). Manual summarisation is slow, inconsistent, and doesn’t scale across many sources.

### What it does

Given a URL, `web_summary_tool(...)`:
- fetches and extracts the readable text from the webpage
- applies a safety cap (`max_chars`) to prevent oversized prompts
- generates a short summary in Markdown via a chat model
- can run via either a hosted API (OpenAI) or a local open-source model (Ollama)
- either returns the summary text or renders it (controlled by `show`)

### Tone / “personality” control

The tool accepts a `chat_personality` parameter to adapt tone and framing to the audience. This is useful when the same underlying content needs to be summarised differently depending on context (e.g., a terse executive brief vs. a more detailed analyst-style summary). The output remains Markdown so it can drop cleanly into docs, notes, or downstream pipelines.

### Interface

`web_summary_tool(url, chat_personality="…", openai_model="…", ollama_model="…", max_chars=25000, show=True, run_open_ai=True, run_ollama=True)`

- `url`: webpage to summarise  
- `openai_model`: hosted chat model used when `run_open_ai=True`  
- `ollama_model`: local model used when `run_ollama=True`  
- `max_chars`: crude guardrail to limit prompt size  
- `show`: if `True`, displays Markdown; if `False`, returns results as strings  
- `run_open_ai`: enable OpenAI backend (requires `OPENAI_API_KEY`)  
- `run_ollama`: enable Ollama backend (requires Ollama installed and running locally)
