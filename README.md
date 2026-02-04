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

## Company Brochure Generator

A small notebook-friendly Python utility that turns a company website into a short, readable Markdown brochure using an LLM. It’s designed for quick prospecting: generating a consistent “who they are / what they do / why they matter” brief for customers, investors, or recruits.

### Business problem

When evaluating companies (for sales, investing, partnerships, or job applications), the relevant information is spread across multiple pages (About, Products, Careers, Customers). Manually collecting and synthesising this is slow, noisy, and inconsistent — especially when you need to do it repeatedly across many companies.

### What it does

Given a company name and homepage URL, `brochure_generator(...)`:
- scrapes the homepage for all available links
- uses a chat model to select a small set of brochure-relevant pages (e.g., About, Products, Careers)
- fetches the text content for the homepage + selected pages
- generates a short brochure in Markdown (no code blocks) covering:
  - what the company does and who it serves
  - products / services and key differentiators (if present)
  - culture and hiring signals (if present)
- streams the output to the notebook with a “typewriter” effect, while also returning the final Markdown string

### Streaming output (“typewriter” mode)

In a notebook context, the generator updates a single Markdown display cell as tokens stream in, creating a typewriter-style reveal. This is purely a presentation layer: the final output is still returned as a plain Markdown string so it can be saved, reused, or exported.

### Interface

`brochure_generator(company_name, url, model="gpt-4.1-mini", max_pages=6)`

- `company_name`: label used to frame the brochure narrative  
- `url`: company homepage to crawl  
- `model`: chat model used for both link selection and brochure generation  
- `max_pages`: maximum number of “relevant” pages to fetch in addition to the landing page
