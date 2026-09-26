# Issue Triage Agent

An AI agent that triages new GitHub issues: it labels them, detects duplicates,
asks for missing info, and drafts a helpful reply. Built on the **free tier** of
the Gemini API using function calling.

## How it works
```
Issue opened -> GitHub Action -> python -m triage.main
  -> Gemini agent loop with tools:
       search_similar_issues (embeddings + cosine similarity)
       add_labels (allow-listed only)
       post_comment
       read_repo_file
```

## Safety
- Issue text is treated as untrusted; the system prompt tells the model to ignore instructions inside it.
- Tools are deliberately limited: no closing, editing or deleting issues; labels are allow-listed.
- `DRY_RUN=true` (default locally) prints planned actions without touching GitHub.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env    # add GEMINI_API_KEY (free at aistudio.google.com/apikey)
python -m triage.main --issue 1
pytest
```

To run it automatically, add `GEMINI_API_KEY` as a repository secret; the workflow in
`.github/workflows/triage.yml` does the rest.

## Free-tier notes
- Free-tier Gemini content may be used by Google to improve its products: use public repos only.
- Rate limits are per project (see aistudio.google.com/rate-limit); 429s are retried with backoff.

## Roadmap
- [ ] Persist embeddings so the duplicate check doesn't re-embed every run
- [ ] Human-approval mode (post suggestions as one comment for maintainers)
- [ ] Provider-agnostic LLM interface (Groq / Ollama)
