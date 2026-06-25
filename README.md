# 📚🔎 Verilume

[![CI](https://github.com/DamingoNdiwa/verilume/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DamingoNdiwa/verilume/actions/workflows/ci.yml)
**The local-first evidence layer for private documents, web search, and AI reasoning.**

Verilume is a desktop RAG assistant that lets you ask questions across local documents, trusted web sources, and AI knowledge. It ranks evidence, separates local `[S1]` citations from web `[W1]` citations, reconciles conflicting sources, and exports chats to Markdown or PDF.

Built for researchers, analysts, students, consultants, and non-programmers who need transparent answers from their own files and the web.

## Why Verilume?

Most RAG tools either search local files or the web. Verilume combines both, then ranks and reconciles the evidence before answering.

It is designed to answer questions like:

- What does this PDF say?
- Is this information still current?
- Do local documents disagree with recent web sources?
- Which source supports this claim?
- Can I export the full cited conversation?

## What It Does
## Highlights
- Local-first document question answering
- PDF, DOCX, TXT, Markdown, and CSV ingestion
- Chroma vector database with incremental indexing
- Hugging Face and Ollama generation backends
- Configurable web search providers
- Evidence ranking and source confidence
- Separate local and web citations
- Persistent dark/light appearance toggle
- Conversation-aware query rewriting
- Markdown and PDF chat export
- Streamlit desktop interface
- CLI commands for running, ingestion, stats, config, and health checks.
- macOS launcher and `.app` build helper.

## Install From PyPI

After the first public release on June 23, 2026:

```bash
python -m pip install verilume
verilume run
```

## Install From Source

```bash
git clone git@github.com:DamingoNdiwa/verilume.git
cd verilume
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Launch The App

```bash
verilume run
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8511
```

If that port is already in use, Verilume automatically tries `8512`, then
`8513`. If all three preferred ports are busy, it prints a message and reclaims
`8511` for the new launch. To force a specific port, set `VERILUME_PORT` before
launching.

On macOS, you can also double-click:

```text
Verilume.command
```

The first launch may download the embedding model, which can take a few minutes.

## Complete App Example

1. Start the app with `verilume run`.
2. Enter a Hugging Face token in the sidebar.
3. Choose a web search provider. Tavily is the default.
4. Enter the selected provider API key if that provider requires one.
5. Choose a Hugging Face model from the dropdown, or enter a custom model ID.
6. Click `Save Configuration` to persist your token, web provider, API key, and model choice locally.
7. Upload one or more files, for example `course_overview.pdf`.
8. Click `Build KB`.
9. Wait until the app shows uploaded documents, PDF pages, and chunks indexed.
10. Ask a local document question, for example:

```text
Summarise the uploaded course overview.
```

11. Ask a general model-knowledge question, for example:

```text
What is econometrics?
```

12. Ask for web search explicitly, for example:

```text
Search the web for the latest information about Florian Felice.
```

13. Review local citations under `Local citations` and web citations under `Web citations`.
14. Export the conversation with `Markdown` or `PDF`.

Verilume answers with a local-first weighting strategy:

1. Search local Chroma documents first.
2. If local files clearly answer, answer from local retrieval and cite `[S1]`, `[S2]` sources.
3. For questions asking whether something exists in the local files, database, uploaded documents, or indexed knowledge base, search local files with the original query and an expanded keyword query. If no local chunks are found, answer exactly: `I could not find this in the indexed local files.`
4. Do not answer local-file existence questions from AI Knowledge or web search.
5. If local files do not clearly answer and web search is enabled, use both Hugging Face AI knowledge and web search, then synthesize the final answer by comparing them.
6. For current events, public figures, recent elections, prices, laws, regulations, company roles, sports results, and other time-sensitive topics, use local or web evidence as the source of truth.
7. Validate current-information evidence before final answer generation: rank sources by authority, freshness, and agreement; prefer official government or organization sources; ignore stale past-office pages and old articles for current claims.
8. Require either one official source or two independent credible sources for current role answers. If sources disagree, Verilume mentions the conflict and selects the newest high-authority evidence.
9. Never allow AI knowledge to override newer or more reliable web evidence.
10. If web search is disabled and local files do not answer, Verilume may answer from AI Knowledge, clearly marked as not externally verified.
11. Show colored answer-origin badges for Local Retrieval, Web Search, AI Knowledge, Hybrid, or Current Information.
12. Show local `[S1]`, `[S2]` citations separately from grouped, clickable web source lists.
13. Switch between the default dark appearance and a light readable appearance from the header toggle. The choice is saved immediately and restored after restart.

If the selected Hugging Face model is out of capacity, overloaded, unavailable, unsupported by the current provider, or incompatible with the token, the app shows a warning and asks you to select another sidebar model or enter a compatible custom model ID.

## Configuration

Copy `.env.example` to `.env` and adjust values as needed:

```bash
cp .env.example .env
```

Appearance can also be set directly:

```bash
VERILUME_APPEARANCE=dark
VERILUME_APPEARANCE=light
```

By default, local user data is stored under:

```text
~/.verilume/documents
~/.verilume/chroma_db
~/.verilume/ingestion_manifest.json
~/.verilume/config.env
```

The sidebar `Save Configuration` button writes tokens, provider settings, and model selection to `~/.verilume/config.env`. This file is local to your machine and should not be committed.

Key environment variables:

```text
HF_TOKEN=
HF_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_PROVIDER=auto
ANSWER_STYLE=Standard
TAVILY_API_KEY=
WEB_SEARCH_PROVIDER=tavily
BRAVE_API_KEY=
EXA_API_KEY=
SERPAPI_API_KEY=
BING_API_KEY=
GOOGLE_CSE_API_KEY=
GOOGLE_CSE_ID=
ENABLE_WEB_SEARCH=true
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT_SECONDS=20
RETRIEVER_K=5
RETRIEVAL_SCORE_THRESHOLD=0.35
HF_TIMEOUT_SECONDS=90
```

Supported `WEB_SEARCH_PROVIDER` values are `tavily`, `duckduckgo`, `brave`, `exa`, `serpapi`, `bing`, `google_cse`, and `custom`. DuckDuckGo uses its Instant Answer endpoint and does not require an API key. Google CSE requires both `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_ID`. Custom providers can be configured with `CUSTOM_WEB_SEARCH_PROVIDER`, `CUSTOM_WEB_SEARCH_API_KEY`, and `CUSTOM_WEB_SEARCH_ENDPOINT`; the endpoint may include `{query}` and `{api_key}` placeholders.

Do not commit `.env`, API keys, uploaded documents, Chroma databases, or logs.

## CLI

```bash
verilume run
verilume ingest
verilume ingest --reset
verilume stats
verilume config
verilume doctor
```

`verilume doctor` prints a deployment health report without exposing secret values.

## Citation Rules

Local document citations use `[S1]`, `[S2]`, `[S3]` and include document names plus PDF page numbers when available.

Web citations use `[W1]`, `[W2]`, `[W3]` and render as clickable links.

The model is instructed not to invent citation labels. Unsupported citation labels are stripped before display.

## macOS App Bundle

To build a `.app` bundle:

```bash
python scripts/build_macos_app.py
```

The generated app appears in `dist/`. For a simple local launcher, keep using `Verilume.command`.

## Development Checks

```bash
python -m ruff check src tests launcher.py scripts
python -m compileall -q src tests launcher.py scripts/build_macos_app.py
python -m unittest discover -s tests -v
verilume doctor
```

## Release Checklist

Target release date: June 23, 2026.

1. Confirm `pyproject.toml` has the intended version.
2. Confirm `CHANGELOG.md` has the June 23, 2026 release notes.
3. Run the full local checks.
4. Build from a clean tree:

```bash
rm -rf dist build
find . -name "*.egg-info" -prune -exec rm -rf {} +
python -m build
python -m twine check dist/*
```

5. Tag the release on GitHub as `v0.1.0`.
6. Upload to PyPI only after the GitHub repository metadata and package owner are final.

## Build For PyPI

```bash
rm -rf dist build
find . -name "*.egg-info" -prune -exec rm -rf {} +
python -m build
python -m twine check dist/*
```

Upload only after setting the real package owner and repository metadata:

```bash
python -m twine upload dist/*
```

## License

Apache-2.0. See `LICENSE`.
