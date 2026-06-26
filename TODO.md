# Verilume TODO

Remaining work for making Verilume easier to install, extend, and trust as a
local-first research assistant.

---

# Release Priorities

- Ship a single downloadable macOS desktop installer (`.dmg` or `.pkg`).
- Sign and notarize the macOS app so end users can install it normally.
- Publish `verilume` to PyPI.
- Add automated smoke tests for installing from GitHub and PyPI.
- Add a Windows installer or executable if Windows packaging tests pass.
- Automate releases with builds, checksums, and smoke tests.

---

# Ollama and Local Models

Ollama support exists in the app. These remaining tasks focus on setup clarity,
model discovery, and offline-friendly use.

- Show clearly whether Ollama is installed and running.
- Show the Ollama models already available in the local Ollama runtime.
- If a selected Ollama model is missing, show the exact command to install it.
- Add a one-click local-only mode that uses Ollama and turns web search off.
- Explain Ollama in the app as a local AI runtime.

---

# Other Model Support

- Add independently developed online model providers alongside hosted APIs.
- Add setup screens for any future model provider that needs keys, endpoints, or model IDs.

---

# Retrieval and Answers

- Tune ranking thresholds using real document collections.
- Improve entity verification for difficult person, company, and same-name lookups.
- Improve query decomposition for complex multi-part questions.
- Add two-stage retrieval: choose the best document first, then the best chunks.
- Add persisted document summaries for summary-first retrieval.
- Add broader semantic duplicate clustering across mirrored or repeated web sources.
- Expand answer verification tests for edge cases and conflicting evidence.
- Add clearer user messages when evidence is weak or incomplete.

---

# Documents and Vision

- Improve metadata extraction beyond the current manifest and extracted fields.
- Improve visual understanding beyond the current OCR, formula, and fallback-caption path.
- Add support for Excel, HTML, XML, EPUB, JSON, and ZIP archives.

---

# App Experience

- Add conversation search.
- Add source filtering.
- Add a document explorer.
- Add clickable page-level highlights for local citations.
- Add search suggestions and autocomplete.
- Add keyboard shortcuts.
- Add streaming tokens.
- Add preference memory for recurring user choices.
- Add opt-in anonymous latency analytics.
- Improve mobile layout.

---

# Integrations and Platform

- Add GitHub, Google Drive, OneDrive, Dropbox, Notion, Confluence, and SharePoint connectors.
- Add a plugin system, Python SDK, and REST API.
- Add Docker images for advanced deployments.
- Expand benchmarks for retrieval quality and ingestion speed.

---

# Later

- Voice assistant.
- Mobile app.
- Browser extension.
- Email, Slack, and Teams assistants.
