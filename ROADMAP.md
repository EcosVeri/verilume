# Verilume Roadmap

> Privacy-first desktop AI for documents, research, evidence verification, and hybrid local plus web search.

This roadmap is intentionally public. It gives contributors a clean view of what comes next and where the broader EcosVeri ecosystem is heading. Shipped work is recorded in the [changelog](CHANGELOG.md).

## Version 1.1

- [ ] Windows builds
- [ ] PyPI release
- [ ] Better ranking
- [ ] Conversation memory polish
- [ ] Release automation with checksums
- [ ] Installer smoke tests
- [ ] PDF preview panel with open-at-page and jump-to-page (see below)
- [ ] Unified colour language across the UI (see "Design System" below)
- [ ] Colour-coded search-mode indicator (dot per mode, matching the colour language)

## Version 2.0

- [ ] Ollama-first local setup
- [ ] Vision models
- [ ] Local embeddings controls
- [ ] Knowledge graphs
- [ ] Multi-agent pipeline
- [ ] Coordinate-accurate source highlighting in the PDF viewer (see below)
- [ ] Click-to-source navigation from a citation straight into the PDF
- [ ] Evidence timeline: Query → Retrieval → Reranking → Verification → Answer

## Retrieval & Reasoning Engine

The UI and evidence layer are strong; retrieval and query understanding are the
current accuracy ceiling. This is the plan to make the backend behave like a
modern retrieval engine rather than a conventional RAG lookup.

**Query understanding**

- [ ] Conversation State agent: track entities, current focus, and topic across turns instead of raw `history[-k:]`; rewrite "he/it" to the resolved entity before retrieval.
- [ ] Named-entity resolution (spaCy / GLiNER / HF NER) so "Luc Frieden" resolves to PERSON + COUNTRY + TYPE, with an entity boost during retrieval.
- [ ] Entity expansion (Luc Frieden → Prime Minister, Luxembourg, CSV, government, biography).
- [ ] Query rewriting: generate 5–10 diverse sub-queries per question.
- [ ] Query classification (fact / biography / comparison / calculation / time-sensitive / follow-up / opinion) routing to tailored pipelines.

**Retrieval**

- [ ] True parallel hybrid retrieval: run local BM25, local dense, web, and model knowledge concurrently and merge — not local-then-web-then-AI fallback.
- [ ] Larger candidate pool: BM25 top-100 + dense top-100 → merge → cross-encoder → top-N, instead of K≈7.
- [ ] Cross-encoder reranking (bge-reranker-v2 / ms-marco / jina-reranker-v2) over cosine similarity.
- [ ] Source-quality scoring v2: explicit numeric authority tiers (government 1.0, peer-reviewed 0.98, university 0.95, Wikipedia 0.75, blog 0.45, unknown 0.25) folded into ranking. *(v1 shipped: authority tiers + stock/aggregator downranking.)*
- [ ] Freshness reasoning v2: parse publication/as-of dates and boost recency for time-sensitive queries, rather than relying on past-tenure phrasing. *(v1 shipped: office-scoped past-tenure downranking.)*

**Evidence fusion**

- [ ] Always combine model knowledge with retrieved evidence (model drafts + facts + confidence), never model-as-fallback-only.
- [ ] Claim-level extraction: pull discrete claims (subject, value, date) and compare claims across sources rather than whole documents.
- [ ] Weighted / Bayesian evidence aggregation where each source votes with an authority weight.
- [ ] Knowledge graph of entities and relations (predecessor/successor, party, office, dates) for zero-search relational answers.

**Agentic pipeline (post-launch, v1.1+)**

Compose the pieces above into one explicit, staged pipeline where every stage is
an inspectable agent with typed inputs/outputs — the same shape modern AI search
assistants (Perplexity, Gemini, ChatGPT Search) approximate. Target ordering:

- [ ] **Intent agent** — classify the query (fact / biography / comparison / calculation / time-sensitive / follow-up / opinion / conversation) and pick the pipeline variant.
- [ ] **Conversation agent** — maintain evolving conversation state (entities, current focus, topic, roles) and rewrite references ("he", "it", "there") to the resolved entity before anything else runs.
- [ ] **NER agent** — extract and type entities (PERSON / ORG / GPE / DATE) via spaCy / GLiNER / an HF NER model.
- [ ] **Entity-expansion agent** — expand each entity to related terms (office, party, country, aliases, biography).
- [ ] **Query-rewrite agent** — generate 5–10 diverse sub-queries from the resolved question.
- [ ] **Search planner** — decide which sources to hit per sub-query and allocate budget.
- [ ] **Parallel retrieval** — run concurrently and merge: local BM25, local dense, web, government, Wikipedia, news, academic, model knowledge, knowledge graph.
- [ ] **Merge + dedup** — normalise, deduplicate, and canonicalise the candidate pool.
- [ ] **Cross-encoder reranker** — score (question, passage) pairs; keep top-N.
- [ ] **Claim-extraction agent** — pull discrete claims (subject, relation, value, date) from the top passages.
- [ ] **Evidence verification** — check each claim against its supporting sources.
- [ ] **Conflict detection** — surface disagreements and freshness conflicts across claims/sources.
- [ ] **Confidence estimation** — weighted / Bayesian aggregation over authority-scored source votes (supersedes the current heuristic + shipped cap).
- [ ] **Generation** — compose the answer strictly from verified claims.
- [ ] **Citation verification** — confirm every citation resolves to a supporting source before display.
- [ ] Expose the run as an inspectable trace in the UI (ties into the "Evidence timeline" 2.0 item).

**Model options**

- [ ] Offer larger local models (Qwen3 14B, Qwen2.5 14B/32B, Llama 3.3 70B) for stronger multi-step reasoning and entity disambiguation; keep Qwen2.5 7B for lightweight deployments. (Note: a bigger model helps reasoning but does not fix the retrieval-level issues above.)

## Design System

A consistent colour language so users learn to read the UI at a glance. The
current palette is close but not yet standardised; this unifies it.

- [ ] Green — Confidence
- [ ] Blue — Local evidence
- [ ] Purple — AI knowledge
- [ ] Cyan — Web evidence
- [ ] Gold — Hybrid
- [ ] Apply the palette consistently across evidence summary, strength bars, knowledge chips, source groups, and the search-mode indicator.
- [ ] Note: unicode has no cyan "circle" glyph, so the web/cyan indicator needs a small coloured HTML dot rather than an emoji.

## PDF Preview & Highlighting

Turning Verilume's page-level citations into a viewable, verifiable source
experience. This is split into two tiers because accurate highlighting depends
on storing page geometry at ingest time, which is the heavier lift.

### Tier 1 — Basic viewer (Version 1.1)

- [ ] Render a PDF inline in a side panel or modal from the file already in `docs_dir` (no re-upload).
- [ ] Open at a specific page — retrieval cards already carry the page number.
- [ ] Page navigation (prev/next, jump-to-page box), zoom, and fit-to-width.
- [ ] Lazy-load / paginate large PDFs so big files do not block the UI.
- [ ] Retrieval-card actions: Open PDF, Preview snippet, Jump to Page.
- [ ] Click a `[S1]` citation to open the source PDF at that page.
- [ ] Serve only files inside `docs_dir`; block path traversal; never expose absolute paths to the browser.
- [ ] Graceful degrade for non-paged sources (Word, Excel) — "open source" without page targeting.
- [ ] Clear "source unavailable" state when a file was moved, renamed, or deleted since indexing.

### Tier 2 — Evidence highlighting (Version 2.0)

- [ ] Highlight the retrieved chunk's text on the page when opened from a citation.
- [ ] Persist per-chunk page geometry (bounding boxes / text spans) in the index schema, with a backfill migration for already-indexed documents.
- [ ] Fuzzy text-match fallback when exact rects are unavailable (highlight or scroll to nearest paragraph).
- [ ] Per-source highlight colors that match the citation badges (`[S1]`, `[S2]`, ...).
- [ ] Highlight over the image layer for OCR'd/scanned PDFs using existing OCR word boxes.
- [ ] "Next / previous evidence" stepper to cycle through highlighted spans in a document.
- [ ] Deep-link state so a reloaded or shared session reopens the same page and highlight.
- [ ] Handle edge cases: chunks spanning two pages, multi-column layouts, rotated pages, tables, formulas, and encrypted PDFs.
- [ ] Side-by-side answer ↔ PDF split view.

### Notes

- Viewer tech decision: `pdf.js` in a Streamlit custom component (selectable text + real highlights) vs. server-rendered page images (simpler, no text selection).
- Accuracy payoff: visible highlights expose retrieval/ranking mistakes to the user, which builds trust and can later feed reranker tuning.

## Future

- [ ] EcosVeri ecosystem
- [ ] ecosveri.dev
- [ ] Plugin system
- [ ] Cloud sync
- [ ] API
- [ ] Enterprise edition

## Benchmark Direction

Planned benchmark coverage will compare retrieval quality, citation quality, answer faithfulness, and ingestion speed across:

```text
Question -> Needle -> Needlite -> Verilume
```

Later comparisons should include LangChain, LlamaIndex, Haystack, NotebookLM-style workflows, and other document AI systems.
