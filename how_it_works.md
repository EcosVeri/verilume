# How Verilume Works

This file is intentionally local-only documentation. It is meant to explain the current implementation and usage, and it is not intended to be pushed to GitHub.

## 1. What Verilume Is

Verilume is a local-first retrieval-augmented generation application with three evidence channels:

1. Local files indexed into a Chroma knowledge base.
2. AI model knowledge from the configured generation backend.
3. Web evidence from the configured web search provider.

The app is built so that answers are not chosen from one channel blindly. It gathers evidence, ranks it, reconciles conflicts, verifies citations, and then decides which answer form should be shown.

## 2. Main Implementation Pieces

The main files and their jobs are:

| File | Role |
| --- | --- |
| `src/verilume/app.py` | Streamlit entrypoint. Wires the sidebar, chat view, document actions, and cached RAG service. |
| `src/verilume/cli.py` | CLI entrypoint. Supports `run`, `ingest`, `stats`, `config`, and `doctor`. |
| `src/verilume/settings.py` | Loads environment variables and local user settings from `~/.verilume/config.env`. |
| `src/verilume/ingest.py` | Parses files, normalizes content, chunks text, embeds it, and builds the local Chroma knowledge base. |
| `src/verilume/core/retrieval.py` | Chroma retriever with dense, lexical, and hybrid retrieval. |
| `src/verilume/core/query_interpreter.py` | Interprets user intent, follow-up context, source policy, and search preferences. |
| `src/verilume/core/search_planner.py` | Produces a search plan describing whether local, model knowledge, or web evidence should be used. |
| `src/verilume/rag.py` | Main orchestration layer. It runs local retrieval, AI knowledge generation, web search, reranking, answer selection, citation verification, and final answer validation. |
| `src/verilume/ui/sidebar.py` | Settings, upload controls, remove-document controls, and stats cards. |
| `src/verilume/ui/chat.py` | Chat rendering, source blocks, evidence panels, and exports. |

## 3. Data and Storage Layout

By default, user data is stored in the local user directory:

- `~/.verilume/documents` stores uploaded source files.
- `~/.verilume/chroma_db` stores the Chroma vector database.
- `~/.verilume/ingestion_manifest.json` stores file hashes and ingestion metadata.
- `~/.verilume/config.env` stores local app settings such as tokens and model choices.

This keeps user content outside the repository and makes the app usable as a desktop tool without polluting the project tree.

## 4. How Ingestion Works

When files are uploaded or `verilume ingest` is run, Verilume does the following:

1. Detect the file type.
2. Route the file to a format-specific handler.
3. Extract text page by page for PDF files, slide by slide for PowerPoint files, OCR image uploads directly, and read full text for DOCX, TXT, Markdown, and CSV files.
4. Normalize extracted text before indexing. For PDFs this includes cleanup for broken icon-font fragments and hyphenated line breaks.
4. Split the content into chunks using semantic chunking by default.
5. Compute embeddings for each chunk.
6. Store chunk text, metadata, and embeddings in Chroma.

The current ingest pipeline uses a staged rebuild-and-swap model instead of mutating the live Chroma store in place. In practice that means Verilume builds a temporary Chroma database and manifest first, and only swaps them into `~/.verilume` after a successful ingest. This makes resets and rebuilds much more reliable.

### 4.1 Supported local document types

The local ingest surface now supports:

- PDF
- scanned PDF
- DOCX
- PPTX and related Open XML PowerPoint variants
- image uploads such as PNG, JPG, JPEG, BMP, TIFF, and WebP
- TXT, Markdown, and CSV

This means the local knowledge base can now be built from born-digital documents, scanned documents, image-only evidence, and slide decks without changing the higher-level web or model routing logic.

### 4.2 OCR and scanned-document handling

OCR is handled inside `src/verilume/ingest.py`.

The current behavior is:

1. Normal PDFs still use native text extraction first.
2. A PDF page falls back to OCR only when extracted text is empty or nearly empty.
3. Scanned PDF OCR renders the page with `pypdfium2` and then runs `rapidocr-onnxruntime` over the rendered image.
4. Image uploads are OCRed directly.
5. PowerPoint files extract normal slide text, tables, speaker notes, and can OCR embedded images when needed.

This keeps ordinary PDFs fast while still allowing image-only or scan-heavy documents to become searchable.

### 4.3 Why the staged rebuild is now safer

The staged ingest path had one subtle failure mode during this work: app-side Chroma clients created for stats or staged indexing could keep stale handles open while the database directory was being swapped.

The implementation now closes those temporary Chroma clients before:

1. installing a staged snapshot
2. restoring a backup snapshot
3. reading persisted collection counts for sidebar and header stats
4. mutating the local store from the Streamlit app while a cached RAG retriever from a prior rerun may still be open

That fix prevents successful indexing from being rolled back incorrectly as an "empty retriever" state.

When a build does fail, the Streamlit UI now also surfaces the actual exception text instead of only showing a generic "check the terminal" banner. That makes document-specific ingestion problems much easier to diagnose.

## 5. How Search Works

Verilume does not treat every question the same way. It first classifies the question and then follows a search plan.

### 5.1 Intent routing

Some prompts never enter the full RAG pipeline. Examples are:

- greetings
- thanks
- identity questions such as “Who are you?”
- capability questions such as “What can you do?”

Those are answered directly by the intent router.

### 5.2 Query interpretation

For normal questions, the query interpreter resolves:

- whether the prompt is a follow-up
- whether the user is asking about uploaded local files
- whether the question is time-sensitive or current
- whether the question is a stable fact, dynamic fact, news, scientific explanation, local-document question, person lookup, or company lookup
- which evidence policy applies: `local_only`, `local_plus_model`, `local_plus_web`, `local_model_web`, or `web_only`
- whether web evidence is needed
- which search queries should be generated

This stage also carries forward conversation state such as the active person, country, role, or research topic.

### 5.3 Local retrieval first

For real knowledge questions, Verilume searches the local knowledge base first.

Local retrieval uses one of three modes:

1. Dense retrieval with embeddings.
2. Lexical BM25-style retrieval.
3. Hybrid retrieval using reciprocal-rank fusion.

Some question types force lexical retrieval, especially identity lookups and local-file questions, because exact names and filenames matter more than semantic similarity.

### 5.4 AI knowledge usage

AI knowledge is now part of the normal answer path instead of a last-resort add-on.

The current behavior is:

1. If local evidence exists for a stable non-current question, Verilume still asks the AI model for useful background or explanation.
2. If the AI answer is relevant and sufficient, final synthesis combines the AI knowledge with the local evidence.
3. Local file evidence has more weight than AI knowledge. If they disagree, local evidence wins.
4. If AI knowledge cannot answer or is insufficient, Verilume keeps the answer local-only when local evidence answers the question.
5. If web search is disabled and local evidence is missing, Verilume can answer stable non-current questions from AI knowledge alone.

For model-only answers, the app labels the result as AI knowledge and marks it as not externally verified.

### 5.5 Web search usage

Web search is used when:

1. Web search is enabled and the question is a normal stable/static knowledge question.
2. The question explicitly asks for web search.
3. The question is current, time-sensitive, or otherwise likely to change.
4. Local evidence and AI knowledge are both insufficient and web search is enabled.

For stable/static questions, web search is paired with AI knowledge rather than replacing it. For current or changeable questions, web evidence becomes the source of truth and AI knowledge is not used as final evidence.

Web search is not supposed to override strong, stable local evidence unnecessarily. It supplements or validates when appropriate, and local evidence keeps the highest weight when the question is about uploaded documents.

### 5.6 Local-file exceptions

Local-file existence and direct local-document questions are treated specially.

For example:

- “Is this in my local files?”
- “Which document contains my language certificate?”
- “What date is written on this uploaded certificate?”
- “Summarise the docs in the database.”
- “How many files are in the data base?”
- “In the uploaded file scanned-smoke.pdf, what OCR token appears in the document?”

These are answered strictly from local indexed files. AI knowledge and web search should not invent a local-file answer.

If the user names a specific local filename such as `slides-smoke.pptx` or `scanned-smoke.pdf`, that filename is now preserved as an anchor during local ranking and filtering. This matters for short OCR chunks, where the document name may be the strongest signal.

## 6. How Verilume Decides on the Final Answer

The final answer is chosen by combining several stages.

### 6.1 Candidate answers

The RAG layer may have up to three candidate sources of truth:

1. A local answer candidate from retrieved local chunks.
2. A model-knowledge candidate from the LLM.
3. Web sources, possibly with a web-grounded answer candidate.

### 6.2 Evidence ranking

Local, web, and model evidence are converted into structured evidence items before final synthesis. Each evidence item stores:

- source type: local, web, or AI knowledge
- source name or title
- citation label
- source text
- retrieval score
- cross-encoder or reranker score
- entity-match score for person and company lookups
- authority score
- freshness score
- AI-consistency score
- final evidence score

The final evidence score is weighted as:

```text
final_score =
  0.30 * retrieval_score +
  0.25 * cross_encoder_score +
  0.20 * entity_match_score +
  0.10 * authority_score +
  0.10 * freshness_score +
  0.05 * ai_consistency_score
```

The weights can be adjusted from settings later, but the default behavior is designed to make evidence arbitration happen before the final answer is written.

Local and web sources are also reranked before final synthesis. The system looks at:

- lexical overlap
- semantic similarity
- exact phrase matches
- identity relevance
- source authority
- source freshness for current questions

Authority currently uses these rough trust levels:

| Source class | Default authority score |
| --- | ---: |
| official government / official institution | 1.00 |
| university | 0.95 |
| scientific paper or academic source | 0.95 |
| local indexed document | 0.90 |
| news source | 0.80 |
| Wikipedia | 0.70 |
| AI model knowledge | 0.60 |
| standard web source | 0.55 |
| blog | 0.40 |
| social media | 0.25 |

For person lookups, private identity-document chunks such as passports are demoted unless the question is specifically asking about that document. University profiles, thesis material, supervisor references, CV material, and publication context get a relevance boost. This prevents a passport chunk from dominating a general person search simply because it contains an exact name.

Person lookups also use entity verification before final synthesis. If the query is a bare person name such as `Christophe Ley`, local and web evidence must actually match that queried person. Chunks that foreground a different named person, such as `Gabriella Vinco`, are discarded even if they mention the queried name incidentally. Coursework, regression exercises, invoices, certificates, and other non-profile documents are treated as negative identity context unless the question is explicitly about those documents.

### 6.3 Evidence policies

The classifier now assigns an explicit evidence policy before final arbitration:

| Policy | Meaning |
| --- | --- |
| `local_only` | Use indexed local files as the source of truth. This is used for uploaded-document questions, document summaries, database/library inventory, and direct local facts. |
| `local_plus_model` | Search local first and allow AI knowledge only as stable explanatory support. |
| `local_plus_web` | Search local first and use web evidence for external validation. |
| `local_model_web` | Collect candidates from local files, AI knowledge, and web when web search is enabled, then rank and reconcile them. |
| `web_only` | Search local first, but use web evidence as factual authority because the answer can change. AI knowledge can help wording but is not treated as final evidence. |

The local search step still runs first for real knowledge questions. The policy controls arbitration after evidence collection; it does not disable the local database.

### 6.4 Conflict resolution

If evidence disagrees, Verilume tries to resolve the conflict rather than merging everything blindly.

Examples:

- current-office questions prefer authoritative and recent sources
- identity lookups filter out same-name but wrong-person pages and use entity-match scoring before synthesis
- stale public directory pages are not trusted as current-role evidence

### 6.5 Final synthesis

When web evidence is involved, Verilume builds a verified evidence payload and asks the model to synthesize only from that evidence. The model is instructed not to invent facts or citation labels.

For stable/static questions, final synthesis combines the useful streams that survived validation:

1. Local evidence carries the most weight and wins conflicts.
2. AI knowledge can add definitions, framing, and stable background when it is relevant.
3. Web evidence can add externally sourced support and clickable `[W#]` citations.

When AI knowledge cannot answer or is insufficient, Verilume can still produce a local-only answer from the indexed files. When the question is current or otherwise changeable, Verilume keeps the final answer web-grounded and does not rely on AI knowledge as evidence.

## 7. How Answer Validation Works

Validation happens in two distinct layers.

### 7.1 Citation verification

The citation verifier checks whether the cited labels in the answer actually exist in the retrieved source set.

It ensures that:

- local citations are valid `[S1]`, `[S2]`, and so on
- web citations are valid `[W1]`, `[W2]`, and so on
- missing or unsupported labels are detected

### 7.2 Answer verification against evidence

After citation verification, Verilume compares the answer text against the supporting evidence and produces a verification status such as:

- `verified`
- `partial`
- `unsupported`

This is especially important when the system has to decide whether the final answer is strong enough to trust or whether it should fall back to a more conservative, evidence-only answer.

## 8. How the App Labels Results

The UI labels answers based on which evidence streams actually survived into the final answer.

Typical outcomes are:

| UI label | Meaning |
| --- | --- |
| `Local Retrieval` | The answer is grounded in local files. |
| `AI Knowledge` | The answer came from model knowledge and is not externally verified. |
| `Web Search` | The answer is primarily supported by web evidence. |
| `Hybrid` | The answer used multiple evidence streams together. |
| `Current Information` | The answer is time-sensitive and validated through current evidence. |

The internal diagnostics also track whether the final answer used local evidence, model knowledge, web evidence, or a hybrid of these streams.

## 9. Current Search Decision Rules in Plain English

The practical answer policy is:

1. Search local files first.
2. Greetings, thanks, identity prompts such as "Who are you?", and simple capability prompts do not enter the full RAG flow.
3. If the question is about uploaded local material or a personal document fact such as passport issue date or passport expiry, answer from local files only.
4. If local evidence is strong and the question is stable/non-current, combine it with relevant AI knowledge, but weight local evidence more heavily.
5. If AI knowledge cannot answer or is insufficient, keep the answer local-only when the local files answer the question.
6. If web search is enabled for a stable/static question, run web search and AI knowledge together and synthesize them with any local evidence that was found.
7. Use web-only evidence for current events, live facts, recent roles, population, GDP, prices, laws, schedules, weather, latest papers, and other answers that can change.
8. For bare person-name lookups, discard evidence that is primarily about another named person before the final answer is written.
9. If web search is disabled and local evidence is missing, AI knowledge can answer stable questions by itself.
10. The final answer is selected by ranking the surviving local, model, and web evidence streams rather than blindly trusting whichever stream ran last.
11. Never allow unsupported citations or obviously wrong same-name identity pages to decide the answer.

## 10. How to Install and Run the macOS App Now

There are two practical macOS paths today.

### Option A: simple local launcher

This is the fastest way to run Verilume on macOS from the repository:

```bash
git clone git@github.com:DamingoNdiwa/verilume.git
cd verilume
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python launcher.py
```

You can also double-click `Verilume.command`. That script installs the package in editable mode and launches the app.

### Option B: build a real `.app` bundle

If you want a macOS application bundle:

1. Install `uv`.
2. From the repository root, run:

```bash
python scripts/build_macos_app.py
```

That script installs the `mac` extra and builds a PyInstaller bundle. The result is placed in `dist/`, typically as `dist/Verilume.app`.

After building:

1. Move `Verilume.app` to `Applications` if you want.
2. On first launch, macOS may require right-click then `Open`.
3. Enter your Hugging Face token and optional web provider key in the sidebar.

## 11. How to Install the Python Package Later

### Install from GitHub now

Before the package is published to PyPI, the simplest install path is directly from GitHub:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install "git+https://github.com/DamingoNdiwa/verilume.git"
verilume run
```

If you prefer SSH:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install "git+ssh://git@github.com/DamingoNdiwa/verilume.git"
verilume run
```

The runtime dependency set now includes OCR and PowerPoint support libraries, so the package install also brings in:

- `rapidocr-onnxruntime`
- `pypdfium2`
- `python-pptx`
- `Pillow`

The package metadata now pins `rapidocr-onnxruntime` to the published version line starting at `1.2.3`. Earlier install instructions that implied a `1.4.4+` requirement were incorrect because that version was never published on PyPI.

### Install from PyPI later

Once the package is published to PyPI, installation should become:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install verilume
verilume run
```

## 12. How the Package Will Be Published

The package metadata already lives in `pyproject.toml` and exposes the CLI entrypoint `verilume = verilume.cli:main`.

The normal publish path is:

```bash
rm -rf dist build
find . -name "*.egg-info" -prune -exec rm -rf {} +
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

After publication, GitHub can remain the source repository while PyPI becomes the installation source for end users.

## 13. Useful Commands

```bash
verilume run
verilume ingest
verilume ingest --reset
verilume stats
verilume config
verilume doctor
```

`verilume doctor` is the fastest health check. It reports whether the docs directory exists, whether Chroma exists, whether the manifest exists, whether tokens are configured, and how many chunks are indexed.

## 14. Short Summary

Verilume is implemented as a local-first evidence system, not just a chat wrapper around an LLM.

Its decision flow is:

1. Understand the question.
2. Search local files first, including OCR-backed images, scanned PDFs, and PowerPoint content.
3. Use AI knowledge when local evidence is missing or as corroboration.
4. Use web search when enabled and needed.
5. Rank evidence.
6. Verify citations.
7. Validate the final answer against the evidence.
8. Show the answer with transparent source labeling.
