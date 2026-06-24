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

1. If local evidence is insufficient and web search is disabled, Verilume uses AI knowledge.
2. If web search is enabled and the question needs broader evidence, Verilume can run AI knowledge and web search in parallel.
3. If local evidence is already sufficient for a non-current answer, Verilume still keeps the answer local-first, but it can still use AI knowledge as corroborating context internally.

For model-only answers, the app labels the result as AI knowledge and marks it as not externally verified.

### 5.5 Web search usage

Web search is used when:

1. The question is explicitly asking for web search.
2. The question is current or time-sensitive.
3. Local evidence does not clearly answer the question and web search is enabled.

Web search is not supposed to override strong, stable local evidence unnecessarily. It supplements or validates when appropriate.

### 5.6 Local-file exceptions

Local-file existence and direct local-document questions are treated specially.

For example:

- “Is this in my local files?”
- “Which document contains my language certificate?”
- “What date is written on this uploaded certificate?”
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

Local and web sources are reranked before final synthesis. The system looks at:

- lexical overlap
- semantic similarity
- exact phrase matches
- identity relevance
- source authority
- source freshness for current questions

### 6.3 Conflict resolution

If evidence disagrees, Verilume tries to resolve the conflict rather than merging everything blindly.

Examples:

- current-office questions prefer authoritative and recent sources
- identity lookups filter out same-name but wrong-person pages
- stale public directory pages are not trusted as current-role evidence

### 6.4 Final synthesis

When web evidence is involved, Verilume builds a verified evidence payload and asks the model to synthesize only from that evidence. The model is instructed not to invent facts or citation labels.

When the answer is local-only, Verilume can still validate the result against the evidence set instead of returning it completely raw.

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
4. If local evidence is strong and the question is not current or otherwise changeable, keep the answer local-first.
5. If local evidence is missing or weak for a stable question, use AI knowledge next.
6. Only after local evidence and AI knowledge are insufficient should web evidence be pulled in for stable questions.
7. If the question is truly current, time-sensitive, explicitly asks for web search, or is the kind of answer that can change over time, Verilume can still use AI knowledge and web evidence together and then reconcile and rank the result.
8. The final answer is still selected by ranking the surviving local, model, and web evidence streams rather than blindly trusting whichever stream ran last.
9. Never allow unsupported citations or obviously wrong same-name identity pages to decide the answer.

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
