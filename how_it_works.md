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
| `src/verilume/core/agentic_planner.py` | Produces explicit pipeline actions such as local search, model answer, web search, document summarization, table extraction, and calculation. |
| `src/verilume/core/benchmark.py` | Compares Full, Local Only, AI Only, and Web Only answer strategies with latency, citation counts, confidence, and faithfulness diagnostics. |
| `src/verilume/core/claim_extraction.py` | Extracts atomic factual claims from final answers for evidence comparison. |
| `src/verilume/core/evidence_comparison.py` | Compares each claim against local, web, and AI evidence streams. |
| `src/verilume/core/graphrag.py` | Expands entity/topic questions through the knowledge graph and produces graph-backed local source candidates. |
| `src/verilume/core/knowledge_graph.py` | SQLite knowledge graph for people, organizations, locations, topics, publications, documents, and their mentions/relations. |
| `src/verilume/core/multimodal_store.py` | SQLite store for visual items, OCR text, formula text, captions, bounding boxes, and image paths. |
| `src/verilume/core/multimodal_retrieval.py` | Retrieves visual/OCR evidence for page, figure, formula, plot, and scanned-image questions. |
| `src/verilume/core/figure_captioning.py` | Conservative fallback captioning from existing captions, OCR text, or formula text. |
| `src/verilume/core/entity_filter.py` | Strict short-name/entity filtering so bare person-name queries cannot use unrelated local or web chunks. |
| `src/verilume/core/equation_repair.py` | Conservative repair for obvious OCR/PDF equation extraction errors before local chunking. |
| `src/verilume/core/retrieval.py` | Chroma retriever with dense, lexical, and hybrid retrieval. |
| `src/verilume/core/query_interpreter.py` | Interprets user intent, follow-up context, source policy, and search preferences. |
| `src/verilume/core/search_planner.py` | Produces a search plan describing whether local, model knowledge, or web evidence should be used. |
| `src/verilume/core/search_modes.py` | Canonical search-mode enum and label/parser helpers. |
| `src/verilume/core/search_policy.py` | Strict source-use policy for local, AI, and web decisions based on the selected search mode and current/dynamic status. |
| `src/verilume/core/semantic_cache.py` | Persistent semantic answer cache keyed by question meaning, evidence policy, document fingerprint, web settings, backend, and model. |
| `src/verilume/core/table_store.py` | SQLite-backed table metadata store with local CSV snapshots. |
| `src/verilume/core/table_retrieval.py` | Finds the best local table for numerical questions. |
| `src/verilume/core/table_agent.py` | Performs safe pandas calculations and returns calculation-grounded answers. |
| `src/verilume/rag.py` | Main orchestration layer. It runs local retrieval, AI knowledge generation, web search, reranking, answer selection, citation verification, and final answer validation. |
| `src/verilume/ui/sidebar.py` | Settings, upload controls, remove-document controls, and stats cards. |
| `src/verilume/ui/chat.py` | Chat rendering, source blocks, evidence panels, and exports. |

## 3. Data and Storage Layout

By default, user data is stored in the local user directory:

- `~/.verilume/documents` stores uploaded source files.
- `~/.verilume/chroma_db` stores the Chroma vector database.
- `~/.verilume/ingestion_manifest.json` stores file hashes and ingestion metadata.
- `~/.verilume/semantic_cache.json` stores reusable evidence-ranked answers.
- `~/.verilume/tables` stores table metadata and local CSV snapshots for calculation questions.
- `~/.verilume/knowledge_graph.sqlite` stores local document entities, relations, and mentions.
- `~/.verilume/multimodal.sqlite` stores visual/OCR/formula evidence metadata.
- `~/.verilume/config.env` stores local app settings such as tokens and model choices.

This keeps user content outside the repository and makes the app usable as a desktop tool without polluting the project tree.

## 4. How Ingestion Works

When files are uploaded or `verilume ingest` is run, Verilume does the following:

1. Detect the file type.
2. Route the file to a format-specific handler.
3. Extract text page by page for PDF files, slide by slide for PowerPoint files, OCR image uploads directly, and read full text for DOCX, TXT, Markdown, and CSV files.
4. Normalize extracted text before indexing. For PDFs this includes cleanup for broken icon-font fragments and hyphenated line breaks.
5. Repair only math-looking lines with obvious OCR/PDF equation notation errors.
6. Split the content into chunks using semantic chunking by default.
7. Compute embeddings for each chunk.
8. Store chunk text, metadata, and embeddings in Chroma.

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

### 5.3 Agentic action planning

After query interpretation, Verilume now builds an explicit action plan. This does not replace evidence arbitration; it makes the intended retrieval/tool sequence visible and future-ready.

Supported actions are:

| Action | Meaning |
| --- | --- |
| `search_local` | Search the local Chroma knowledge base. |
| `answer_model` | Ask the configured model for stable explanatory support. |
| `search_web` | Search the configured web provider for external evidence. |
| `summarize_documents` | Build a local-document summary answer. |
| `extract_table` | Locate relevant table data. |
| `calculate` | Compute a numerical result from table data. |
| `build_graph_context` | Reserved for knowledge graph and GraphRAG context. |
| `retrieve_multimodal` | Reserved for image/page-level multimodal retrieval. |

Planner examples:

| Question type | Example actions |
| --- | --- |
| Stable explanation | `search_local`, `answer_model`, and `search_web` when web is enabled. |
| Current or dynamic fact | `search_local`, `search_web`; model knowledge is excluded as factual evidence. |
| Local document summary | `search_local`, `summarize_documents`. |
| Table calculation | `search_local`, `extract_table`, `calculate`. |
| Explicit web request | `search_local`, `search_web`, and `answer_model` for stable support. |

The planner writes `action_plan`, `planner_reason`, `question_type`, `policy`, and `agentic_plan` into response diagnostics. The existing mature routing still executes the current local/model/web pipeline, while later Version 2 features can attach concrete tools to these actions.

### 5.4 Local retrieval first

For real knowledge questions, Verilume searches the local knowledge base first.

Local retrieval uses one of three modes:

1. Dense retrieval with embeddings.
2. Lexical BM25-style retrieval.
3. Hybrid retrieval using reciprocal-rank fusion.

Some question types force lexical retrieval, especially identity lookups and local-file questions, because exact names and filenames matter more than semantic similarity.

### 5.5 AI knowledge usage

AI knowledge is now part of the normal answer path instead of a last-resort add-on.

The current behavior is:

1. If local evidence exists for a stable non-current question, Verilume still asks the AI model for useful background or explanation.
2. If the AI answer is relevant and sufficient, final synthesis combines the AI knowledge with the local evidence.
3. Local file evidence has more weight than AI knowledge. If they disagree, local evidence wins.
4. If AI knowledge cannot answer or is insufficient, Verilume keeps the answer local-only when local evidence answers the question.
5. If web search is disabled and local evidence is missing, Verilume can answer stable non-current questions from AI knowledge alone.

For model-only answers, the app labels the result as AI knowledge and marks it as not externally verified.

### 5.6 Web search usage

Web search is used when:

1. Web search is enabled and the question is a normal stable/static knowledge question.
2. The question explicitly asks for web search.
3. The question is current, time-sensitive, or otherwise likely to change.
4. Local evidence and AI knowledge are both insufficient and web search is enabled.

For stable/static questions, web search is paired with AI knowledge rather than replacing it. This also applies when local evidence already gives a usable answer: the web stream joins the evidence pool when enabled, while local evidence remains weighted more heavily for document-specific facts and conflict resolution. For current or changeable questions, web evidence becomes the source of truth and AI knowledge is not used as final evidence.

Web search is not supposed to override strong, stable local evidence unnecessarily. It supplements or validates when appropriate, and local evidence keeps the highest weight when the question is about uploaded documents.

### 5.7 Local-file exceptions

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

### 5.8 Search modes

The sidebar now exposes a `Search mode` control for users who want more control than Auto routing.

The modes are:

| Mode | Behavior |
| --- | --- |
| `Auto` | Default local-first routing. Local is searched first, AI knowledge is used for stable support, and web is used when enabled, requested, or needed by the evidence policy. |
| `Local Only` | Searches indexed local files and blocks model knowledge and web search for that turn. |
| `Local + AI` | Searches local files and allows AI knowledge, but blocks web search. |
| `Local + AI + Web` | Forces the full hybrid local/model/web path when web search is configured. |
| `Web Only` | Skips local and model evidence and answers from web evidence only when web search is configured. |
| `Research Mode` | Uses the full hybrid path and is intended for source-heavy answers and broader evidence collection. |

Auto remains the default because it preserves the local-first safety model. The explicit modes are user controls; they should not be used by the classifier as hidden defaults.

Search mode enforcement is centralized in `SearchPolicy`. The query interpreter can suggest intent and useful search queries, but it cannot override the selected search mode:

- `Local Only` searches local evidence only.
- `Local + AI` searches local evidence and asks model knowledge, but never searches web.
- `Local + AI + Web` searches exactly local, AI, and web when web is configured; AI is demoted for current/dynamic facts.
- `Web Only` searches web only and blocks local/model factual evidence.
- `Research Mode` searches local, AI, and web with the same current/dynamic AI demotion.
- `Auto` searches local first, combines AI for stable questions, and uses web when enabled or needed.

### 5.9 Semantic answer cache

Verilume now has two answer caches:

1. A short in-memory response cache for immediate repeated questions inside the same app session.
2. A persistent semantic cache stored at `~/.verilume/semantic_cache.json`.

The semantic cache is deliberately evidence-aware. A cached answer can only be reused when all of these match:

- normalized question meaning
- evidence policy
- local document fingerprint
- web enabled/disabled state
- web search provider
- generation backend
- selected generation model

The document fingerprint includes the configured documents directory, Chroma location, collection name, ingestion manifest content, and uploaded document metadata. This means local-document answers are invalidated when files or the manifest change, even if the user asks a similar question later.

Cache freshness follows the same evidence philosophy:

| Question type | Cache behavior |
| --- | --- |
| Stable/static facts | Can be reused for the stable semantic-cache TTL. |
| Person/company/entity lookups | Can be reused for the entity TTL, but only with the same local/web/model context. |
| Current, dynamic, and news questions | Use a short TTL because the answer may change. |
| Local-document questions | Stay valid until the document fingerprint changes. |

The semantic cache stores the final reconciled answer, local citations, web citations, model-answer support text, diagnostics, and evidence scores. On a hit, the RAG layer returns the cached evidence-ranked answer instead of running retrieval and generation again. On a miss or stale entry, Verilume runs the full local/model/web pipeline.

### 5.10 Table-aware retrieval and calculation

For numerical questions over local tabular files, the planner emits `extract_table` and `calculate`. The current Version 2 slice supports CSV and TSV files in the local documents directory.

The table path works like this:

1. Verilume scans `DOCS_DIR` for `.csv` and `.tsv` files.
2. Each table is saved into `TABLE_STORE_DIR/frames` as a clean CSV snapshot.
3. Metadata is stored in `TABLE_STORE_DIR/tables.sqlite3`, including table ID, document, columns, row count, column types, summary, source path, and file signature.
4. A table retriever matches the question against document names, summaries, and column names.
5. The table agent chooses a numeric column and performs a pandas calculation such as `mean`, `sum`, `min`, `max`, `median`, or `count`.
6. The answer includes the calculation performed, the column used, the numeric result, and a local `[S#]` citation.

The safety rule is strict: if no matching local table or numeric column is found, Verilume does not invent a number. It falls back to the normal evidence pipeline instead.

### 5.11 Knowledge graph

Verilume now includes a lightweight SQLite knowledge graph for local-document entities and relationships. It is intentionally local and rule-based in this version; Neo4j or LLM extraction can be added later.

The graph stores:

- entities: people, organizations, locations, publications, topics, documents, datasets, methods, and laws
- relations: authored, coauthored, affiliated_with, mentions, cites, works_at, supervised_by, related_to, published_in, and located_in
- mentions: document/page/chunk snippets showing where an entity appears

Rule-based extraction currently detects:

- capitalized person names
- university/institute/department-style organizations
- DOI/publication markers
- known methods and topics such as Bayesian inference, Bayesian model selection, Hamiltonian Monte Carlo, regression analysis, spectral analysis, hydrology, statistics, and machine learning
- document entities and document-to-entity mention edges
- simple affiliation and supervision patterns

Every edge and mention stores document, page, and chunk metadata when available. The graph can answer structural lookup questions such as which documents mention a person, which topics are linked to a method, or which organizations are connected to a person. GraphRAG uses this in the next layer to improve retrieval before vector search.

### 5.12 GraphRAG

GraphRAG complements vector retrieval with graph context. It does not replace Chroma search.

For entity-heavy or topic-heavy questions, the planner can emit `build_graph_context`. The GraphRAG retriever then:

1. Detects seed entities from the question.
2. Looks those entities up in the local knowledge graph.
3. Expands one hop to neighboring people, organizations, topics, documents, and methods.
4. Collects related documents and chunk IDs from graph mentions.
5. Converts graph mentions into local source candidates.
6. Merges graph candidates with normal vector/lexical local retrieval before reranking and generation.

This helps questions such as:

- Who is Christophe Ley?
- Which documents connect Christophe Ley and Bayesian inference?
- Which topics are connected to Hamiltonian Monte Carlo?
- Which documents mention both Bayesian model selection and hydrology?

GraphRAG is evidence-bound: graph candidates are created from stored mentions, and each mention keeps document/page/chunk metadata. If the graph has no relevant context, Verilume simply continues with the normal retrieval path.

GraphRAG can be disabled with `ENABLE_GRAPHRAG=false`. This is useful for test isolation or for users who want strict vector/lexical retrieval only.

### 5.13 Multimodal retrieval

Verilume now has the storage and retrieval layer for visual evidence. The current slice is conservative and local:

- visual item ID
- document
- page
- bounding box
- caption
- OCR text
- formula text
- image path
- creation time

The fallback captioner never guesses about unseen images. It uses, in order:

1. an existing caption
2. OCR text
3. formula text
4. a clear note that visual content was stored without readable OCR text

The multimodal retriever searches captions, OCR text, formula text, document names, and page references. It can retrieve evidence for questions such as:

- What does the diagram on page 4 show?
- Which figure explains the model architecture?
- Find the plot about regression analysis.
- What formula appears on page 8?
- Which image contains the word Luxembourg?

If no visual/OCR evidence exists, the system should say that instead of guessing.

### 5.14 Benchmark Mode

Benchmark Mode is an opt-in diagnostic switch in the sidebar. It is off by default.

When enabled, a user question is run through four isolated strategies:

| Benchmark strategy | Behavior |
| --- | --- |
| `full` | Uses the normal Verilume evidence policy and the selected search mode, with benchmark and semantic caches disabled for the run. |
| `local_only` | Searches indexed local files only and blocks model/web evidence. |
| `ai_only` | Skips local and web evidence and asks the configured model for a model-knowledge answer. |
| `web_only` | Skips local and model evidence and uses the configured web provider when available. |

Each strategy records:

- answer text
- confidence
- local source count
- web source count
- total source count
- latency
- answer-verification or faithfulness score when available
- diagnostics from the underlying RAG pass

The benchmark report picks a best diagnostic mode using confidence, source count, faithfulness, and latency. The normal Verilume answer remains the main answer. The UI then shows a `Benchmark Results` table and collapsed per-mode answers below the normal answer so the user can compare behavior without losing the answer-first experience.

Benchmark Mode is not meant to replace Auto routing. It is a research and debugging tool for checking whether local, model, web, or full hybrid evidence is producing the strongest answer for a given question.

Each benchmark mode is wrapped independently. If one strategy fails, the benchmark table records that failure and the app still returns the normal answer.

### 5.15 Short entity filtering

Short name-like prompts such as `Rene`, `Christophe Ley`, `Gabriella Vinco`, `Florian Felice`, and `Damian Mingo Ndiwago` use strict entity filtering before evidence ranking.

For these prompts, local candidates must match the queried entity in the document name or source text. Web candidates must match the entity in the title, URL, or snippet. One-word names require an exact word-boundary match. Multi-part names require all or nearly all meaningful name parts.

This prevents unrelated chunks, such as language-test pages or regression examples, from being treated as evidence for a person lookup. If no reliable source matches the exact entity/name, Verilume returns a low-confidence no-match answer instead of blending unrelated evidence.

Lowercase concept prompts such as `photonics` are not treated as person/entity lookups. Natural-language prompts such as `what is photonics` and `The largest country in Europe` remain normal knowledge questions.

### 5.16 Equation repair

Equation repair runs during ingestion before chunking. It only touches lines that already look mathematical, for example lines containing `=` and regression markers such as `price`, `beta`, `b0`, `b1`, `m2`, `age`, `dis`, or `epsilon`.

The safe regression repair currently normalizes the apartment-price equation to:

```text
PRICE = β₀ + β₁M2 + β₂AGE + β₃DIS + ε
```

Ordinary text is not globally rewritten. In particular, the repair layer does not replace normal letters with Greek symbols unless the line already looks like an equation.

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

For web identity evidence, the contamination check reads the source title, URL, and content together. This prevents search-result pollution where a page about one person survives only because the queried name appears somewhere in the snippet. Entity lookup fallbacks also synthesize a short answer from the evidence text or affiliation phrase instead of using a source heading such as "Author Details" as the answer.

Identity web results are deduplicated by normalized title after filtering. For example, two `Google Scholar` results for the same person should count as one supporting source, while an authoritative university or conference profile can still survive as the main answer source.

### 6.3 Claim-level evidence comparison

After the final answer is selected, Verilume extracts sentence-level factual claims from the answer and compares each claim against the surviving evidence streams:

- local document citations
- web citations
- AI model knowledge when it was actually used

The comparison is rule-based in Version 2:

1. Split the answer into atomic factual claims.
2. Ignore vague helper text, greetings, and stylistic filler.
3. Compare each claim with local snippets, web snippets, and the model answer using term overlap, entity overlap, source score, and simple negation checks.
4. Mark each stream as `supports`, `contradicts`, `not_found`, or `unclear`.
5. Assign a decision such as local wins, web wins, local and web agree, AI-only unverified, or unsupported claim.

The current decision policy follows the same arbitration philosophy as the main answer:

| Situation | Claim-level decision |
| --- | --- |
| Local-document question with local support | Local wins. |
| Current or changing fact with web support | Web wins. |
| Local and web both support | Local and web agree. |
| Local supports and AI disagrees | Local wins. |
| Web supports a current fact and AI disagrees | Web wins. |
| Only AI supports | Treat as unverified. |
| No stream supports | Mark as unsupported. |

The Streamlit chat UI shows this in a collapsed `Evidence Comparison` panel below the answer-first summary. Each claim shows Local, Web, and AI status side by side, with a confidence percentage, the best supporting source label, a short snippet, and the final decision.

### 6.4 Evidence policies

The classifier now assigns an explicit evidence policy before final arbitration:

| Policy | Meaning |
| --- | --- |
| `local_only` | Use indexed local files as the source of truth. This is used for uploaded-document questions, document summaries, database/library inventory, and direct local facts. |
| `local_plus_model` | Search local first and allow AI knowledge only as stable explanatory support. |
| `local_plus_web` | Search local first and use web evidence for external validation. |
| `local_model_web` | Collect candidates from local files, AI knowledge, and web when web search is enabled, then rank and reconcile them. |
| `web_only` | Search local first, but use web evidence as factual authority because the answer can change. AI knowledge can help wording but is not treated as final evidence. |

The local search step still runs first for real knowledge questions. The policy controls arbitration after evidence collection; it does not disable the local database.

### 6.5 Conflict resolution

If evidence disagrees, Verilume tries to resolve the conflict rather than merging everything blindly.

Examples:

- current-office questions prefer authoritative and recent sources
- identity lookups filter out same-name but wrong-person pages and use entity-match scoring before synthesis
- stale public directory pages are not trusted as current-role evidence

### 6.6 Final synthesis

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

The UI renders the answer first. Evidence metadata is intentionally placed below the answer so the user does not have to read badges before the conclusion.

The answer layout is:

1. Direct answer text.
2. Evidence Summary with search mode, searched sources, used sources, winner, source type, confidence, source count, agreement/freshness notes, and source-strength bars.
3. Expandable evidence analysis.
4. Local document citations and web source groups.

The source-strength bars are derived from the evidence streams that survived final arbitration:

- Local strength uses the best local retrieval score or local confidence fallback.
- Web strength uses the best web source score or domain/source confidence fallback.
- AI strength is shown only when model knowledge actually contributed.

Typical source labels are:

Typical outcomes are:

| UI label | Meaning |
| --- | --- |
| `Local Retrieval` | The answer is grounded in local files. |
| `AI Knowledge` | The answer came from model knowledge and is not externally verified. |
| `Web Search` | The answer is primarily supported by web evidence. |
| `Hybrid` | The answer used multiple evidence streams together. |
| `Current Information` | The answer is time-sensitive and validated through current evidence. |

The internal diagnostics also track whether the final answer used local evidence, model knowledge, web evidence, or a hybrid of these streams.

The source tree groups web evidence by source type, such as Government, University, Research, News, Social, and general Web sources. Local document citations remain separate and include page/document metadata.

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
12. Benchmark Mode is diagnostic only; it never replaces the normal answer.
13. Search mode is the final source-use authority.

## 10. Roadmap From Current Suggestions

Several suggested improvements are now partially implemented and documented, while others are intentionally roadmap-sized:

| Suggestion | Current status |
| --- | --- |
| Answer before metadata | Implemented. The answer now appears before the Evidence Summary. |
| Search modes | Implemented in settings, sidebar, and RAG routing. |
| Source confidence bars | Implemented in the Evidence Summary for Local, Web, and AI streams. |
| Benchmark mode | Implemented as an opt-in sidebar diagnostic that compares Full, Local Only, AI Only, and Web Only answer strategies. |
| Strict search policy | Implemented through `SearchMode` and `SearchPolicy`; the interpreter cannot override the selected mode. |
| Equation repair | Implemented conservatively for math-looking local text before chunking. |
| Progressive generation stages | Partially implemented through the Streamlit evidence-collection status log. More granular streaming token output is still future work. |
| Entity verification | Implemented for person/company evidence filtering, web title/content/URL contamination checks, and entity-match scoring; still the top retrieval quality priority. |
| Duplicate clustering | Implemented for exact URL/source merging and normalized-title clustering in identity web results. Broader semantic clustering across mirrors is future work. |
| Multi-document summarisation | Current implementation can browse one representative source per indexed document. Persisted document summaries for summary-first retrieval are future work. |
| Two-stage document then chunk retrieval | Future retrieval improvement. |
| Document explorer | Future UI feature. |
| Clickable page-level highlights | Future citation UX feature. |
| Search suggestions/autocomplete | Future UI feature. |
| Preference memory | Future conversation/profile feature. |
| Anonymous latency analytics | Future opt-in diagnostics feature. |

## 11. How to Install and Run the macOS App Now

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

## 12. How to Install the Python Package Later

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

## 13. How the Package Will Be Published

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

## 14. Useful Commands

```bash
verilume run
verilume ingest
verilume ingest --reset
verilume stats
verilume config
verilume doctor
```

`verilume doctor` is the fastest health check. It reports whether the docs directory exists, whether Chroma exists, whether the manifest exists, whether tokens are configured, and how many chunks are indexed.

## 15. Short Summary

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
