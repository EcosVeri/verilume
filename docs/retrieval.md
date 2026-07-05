# Retrieval

Verilume uses a local-first evidence strategy.

## Local Retrieval

Local documents are indexed into Chroma. Retrieval can use:

- Dense embedding search
- BM25-style lexical search
- Hybrid retrieval with reciprocal-rank fusion
- Reranking when available

## Search Modes

- `Auto`: local-first routing with AI and web support when useful
- `Local Only`: local indexed files only
- `Local + AI`: local files plus model knowledge
- `Local + AI + Web`: local, model, and web evidence
- `Web Only`: web evidence only when configured
- `Research Mode`: source-heavy hybrid search

## Query Handling

The query layer decides whether a prompt is a local-document question, stable knowledge question, current-information question, table question, person lookup, or web request.

For local-file questions, Verilume keeps the answer grounded in indexed files and does not invent local facts from model knowledge or web search.

## Caching

The semantic cache can reuse evidence-ranked answers when the question meaning, evidence policy, local document fingerprint, web settings, backend, and model are still compatible.
