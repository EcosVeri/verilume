# Evidence

Verilume is designed to show why an answer was chosen.

## Citation Rules

Local document citations use:

```text
[S1], [S2], [S3]
```

Web citations use:

```text
[W1], [W2], [W3]
```

Local and web sources are displayed separately.

## Evidence Ranking

Evidence can come from:

- Local indexed files
- Model knowledge
- Optional web search

The system ranks evidence using retrieval strength, lexical match, semantic match, source authority, freshness, and entity relevance.

## Claim Verification

After an answer is drafted, Verilume checks whether factual claims are supported by the surviving local, web, or AI evidence streams.

If evidence is weak, incomplete, or conflicting, the app should make that uncertainty visible instead of hiding it.

## Benchmark Mode

Benchmark mode compares isolated answer routes:

- Full
- Local Only
- AI Only
- Web Only

It records latency, confidence, source counts, diagnostics, and the strongest route for the question.
