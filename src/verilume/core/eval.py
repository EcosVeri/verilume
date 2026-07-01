"""Golden-set retrieval evaluation.

Scoring here is deliberately independent of the embedding/vector stack: it takes
a ``search_fn(query, k) -> sources`` callable, so the metric maths can be unit
tested with a fake retriever, while the CLI wires in a real Chroma retriever
built from a fixture corpus.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

# A retrieved source only needs a document name and (optionally) a page for
# scoring, so we accept anything exposing `.document` / `.page` attributes.
SearchFn = Callable[[str, int], Sequence[Any]]


@dataclass
class GoldQuestion:
    question: str
    expected_documents: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)
    should_find: bool = True


@dataclass
class QuestionResult:
    question: str
    retrieved: list[tuple[str, int | None]]
    hit_at: dict[int, bool]
    page_hit: bool | None
    correct_not_found: bool | None
    latency_seconds: float


@dataclass
class RetrievalMetrics:
    questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    page_accuracy: float
    not_found_accuracy: float
    mean_latency_seconds: float


@dataclass
class EvalReport:
    metrics: RetrievalMetrics
    results: list[QuestionResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": asdict(self.metrics),
            "results": [asdict(result) for result in self.results],
        }


def load_gold_questions(path: str | Path) -> list[GoldQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    return [
        GoldQuestion(
            question=str(item["question"]),
            expected_documents=[str(doc) for doc in item.get("expected_documents", [])],
            expected_pages=[int(page) for page in item.get("expected_pages", [])],
            should_find=bool(item.get("should_find", True)),
        )
        for item in items
    ]


def _doc_matches(expected: str, actual: str) -> bool:
    expected_norm = Path(expected).name.lower()
    actual_norm = Path(actual or "").name.lower()
    return bool(expected_norm) and (expected_norm in actual_norm or actual_norm in expected_norm)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    search_fn: SearchFn,
    gold: Sequence[GoldQuestion],
    ks: tuple[int, ...] = (1, 3, 5),
) -> EvalReport:
    max_k = max(ks) if ks else 5
    results: list[QuestionResult] = []

    for question in gold:
        started = time.perf_counter()
        sources = list(search_fn(question.question, max_k))
        latency = time.perf_counter() - started
        retrieved = [(str(getattr(s, "document", "")), getattr(s, "page", None)) for s in sources]

        hit_at: dict[int, bool] = {}
        for k in ks:
            top_docs = [doc for doc, _page in retrieved[:k]]
            hit_at[k] = any(
                _doc_matches(expected, actual)
                for expected in question.expected_documents
                for actual in top_docs
            )

        page_hit: bool | None = None
        if question.expected_pages and question.expected_documents:
            page_hit = any(
                _doc_matches(expected_doc, doc) and page in question.expected_pages
                for doc, page in retrieved
                for expected_doc in question.expected_documents
            )

        correct_not_found: bool | None = None
        if not question.should_find:
            correct_not_found = len(sources) == 0

        results.append(
            QuestionResult(
                question=question.question,
                retrieved=retrieved,
                hit_at=hit_at,
                page_hit=page_hit,
                correct_not_found=correct_not_found,
                latency_seconds=latency,
            )
        )

    findable = [r for r, q in zip(results, gold) if q.should_find]
    page_scored = [r for r in results if r.page_hit is not None]
    not_found_scored = [r for r in results if r.correct_not_found is not None]

    metrics = RetrievalMetrics(
        questions=len(results),
        hit_at_1=_mean([1.0 if r.hit_at.get(1) else 0.0 for r in findable]),
        hit_at_3=_mean([1.0 if r.hit_at.get(3) else 0.0 for r in findable]),
        hit_at_5=_mean([1.0 if r.hit_at.get(5) else 0.0 for r in findable]),
        page_accuracy=_mean([1.0 if r.page_hit else 0.0 for r in page_scored]),
        not_found_accuracy=_mean([1.0 if r.correct_not_found else 0.0 for r in not_found_scored]),
        mean_latency_seconds=_mean([r.latency_seconds for r in results]),
    )
    return EvalReport(metrics=metrics, results=results)
