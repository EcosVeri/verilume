"""Formula-first retrieval for mathematical questions."""

from __future__ import annotations

import re

from verilume.core.formula_extraction import formula_to_text
from verilume.core.formula_store import FormulaStore
from verilume.core.schemas import LocalSource

FORMULA_QUERY_RE = re.compile(
    r"\b(?:formula|equation|notation|symbol|variable|coefficient|derive|derivation|"
    r"likelihood|posterior|prior|density|distribution|expectation|variance|gradient|"
    r"matrix|model|what does|find formula|explain equation)\b",
    re.IGNORECASE,
)


def is_formula_query(question: str) -> bool:
    text = question or ""
    if FORMULA_QUERY_RE.search(text):
        return True
    if re.search(r"\\(?:alpha|beta|theta|lambda|sigma|sum|int|sqrt)|[α-ωΑ-Ω∑Σ∫≈∝≤≥]", text):
        return True
    return bool(re.search(r"\b[a-zA-Z]\s*[_^]\s*[0-9A-Za-z]+\b|[=≈∝]", text))


class FormulaRetriever:
    def __init__(self, store: FormulaStore) -> None:
        self.store = store

    def retrieve(self, question: str, *, limit: int = 5) -> list[LocalSource]:
        if not is_formula_query(question):
            return []
        sources: list[LocalSource] = []
        for index, item in enumerate(self.store.search(question, limit=limit), start=1):
            metadata = dict(item.metadata or {})
            metadata.update(
                {
                    "content_type": "formula",
                    "formula_id": item.formula_id,
                    "formula_type": item.formula_type or "unknown",
                    "formula_confidence": item.confidence,
                    "formula_variables": item.variables,
                    "raw_formula": item.raw_text,
                    "repaired_formula": item.repaired_text,
                    "surrounding_text": item.surrounding_text,
                    "retrieval": "formula",
                }
            )
            sources.append(
                LocalSource(
                    label=f"S{index}",
                    document=item.document,
                    page=item.page,
                    chunk_id=f"formula:{item.formula_id}",
                    text=formula_to_text(item),
                    score=max(0.0, min(1.0, item.confidence or 0.0)),
                    metadata=metadata,
                )
            )
        return sources
