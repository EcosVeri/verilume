"""Citation verification for synthesized Verilume answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verilume.core.reranking import query_terms
from verilume.core.schemas import LocalSource, WebSource


@dataclass(slots=True)
class CitationVerification:
    answer: str
    supported: bool
    cited_labels: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class CitationVerificationAgent:
    """Verify that cited labels exist and roughly match the resolved question."""

    def verify(
        self,
        answer: str,
        *,
        question: str,
        local_sources: list[LocalSource],
        web_sources: list[WebSource],
    ) -> CitationVerification:
        answer = answer or ""
        source_labels = {source.label for source in [*local_sources, *web_sources]}
        cited = _cited_labels(answer)
        missing = [label for label in cited if label not in source_labels]
        notes = []
        cleaned_answer = answer
        if missing:
            cleaned_answer = _remove_missing_citations(cleaned_answer, missing)
            notes.append(f"Removed unsupported citation labels: {', '.join(missing)}")

        supported = True
        relevant_labels = _relevant_labels(question, local_sources, web_sources)
        surviving = [label for label in _cited_labels(cleaned_answer) if label in source_labels]
        if surviving and relevant_labels and not any(label in relevant_labels for label in surviving):
            notes.append("Citations exist but weakly match the resolved question.")
            supported = False
        elif source_labels and not surviving:
            notes.append("No verified citation label was present in the answer.")
            supported = False

        return CitationVerification(
            answer=re.sub(r"\s+", " ", cleaned_answer).strip()
            if "\n" not in cleaned_answer
            else cleaned_answer.strip(),
            supported=supported,
            cited_labels=surviving,
            missing_labels=missing,
            notes=notes,
        )


def _cited_labels(answer: str) -> list[str]:
    labels = re.findall(r"\[([SW]\d+)\]", answer or "")
    seen = set()
    values = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            values.append(label)
    return values


def _remove_missing_citations(answer: str, labels: list[str]) -> str:
    cleaned = answer
    for label in labels:
        cleaned = re.sub(rf"\s*\[{re.escape(label)}\]", "", cleaned)
    return cleaned


def _relevant_labels(
    question: str,
    local_sources: list[LocalSource],
    web_sources: list[WebSource],
) -> set[str]:
    terms = set(query_terms(question))
    if not terms:
        return {source.label for source in [*local_sources, *web_sources]}
    labels: set[str] = set()
    for source in local_sources:
        text = f"{source.document} {source.text}"
        if terms & set(query_terms(text)):
            labels.add(source.label)
    for source in web_sources:
        text = f"{source.title} {source.content} {source.url}"
        if terms & set(query_terms(text)):
            labels.add(source.label)
    return labels
