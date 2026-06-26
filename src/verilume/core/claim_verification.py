"""Lightweight claim-to-source support verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from verilume.core.claim_extraction import extract_claims
from verilume.core.schemas import LocalSource, WebSource


@dataclass(frozen=True, slots=True)
class ClaimSupport:
    claim: str
    supporting_source_ids: list[str]
    support_score: float
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "supporting_source_ids": self.supporting_source_ids,
            "support_score": round(self.support_score, 4),
            "verdict": self.verdict,
        }


def verify_claim_support(
    answer: str,
    *,
    local_sources: Sequence[LocalSource],
    web_sources: Sequence[WebSource],
) -> list[ClaimSupport]:
    claims = extract_claims(answer)
    if not claims:
        return []

    source_texts = _source_texts(local_sources, web_sources)
    supports: list[ClaimSupport] = []
    for claim in claims:
        claim_terms = _terms(claim.text)
        cited_labels = _cited_labels(claim.text)
        scored_sources: list[tuple[str, float]] = []
        for label, text in source_texts.items():
            score = _support_score(claim_terms, text)
            if label in cited_labels:
                score += 0.18
            if score > 0:
                scored_sources.append((label, min(1.0, score)))
        scored_sources.sort(key=lambda item: item[1], reverse=True)
        best_score = scored_sources[0][1] if scored_sources else 0.0
        supporting_ids = [label for label, score in scored_sources[:3] if score >= 0.18]
        supports.append(
            ClaimSupport(
                claim=claim.text,
                supporting_source_ids=supporting_ids,
                support_score=best_score,
                verdict=_verdict(best_score, cited_labels, supporting_ids),
            )
        )
    return supports


def _source_texts(
    local_sources: Sequence[LocalSource],
    web_sources: Sequence[WebSource],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for source in local_sources:
        label = str(source.label or "").strip()
        if label:
            values[label] = " ".join(
                [
                    str(source.document or ""),
                    str(source.text or ""),
                    str(source.metadata or {}),
                ]
            )
    for source in web_sources:
        label = str(source.label or "").strip()
        if label:
            values[label] = " ".join(
                [
                    str(source.title or ""),
                    str(source.url or ""),
                    str(source.content or ""),
                    str(source.metadata or {}),
                ]
            )
    return values


def _support_score(claim_terms: set[str], source_text: str) -> float:
    if not claim_terms:
        return 0.0
    source_terms = _terms(source_text)
    if not source_terms:
        return 0.0
    overlap = len(claim_terms & source_terms)
    coverage = overlap / max(1, len(claim_terms))
    density = overlap / max(1, min(len(source_terms), len(claim_terms) * 4))
    return min(1.0, coverage * 0.78 + density * 0.22)


def _verdict(score: float, cited_labels: set[str], supporting_ids: Sequence[str]) -> str:
    if score >= 0.42 and supporting_ids:
        return "supported"
    if score >= 0.22 and supporting_ids:
        return "weakly_supported"
    if cited_labels and supporting_ids:
        return "weakly_supported"
    return "unsupported"


def _cited_labels(text: str) -> set[str]:
    return {match.group(1) for match in re.finditer(r"\[([SW]\d+)\]", text or "")}


def _terms(text: str) -> set[str]:
    stopwords = {
        "about",
        "also",
        "and",
        "are",
        "confidence",
        "document",
        "documents",
        "file",
        "from",
        "has",
        "have",
        "indexed",
        "into",
        "local",
        "page",
        "pages",
        "source",
        "summary",
        "that",
        "the",
        "this",
        "used",
        "with",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (text or "").lower())
        if term not in stopwords
    }
