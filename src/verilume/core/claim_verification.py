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
    date_grounded: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "supporting_source_ids": self.supporting_source_ids,
            "support_score": round(self.support_score, 4),
            "verdict": self.verdict,
            "date_grounded": self.date_grounded,
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
    source_dates = set().union(*(_calendar_dates(text) for text in source_texts.values())) if source_texts else set()
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
        verdict = _verdict(best_score, cited_labels, supporting_ids)

        # A specific calendar date is only credible if it appears in a source.
        # Term-overlap scoring cannot tell a fabricated date (e.g. one stitched
        # together from unrelated snippets) from a genuinely cited one, so any
        # asserted full date that is absent from every source downgrades the
        # verdict rather than being reported as supported.
        claim_dates = _calendar_dates(claim.text)
        date_grounded = not claim_dates or bool(claim_dates & source_dates)
        if not date_grounded:
            verdict = _downgrade_ungrounded_date(verdict)

        supports.append(
            ClaimSupport(
                claim=claim.text,
                supporting_source_ids=supporting_ids,
                support_score=best_score,
                verdict=verdict,
                date_grounded=date_grounded,
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


_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# "1 October 2025" / "14 Oct 2021"
_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"
)
# "October 1, 2025" / "Oct 1 2025"
_MONTH_DAY_YEAR = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
)
# ISO "2025-10-01"
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")


def _calendar_dates(text: str) -> set[tuple[int, int, int]]:
    """Return normalized ``(year, month, day)`` tuples for full dates in ``text``.

    Only day+month+year dates are captured; bare years or month/year pairs are
    intentionally ignored, so grounding never over-triggers on partial matches.
    """

    if not text:
        return set()
    dates: set[tuple[int, int, int]] = set()
    for day, month_name, year in _DAY_MONTH_YEAR.findall(text):
        _add_date(dates, year, month_name, day)
    for month_name, day, year in _MONTH_DAY_YEAR.findall(text):
        _add_date(dates, year, month_name, day)
    for year, month, day in _ISO_DATE.findall(text):
        _add_numeric_date(dates, year, month, day)
    return dates


def _add_date(dates: set[tuple[int, int, int]], year: str, month_name: str, day: str) -> None:
    month = _MONTHS.get(month_name.lower()[:3])
    if month is None:
        return
    _add_numeric_date(dates, year, str(month), day)


def _add_numeric_date(dates: set[tuple[int, int, int]], year: str, month: str, day: str) -> None:
    try:
        y, m, d = int(year), int(month), int(day)
    except ValueError:
        return
    if 1 <= m <= 12 and 1 <= d <= 31:
        dates.add((y, m, d))


def _downgrade_ungrounded_date(verdict: str) -> str:
    if verdict == "supported":
        return "weakly_supported"
    if verdict == "weakly_supported":
        return "unsupported"
    return verdict


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
