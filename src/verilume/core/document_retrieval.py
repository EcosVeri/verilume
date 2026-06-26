"""Filename-aware document-level retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from verilume.core.document_index import IndexedDocument
from verilume.core.schemas import LocalSource

_KNOWN_EXTENSIONS = (
    "pdf",
    "doc",
    "docx",
    "pptx",
    "pptm",
    "ppsx",
    "potx",
    "txt",
    "md",
    "markdown",
    "csv",
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "gif",
    "tif",
    "tiff",
    "webp",
)
_EXPLICIT_FILE_RE = re.compile(
    r"\b[a-z0-9][a-z0-9._ -]*\.(?:" + "|".join(_KNOWN_EXTENSIONS) + r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DocumentMatch:
    document: IndexedDocument
    score: float
    reason: str
    requested_name: str


@dataclass(slots=True)
class DocumentMatchResult:
    matches: list[DocumentMatch]
    ambiguous: bool = False

    @property
    def best(self) -> DocumentMatch | None:
        return self.matches[0] if self.matches else None


def detect_requested_document(
    question: str,
    indexed_documents: Sequence[IndexedDocument],
) -> DocumentMatchResult:
    """Detect explicit or implied filename requests against indexed documents."""
    requested_names = requested_document_names(question)
    if not requested_names:
        return DocumentMatchResult([])

    matches: list[DocumentMatch] = []
    for requested_name in requested_names:
        best_for_name = [
            match
            for match in (
                _score_document_name(requested_name, document) for document in indexed_documents
            )
            if match.score >= 0.55
        ]
        matches.extend(best_for_name)

    deduped = _dedupe_matches(matches)
    deduped.sort(key=lambda item: item.score, reverse=True)
    return DocumentMatchResult(deduped, ambiguous=_matches_are_ambiguous(deduped))


def rank_documents(
    query: str,
    indexed_documents: Sequence[IndexedDocument],
    *,
    limit: int = 8,
) -> list[DocumentMatch]:
    """Rank document metadata for summary and corpus-overview requests."""
    requested = detect_requested_document(query, indexed_documents)
    if requested.matches:
        return requested.matches[: max(1, limit)]

    query_terms = _tokens(query)
    ranked: list[DocumentMatch] = []
    for document in indexed_documents:
        haystack = " ".join(
            [
                document.filename,
                document.title,
                document.summary,
                " ".join(document.keywords),
                document.document_type,
                document.authors,
            ]
        )
        haystack_terms = set(_tokens(haystack))
        overlap = len(set(query_terms) & haystack_terms)
        score = min(1.0, 0.25 + overlap * 0.12)
        if overlap:
            ranked.append(DocumentMatch(document, score, "metadata term match", query))
    ranked.sort(key=lambda item: (item.score, item.document.chunk_count), reverse=True)
    return ranked[: max(1, limit)]


def document_matches_to_sources(matches: Sequence[DocumentMatch]) -> list[LocalSource]:
    """Convert document matches into local sources suitable for RAG presentation."""
    sources: list[LocalSource] = []
    for index, match in enumerate(matches, start=1):
        document = match.document
        text = document_source_text(document)
        sources.append(
            LocalSource(
                label=f"S{index}",
                document=document.filename,
                page=None,
                chunk_id=f"document-summary:{document.document_id}",
                text=text,
                score=match.score,
                metadata={
                    "document_summary": True,
                    "document_match_score": match.score,
                    "document_match_reason": match.reason,
                    "requested_document": match.requested_name,
                    "document_title": document.title,
                    "document_level_summary": document.summary,
                    "keywords": ", ".join(document.keywords),
                    "document_keywords": document.keywords,
                    "document_pages": document.page_count,
                    "document_chunks": document.chunk_count,
                    "document_kind": document.document_type,
                    "authors": document.authors,
                    "source_path": document.source_path,
                    "retrieval": "document-summary",
                },
            )
        )
    return sources


def document_source_text(document: IndexedDocument) -> str:
    keywords = ", ".join(document.keywords[:12])
    parts = [
        f"Document: {document.title or document.filename}",
        f"File: {document.filename}",
        f"Summary: {document.summary}",
    ]
    if keywords:
        parts.append(f"Keywords: {keywords}")
    indexed_bits = []
    if document.page_count:
        indexed_bits.append(f"{document.page_count} pages")
    if document.chunk_count:
        indexed_bits.append(f"{document.chunk_count} chunks")
    if indexed_bits:
        parts.append("Indexed as: " + ", ".join(indexed_bits))
    if document.document_type:
        parts.append(f"Document type: {document.document_type}")
    return "\n".join(part for part in parts if part.split(":", maxsplit=1)[-1].strip())


def requested_document_names(question: str) -> tuple[str, ...]:
    explicit = [
        re.sub(r"\s+", " ", match.group(0)).strip()
        for match in _EXPLICIT_FILE_RE.finditer(question or "")
    ]
    if explicit:
        return tuple(dict.fromkeys(explicit))

    normalized = (question or "").strip()
    implied: list[str] = []
    patterns = (
        r"\b(?:summari[sz]e|summary of|describe|what is in|what's in|tell me about)\s+([a-z0-9][a-z0-9._ -]{2,})\b",
        r"\b(?:file|document)\s+([a-z0-9][a-z0-9._ -]{2,})\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            candidate = match.group(1).strip(" .,:;?!")
            if candidate and not _looks_like_generic_target(candidate):
                implied.append(candidate)
    return tuple(dict.fromkeys(implied))


def _score_document_name(requested_name: str, document: IndexedDocument) -> DocumentMatch:
    requested_key = _filename_key(requested_name)
    requested_stem = _stem_key(requested_name)
    filename_key = _filename_key(document.filename)
    filename_stem = _stem_key(document.filename)
    title_key = _filename_key(document.title)

    score = 0.0
    reason = "filename fuzzy match"
    if requested_key == filename_key:
        score = 1.0
        reason = "exact filename match"
    elif requested_stem and requested_stem == filename_stem:
        score = 0.96
        reason = "extension-insensitive filename match"
    elif requested_key and (requested_key in filename_key or filename_key in requested_key):
        score = 0.86
        reason = "partial filename match"
    elif requested_stem and (requested_stem in filename_stem or filename_stem in requested_stem):
        score = 0.82
        reason = "partial filename match"
    elif requested_key and title_key and (requested_key in title_key or requested_stem in title_key):
        score = 0.72
        reason = "title filename match"
    else:
        ratio = max(
            SequenceMatcher(None, requested_key, filename_key).ratio(),
            SequenceMatcher(None, requested_stem, filename_stem).ratio(),
        )
        score = ratio * 0.78
    return DocumentMatch(document=document, score=score, reason=reason, requested_name=requested_name)


def _dedupe_matches(matches: Sequence[DocumentMatch]) -> list[DocumentMatch]:
    best_by_id: dict[str, DocumentMatch] = {}
    for match in matches:
        key = match.document.document_id or match.document.filename
        previous = best_by_id.get(key)
        if previous is None or match.score > previous.score:
            best_by_id[key] = match
    return list(best_by_id.values())


def _matches_are_ambiguous(matches: Sequence[DocumentMatch]) -> bool:
    if len(matches) < 2:
        return False
    best, second = matches[0], matches[1]
    requested = (best.requested_name or "").strip()
    if "." not in requested and len(_stem_key(requested)) <= 4 and second.score >= 0.8:
        return True
    if best.score >= 0.95 and second.score < 0.95:
        return False
    return second.score >= 0.72 and (best.score - second.score) <= 0.08


def _filename_key(value: str) -> str:
    name = Path(str(value or "").strip().lower()).name
    return re.sub(r"[^a-z0-9.]+", "", name)


def _stem_key(value: str) -> str:
    key = _filename_key(value)
    return Path(key).stem if "." in key else key


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (value or "").lower())


def _looks_like_generic_target(value: str) -> bool:
    return value.strip().lower() in {
        "all files",
        "all documents",
        "current document",
        "current file",
        "documents",
        "files",
        "indexed document",
        "indexed documents",
        "indexed file",
        "indexed files",
        "local document",
        "local documents",
        "local file",
        "local files",
        "my document",
        "my documents",
        "my file",
        "my files",
        "that document",
        "that file",
        "the document",
        "the documents",
        "the file",
        "the files",
        "this document",
        "this file",
        "uploaded document",
        "uploaded documents",
        "uploaded file",
        "uploaded files",
    }
