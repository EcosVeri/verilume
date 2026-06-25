"""Shared dataclasses used by ingestion, retrieval, and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LocalSource:
    label: str
    document: str
    page: int | None
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WebSource:
    label: str
    title: str
    url: str
    content: str
    score: float | None = None
    published_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class DocumentMetadata:
    document: str
    title: str
    summary: str
    keywords: list[str]
    pages: int
    chunks: int
    source_path: str = ""
    authors: str = ""
    document_kind: str = "document"


@dataclass(slots=True)
class RAGResponse:
    answer: str
    local_sources: list[LocalSource]
    web_sources: list[WebSource]
    used_web: bool
    confidence: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    conversation_state: Any | None = None
    resolved_query: str | None = None
    original_query: str | None = None


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_path: Path
    document: str
    page: int | None
    chunk_index: int
    file_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestResult:
    files_seen: int
    files_indexed: int
    files_skipped: int
    chunks_indexed: int
    pdf_pages: int
    errors: list[str] = field(default_factory=list)
