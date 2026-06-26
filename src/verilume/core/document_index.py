"""Document-level index built from Verilume's ingestion manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from verilume.core.schemas import DocumentMetadata


@dataclass(slots=True)
class IndexedDocument:
    document_id: str
    filename: str
    title: str
    page_count: int
    chunk_count: int
    document_type: str
    summary: str
    keywords: list[str]
    first_page_text: str = ""
    last_updated: str = ""
    source_path: str = ""
    authors: str = ""


def build_document_index(documents: Sequence[DocumentMetadata]) -> list[IndexedDocument]:
    """Convert manifest metadata into document-level index records."""
    indexed: list[IndexedDocument] = []
    for metadata in documents:
        filename = Path(str(metadata.document or metadata.source_path)).name
        if not filename:
            continue
        indexed.append(
            IndexedDocument(
                document_id=_document_id(metadata),
                filename=filename,
                title=str(metadata.title or Path(filename).stem),
                page_count=max(0, int(metadata.pages or 0)),
                chunk_count=max(0, int(metadata.chunks or 0)),
                document_type=str(metadata.document_kind or "document"),
                summary=str(metadata.summary or ""),
                keywords=[str(keyword) for keyword in metadata.keywords if str(keyword).strip()],
                first_page_text="",
                last_updated=_source_mtime(metadata.source_path),
                source_path=str(metadata.source_path or ""),
                authors=str(metadata.authors or ""),
            )
        )
    return indexed


def iter_document_names(documents: Iterable[IndexedDocument]) -> list[str]:
    return [document.filename for document in documents]


def _document_id(metadata: DocumentMetadata) -> str:
    source_path = str(metadata.source_path or "").strip()
    if source_path:
        return source_path
    return Path(str(metadata.document or "")).name


def _source_mtime(source_path: str) -> str:
    path = Path(str(source_path or ""))
    try:
        return str(path.stat().st_mtime) if path.exists() else ""
    except OSError:
        return ""
