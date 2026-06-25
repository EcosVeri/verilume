"""SQLite store for generic structured-document OCR fields."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from verilume.core.structured_ocr import StructuredDocument, StructuredField


class StructuredDocumentStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_structured_document(self, document: StructuredDocument) -> None:
        if not document.fields:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO structured_documents (
                    id, document, page, document_type, confidence, fields_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    document.document_id,
                    document.document,
                    document.page,
                    document.document_type,
                    document.confidence,
                    json.dumps([_field_to_dict(field) for field in document.fields], sort_keys=True),
                ),
            )
            conn.execute("DELETE FROM structured_fields WHERE structured_document_id = ?", (document.document_id,))
            conn.executemany(
                """
                INSERT OR REPLACE INTO structured_fields (
                    id, structured_document_id, document, page, canonical_name, raw_label,
                    value, raw_value, field_type, confidence, bbox_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"{document.document_id}:{index}",
                        document.document_id,
                        document.document,
                        field.page,
                        field.canonical_name,
                        field.raw_label,
                        field.value,
                        field.raw_value,
                        field.field_type,
                        field.confidence,
                        json.dumps(field.bbox) if field.bbox else None,
                    )
                    for index, field in enumerate(document.fields, start=1)
                ],
            )

    def delete_document(self, document: str) -> None:
        with self._connect() as conn:
            ids = [
                str(row["id"])
                for row in conn.execute("SELECT id FROM structured_documents WHERE document = ?", (document,)).fetchall()
            ]
            conn.execute("DELETE FROM structured_documents WHERE document = ?", (document,))
            for doc_id in ids:
                conn.execute("DELETE FROM structured_fields WHERE structured_document_id = ?", (doc_id,))

    def search_fields(
        self,
        query: str,
        *,
        canonical_name: str | None = None,
        field_type: str | None = None,
        document_type: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        terms = _terms(query)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.*, d.document_type
                FROM structured_fields f
                LEFT JOIN structured_documents d ON f.structured_document_id = d.id
                """
            ).fetchall()
        scored: list[tuple[dict, float]] = []
        for row in rows:
            item = dict(row)
            if canonical_name and item.get("canonical_name") != canonical_name:
                continue
            if field_type and item.get("field_type") != field_type:
                continue
            if document_type and item.get("document_type") != document_type:
                continue
            haystack = " ".join(str(item.get(key) or "") for key in ("canonical_name", "raw_label", "value", "field_type", "document", "document_type"))
            overlap = len(terms & _terms(haystack))
            score = overlap + float(item.get("confidence") or 0.0)
            if canonical_name or field_type:
                score += 1.0
            if overlap or canonical_name or field_type:
                scored.append((item, score))
        return [item for item, _score in sorted(scored, key=lambda pair: pair[1], reverse=True)[:limit]]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS structured_documents (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    page INTEGER,
                    document_type TEXT,
                    confidence REAL,
                    fields_json TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS structured_fields (
                    id TEXT PRIMARY KEY,
                    structured_document_id TEXT,
                    document TEXT NOT NULL,
                    page INTEGER,
                    canonical_name TEXT,
                    raw_label TEXT,
                    value TEXT,
                    raw_value TEXT,
                    field_type TEXT,
                    confidence REAL,
                    bbox_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_structured_fields_document ON structured_fields(document)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_structured_fields_canonical ON structured_fields(canonical_name)")


def _field_to_dict(field: StructuredField) -> dict:
    return {
        "field_name": field.field_name,
        "canonical_name": field.canonical_name,
        "value": field.value,
        "raw_label": field.raw_label,
        "raw_value": field.raw_value,
        "field_type": field.field_type,
        "confidence": field.confidence,
        "page": field.page,
        "bbox": field.bbox,
    }


def _terms(text: str) -> set[str]:
    return {term.lower() for term in __import__("re").findall(r"[A-Za-z0-9]{3,}", text or "")}
