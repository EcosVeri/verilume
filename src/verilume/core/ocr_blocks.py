"""SQLite storage for OCR text blocks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class OCRBlock:
    block_id: str
    document: str
    page: int
    text: str
    bbox: tuple[int, int, int, int] | None = None
    confidence: float | None = None
    block_type: str | None = "page_text"


class OCRBlockStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_many(self, blocks: Iterable[OCRBlock]) -> None:
        values = list(blocks)
        if not values:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO ocr_blocks (
                    block_id, document, page, text, bbox_json, confidence, block_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        block.block_id,
                        block.document,
                        block.page,
                        block.text,
                        json.dumps(block.bbox) if block.bbox else None,
                        block.confidence,
                        block.block_type,
                    )
                    for block in values
                ],
            )

    def delete_document(self, document: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ocr_blocks WHERE document = ?", (document,))

    def search(self, query: str, *, limit: int = 5) -> list[OCRBlock]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        blocks = self.all()
        scored = []
        for block in blocks:
            overlap = len(query_terms & _terms(block.text))
            if overlap:
                scored.append((block, overlap))
        return [block for block, _score in sorted(scored, key=lambda pair: pair[1], reverse=True)[:limit]]

    def all(self) -> list[OCRBlock]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT block_id, document, page, text, bbox_json, confidence, block_type FROM ocr_blocks"
            ).fetchall()
        return [_block_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_blocks (
                    block_id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    page INTEGER,
                    text TEXT,
                    bbox_json TEXT,
                    confidence REAL,
                    block_type TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_blocks_document ON ocr_blocks(document)")


def page_text_block(document: str, page: int | None, text: str, *, confidence: float | None = None) -> OCRBlock:
    page_number = int(page or 0)
    digest = hashlib.blake2b(f"{document}:{page_number}:{text[:120]}".encode("utf-8"), digest_size=10)
    return OCRBlock(
        block_id=digest.hexdigest(),
        document=document,
        page=page_number,
        text=text,
        confidence=confidence,
        block_type="page_text",
    )


def _block_from_row(row: sqlite3.Row) -> OCRBlock:
    bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
    return OCRBlock(
        block_id=str(row["block_id"]),
        document=str(row["document"]),
        page=int(row["page"] or 0),
        text=str(row["text"] or ""),
        bbox=tuple(bbox) if bbox else None,
        confidence=float(row["confidence"]) if row["confidence"] is not None else None,
        block_type=str(row["block_type"] or "page_text"),
    )


def _terms(text: str) -> set[str]:
    return {term.lower() for term in __import__("re").findall(r"[A-Za-z0-9]{3,}", text or "")}
