"""SQLite store for visual/OCR evidence extracted from local documents."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class VisualItem:
    image_id: str
    document: str
    page: int | None
    bbox: dict[str, float]
    caption: str
    ocr_text: str
    formula_text: str
    image_path: str
    created_at: str


class MultimodalStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_visual_item(
        self,
        *,
        document: str,
        page: int | None = None,
        bbox: dict[str, float] | None = None,
        caption: str = "",
        ocr_text: str = "",
        formula_text: str = "",
        image_path: str = "",
        image_id: str | None = None,
    ) -> VisualItem:
        bbox = bbox or {}
        image_id = image_id or _image_id(document, page, caption, ocr_text, formula_text, image_path)
        item = VisualItem(
            image_id=image_id,
            document=document,
            page=page,
            bbox=bbox,
            caption=caption,
            ocr_text=ocr_text,
            formula_text=formula_text,
            image_path=image_path,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO visual_items(
                    image_id, document, page, bbox_json, caption, ocr_text,
                    formula_text, image_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.image_id,
                    item.document,
                    item.page,
                    json.dumps(item.bbox),
                    item.caption,
                    item.ocr_text,
                    item.formula_text,
                    item.image_path,
                    item.created_at,
                ),
            )
        return item

    def search_visual_items(self, question: str, *, limit: int = 6) -> list[VisualItem]:
        terms = _terms(question)
        page = _page_from_question(question)
        scored: list[tuple[float, VisualItem]] = []
        for item in self.list_visual_items():
            haystack = " ".join([item.document, item.caption, item.ocr_text, item.formula_text])
            item_terms = _terms(haystack)
            score = len(terms & item_terms) / max(1, len(terms))
            if page is not None and item.page == page:
                score += 0.6
            if score > 0:
                scored.append((score, item))
        return [item for _, item in sorted(scored, key=lambda row: row[0], reverse=True)[:limit]]

    def get_visual_items_for_page(self, document: str, page: int) -> list[VisualItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM visual_items WHERE document = ? AND page = ? ORDER BY created_at DESC",
                (document, page),
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    def list_visual_items(self) -> list[VisualItem]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM visual_items ORDER BY created_at DESC").fetchall()
        return [_item_from_row(row) for row in rows]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_items (
                    image_id TEXT PRIMARY KEY,
                    document TEXT,
                    page INTEGER,
                    bbox_json TEXT,
                    caption TEXT,
                    ocr_text TEXT,
                    formula_text TEXT,
                    image_path TEXT,
                    created_at TEXT
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()


def _item_from_row(row: sqlite3.Row) -> VisualItem:
    return VisualItem(
        image_id=str(row["image_id"]),
        document=str(row["document"] or ""),
        page=row["page"],
        bbox=dict(json.loads(row["bbox_json"] or "{}")),
        caption=str(row["caption"] or ""),
        ocr_text=str(row["ocr_text"] or ""),
        formula_text=str(row["formula_text"] or ""),
        image_path=str(row["image_path"] or ""),
        created_at=str(row["created_at"] or ""),
    )


def _image_id(*parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:18]
    return f"visual_{digest}"


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", (text or "").lower())
        if len(token) > 1
    }


def _page_from_question(question: str) -> int | None:
    match = re.search(r"\bpage\s+(\d+)\b", (question or "").lower())
    return int(match.group(1)) if match else None
