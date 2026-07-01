"""SQLite store for extracted formula evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from verilume.core.formula_extraction import FormulaItem


class FormulaStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_formula(self, item: FormulaItem) -> None:
        self.add_many([item])

    def add_many(self, items: Iterable[FormulaItem]) -> None:
        values = list(items)
        if not values:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO formulas (
                    formula_id, document, page, raw_text, repaired_text, latex,
                    surrounding_text, variables_json, formula_type, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.formula_id,
                        item.document,
                        item.page,
                        item.raw_text,
                        item.repaired_text,
                        item.latex,
                        item.surrounding_text,
                        json.dumps(item.variables, sort_keys=True),
                        item.formula_type or "unknown",
                        float(item.confidence or 0.0),
                        json.dumps(item.metadata or {}, sort_keys=True),
                    )
                    for item in values
                ],
            )

    def delete_document(self, document: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM formulas WHERE document = ?", (document,))

    def search(self, query: str, *, limit: int = 5) -> list[FormulaItem]:
        items = self.all()
        query_terms = _terms(query)
        scored: list[tuple[FormulaItem, float]] = []
        for item in items:
            haystack = " ".join(
                [
                    item.document,
                    item.raw_text,
                    item.repaired_text,
                    item.surrounding_text,
                    item.formula_type or "",
                    " ".join(item.variables),
                    " ".join(item.variables.values()),
                ]
            )
            item_terms = _terms(haystack)
            overlap = len(query_terms & item_terms)
            if not overlap and not _symbol_overlap(query, item.repaired_text):
                continue
            score = overlap + float(item.confidence or 0.0)
            if _symbol_overlap(query, item.repaired_text):
                score += 2.0
            scored.append((item, score))
        return [item for item, _score in sorted(scored, key=lambda pair: pair[1], reverse=True)[:limit]]

    def all(self) -> list[FormulaItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT formula_id, document, page, raw_text, repaired_text, latex,
                       surrounding_text, variables_json, formula_type, confidence, metadata_json
                FROM formulas
                ORDER BY document, page, formula_id
                """
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS formulas (
                    formula_id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    page INTEGER,
                    raw_text TEXT,
                    repaired_text TEXT,
                    latex TEXT,
                    surrounding_text TEXT,
                    variables_json TEXT,
                    formula_type TEXT,
                    confidence REAL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_formulas_document ON formulas(document)")


def _item_from_row(row: sqlite3.Row) -> FormulaItem:
    return FormulaItem(
        formula_id=str(row["formula_id"]),
        document=str(row["document"]),
        page=int(row["page"]) if row["page"] is not None else None,
        raw_text=str(row["raw_text"] or ""),
        repaired_text=str(row["repaired_text"] or ""),
        latex=str(row["latex"]) if row["latex"] else None,
        surrounding_text=str(row["surrounding_text"] or ""),
        variables=json.loads(row["variables_json"] or "{}"),
        formula_type=str(row["formula_type"] or "unknown"),
        confidence=float(row["confidence"] or 0.0),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _terms(text: str) -> set[str]:
    return {term.lower() for term in __import__("re").findall(r"[A-Za-z0-9α-ωΑ-Ω_₀-₉²³]{2,}", text or "")}


def _symbol_overlap(query: str, formula: str) -> bool:
    query_symbols = {char for char in query or "" if not char.isalnum() and not char.isspace()}
    formula_symbols = {char for char in formula or "" if not char.isalnum() and not char.isspace()}
    return bool(query_symbols & formula_symbols)
