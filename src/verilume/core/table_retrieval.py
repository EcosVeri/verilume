"""Table retrieval helpers for numerical local questions."""

from __future__ import annotations

from dataclasses import dataclass

from verilume.core.table_store import TableMetadata, TableStore


@dataclass(frozen=True, slots=True)
class TableMatch:
    metadata: TableMetadata
    score: float


class TableRetrieval:
    def __init__(self, store: TableStore) -> None:
        self.store = store

    def find_best_table(self, question: str) -> TableMetadata | None:
        matches = self.search(question)
        return matches[0].metadata if matches else None

    def search(self, question: str) -> list[TableMatch]:
        tables = self.store.search_tables(question)
        if not tables:
            return []
        question_terms = _terms(question)
        matches: list[TableMatch] = []
        for table in tables:
            haystack_terms = _terms(" ".join([table.document, table.summary, *table.columns]))
            score = len(question_terms & haystack_terms) / max(1, len(question_terms))
            matches.append(TableMatch(table, round(score, 4)))
        return sorted(matches, key=lambda item: item.score, reverse=True)


def _terms(text: str) -> set[str]:
    import re

    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", (text or "").lower())
        if len(token) > 1
    }
