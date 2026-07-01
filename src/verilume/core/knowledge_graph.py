"""Lightweight SQLite knowledge graph for local document entities."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence


ENTITY_TYPES = {
    "person",
    "organization",
    "location",
    "publication",
    "topic",
    "document",
    "dataset",
    "method",
    "law",
}

RELATION_TYPES = {
    "authored",
    "coauthored",
    "affiliated_with",
    "mentions",
    "cites",
    "works_at",
    "supervised_by",
    "related_to",
    "published_in",
    "located_in",
}

TOPIC_TERMS = (
    "Bayesian inference",
    "Bayesian model selection",
    "Hamiltonian Monte Carlo",
    "regression analysis",
    "spectral analysis",
    "hydrology",
    "statistics",
    "machine learning",
)


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    name: str
    normalized_name: str
    type: str
    confidence: float = 0.8


@dataclass(frozen=True, slots=True)
class GraphNeighbor:
    entity: Entity
    relation: str
    direction: str
    document: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    confidence: float = 0.8


@dataclass(frozen=True, slots=True)
class GraphContext:
    seed_entities: list[Entity] = field(default_factory=list)
    neighbors: list[GraphNeighbor] = field(default_factory=list)
    related_documents: list[str] = field(default_factory=list)
    related_chunks: list[str] = field(default_factory=list)
    summary: str = ""


class KnowledgeGraph:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_entity(self, name: str, type: str, confidence: float = 0.8) -> str:
        clean_name = _clean_name(name)
        entity_type = type if type in ENTITY_TYPES else "topic"
        normalized = normalize_name(clean_name)
        entity_id = _id("entity", normalized, entity_type)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entities(id, name, normalized_name, type, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    type=excluded.type,
                    confidence=max(entities.confidence, excluded.confidence)
                """,
                (entity_id, clean_name, normalized, entity_type, float(confidence)),
            )
        return entity_id

    def add_relation(
        self,
        source_name: str,
        relation: str,
        target_name: str,
        document: str = "",
        page: int | None = None,
        chunk_id: str = "",
        confidence: float = 0.8,
        source_type: str = "topic",
        target_type: str = "topic",
    ) -> str:
        relation_type = relation if relation in RELATION_TYPES else "related_to"
        source_id = self.add_entity(source_name, source_type, confidence)
        target_id = self.add_entity(target_name, target_type, confidence)
        relation_id = _id("relation", source_id, relation_type, target_id, document, str(page), chunk_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO relations(
                    id, source_id, relation, target_id, document, page, chunk_id, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (relation_id, source_id, relation_type, target_id, document, page, chunk_id, confidence),
            )
        return relation_id

    def add_mention(
        self,
        entity_name: str,
        document: str,
        page: int | None,
        chunk_id: str,
        text_snippet: str,
        entity_type: str = "topic",
    ) -> str:
        entity_id = self.add_entity(entity_name, entity_type)
        mention_id = _id("mention", entity_id, document, str(page), chunk_id, text_snippet[:80])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mentions(
                    id, entity_id, document, page, chunk_id, text_snippet
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mention_id, entity_id, document, page, chunk_id, text_snippet[:1000]),
            )
        return mention_id

    def index_chunk(
        self,
        text: str,
        *,
        document: str,
        page: int | None = None,
        chunk_id: str = "",
    ) -> list[Entity]:
        document_id = self.add_entity(document, "document", 1.0)
        extracted = extract_entities(text, document=document)
        entities: list[Entity] = []
        for name, entity_type, confidence in extracted:
            entity_id = self.add_entity(name, entity_type, confidence)
            self.add_mention(name, document, page, chunk_id, _snippet_for_entity(text, name), entity_type)
            self.add_relation(
                document,
                "mentions",
                name,
                document=document,
                page=page,
                chunk_id=chunk_id,
                confidence=confidence,
                source_type="document",
                target_type=entity_type,
            )
            entity = self.get_entity(entity_id)
            if entity is not None:
                entities.append(entity)

        for source, relation, target, source_type, target_type in extract_relations(text):
            self.add_relation(
                source,
                relation,
                target,
                document=document,
                page=page,
                chunk_id=chunk_id,
                source_type=source_type,
                target_type=target_type,
            )
        if not entities:
            entity = self.get_entity(document_id)
            if entity is not None:
                entities.append(entity)
        return entities

    def search_entity(self, name: str) -> list[Entity]:
        normalized = normalize_name(name)
        pattern = f"%{normalized}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM entities
                WHERE normalized_name = ? OR normalized_name LIKE ?
                ORDER BY confidence DESC, length(normalized_name) ASC
                """,
                (normalized, pattern),
            ).fetchall()
        return [_entity_from_row(row) for row in rows]

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return _entity_from_row(row) if row is not None else None

    def neighbors(self, entity_id: str, depth: int = 1) -> list[GraphNeighbor]:
        seen = {entity_id}
        frontier = {entity_id}
        results: list[GraphNeighbor] = []
        for _ in range(max(1, depth)):
            next_frontier: set[str] = set()
            for current in frontier:
                for neighbor in self._direct_neighbors(current):
                    if neighbor.entity.id not in seen:
                        next_frontier.add(neighbor.entity.id)
                        seen.add(neighbor.entity.id)
                    results.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return results

    def graph_context_for_query(self, question: str) -> GraphContext:
        seeds = _dedupe_entities(
            entity
            for name in candidate_entity_names(question)
            for entity in self.search_entity(name)
        )
        neighbors = _dedupe_neighbors(
            neighbor
            for seed in seeds[:5]
            for neighbor in self.neighbors(seed.id, depth=1)
        )
        related_documents = _related_documents(self, [*seeds, *(neighbor.entity for neighbor in neighbors)])
        related_chunks = _related_chunks(self, [*seeds, *(neighbor.entity for neighbor in neighbors)])
        summary = _context_summary(seeds, neighbors, related_documents)
        return GraphContext(
            seed_entities=seeds,
            neighbors=neighbors,
            related_documents=related_documents,
            related_chunks=related_chunks,
            summary=summary,
        )

    def _direct_neighbors(self, entity_id: str) -> list[GraphNeighbor]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.source_id, r.target_id, r.relation, r.document, r.page, r.chunk_id,
                    r.confidence AS relation_confidence,
                    e.id AS entity_id, e.name, e.normalized_name, e.type,
                    e.confidence AS entity_confidence
                FROM relations r
                JOIN entities e ON e.id = r.target_id
                WHERE r.source_id = ?
                UNION ALL
                SELECT
                    r.source_id, r.target_id, r.relation, r.document, r.page, r.chunk_id,
                    r.confidence AS relation_confidence,
                    e.id AS entity_id, e.name, e.normalized_name, e.type,
                    e.confidence AS entity_confidence
                FROM relations r
                JOIN entities e ON e.id = r.source_id
                WHERE r.target_id = ?
                """,
                (entity_id, entity_id),
            ).fetchall()
        neighbors: list[GraphNeighbor] = []
        for row in rows:
            direction = "out" if row["source_id"] == entity_id else "in"
            neighbors.append(
                GraphNeighbor(
                    entity=_entity_from_relation_row(row),
                    relation=str(row["relation"]),
                    direction=direction,
                    document=row["document"],
                    page=row["page"],
                    chunk_id=row["chunk_id"],
                    confidence=float(row["relation_confidence"] or 0.0),
                )
            )
        return neighbors

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    document TEXT,
                    page INTEGER,
                    chunk_id TEXT,
                    confidence REAL DEFAULT 0.8
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mentions (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    document TEXT,
                    page INTEGER,
                    chunk_id TEXT,
                    text_snippet TEXT
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


def extract_entities(text: str, *, document: str = "") -> list[tuple[str, str, float]]:
    found: list[tuple[str, str, float]] = []
    if document:
        found.append((document, "document", 1.0))
    for topic in TOPIC_TERMS:
        if topic.lower() in (text or "").lower():
            entity_type = "method" if topic.lower() in {"hamiltonian monte carlo", "regression analysis", "spectral analysis"} else "topic"
            found.append((topic, entity_type, 0.86))
    for org in re.findall(
        r"\b(?:University|Université|Institute|Laboratory|Department|Faculty)\s+of\s+"
        r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5}",
        text or "",
    ):
        found.append((org.strip(" .,;:"), "organization", 0.88))
    for person in re.findall(
        r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3}\b",
        text or "",
    ):
        if _looks_like_person(person):
            found.append((person.strip(" .,;:"), "person", 0.78))
    doi_match = re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text or "", flags=re.IGNORECASE)
    for doi in doi_match:
        found.append((doi, "publication", 0.9))
    return _dedupe_extracted(found)


def extract_relations(text: str) -> list[tuple[str, str, str, str, str]]:
    relations: list[tuple[str, str, str, str, str]] = []
    for match in re.finditer(
        r"(?P<person>[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3})"
        r".{0,80}?\b(?:at|from|with)\s+"
        r"(?P<org>University of [A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})",
        text or "",
    ):
        relations.append(
            (
                match.group("person").strip(),
                "affiliated_with",
                match.group("org").strip(" .,;:"),
                "person",
                "organization",
            )
        )
    for match in re.finditer(
        r"(?P<person>[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3})"
        r".{0,60}?\b(?:supervised by|supervisor[: ]+)\s+"
        r"(?P<supervisor>[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3})",
        text or "",
        flags=re.IGNORECASE,
    ):
        relations.append(
            (
                match.group("person").strip(),
                "supervised_by",
                match.group("supervisor").strip(),
                "person",
                "person",
            )
        )
    return relations


def candidate_entity_names(question: str) -> list[str]:
    names = [name for name, _, _ in extract_entities(question)]
    for topic in TOPIC_TERMS:
        if topic.lower() in (question or "").lower():
            names.append(topic)
    return _dedupe([name for name in names if name])


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())).strip()


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:18]
    return f"{prefix}_{digest}"


def _looks_like_person(value: str) -> bool:
    lower = value.lower()
    if any(word in lower for word in ("university", "department", "faculty", "institute")):
        return False
    return len(value.split()) <= 4


def _dedupe_extracted(items: Sequence[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str, float]] = []
    for name, entity_type, confidence in items:
        key = (normalize_name(name), entity_type)
        if key in seen or not key[0]:
            continue
        seen.add(key)
        result.append((name, entity_type, confidence))
    return result


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_name(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _dedupe_entities(values) -> list[Entity]:
    seen: set[str] = set()
    result: list[Entity] = []
    for entity in values:
        if entity.id not in seen:
            seen.add(entity.id)
            result.append(entity)
    return result


def _dedupe_neighbors(values) -> list[GraphNeighbor]:
    seen: set[tuple[str, str, str]] = set()
    result: list[GraphNeighbor] = []
    for neighbor in values:
        key = (neighbor.entity.id, neighbor.relation, neighbor.direction)
        if key not in seen:
            seen.add(key)
            result.append(neighbor)
    return result


def _snippet_for_entity(text: str, name: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    lowered = name.lower()
    for sentence in sentences:
        if lowered in sentence.lower():
            return sentence.strip()
    return (text or "").strip()[:500]


def _entity_from_row(row: sqlite3.Row) -> Entity:
    return Entity(
        id=str(row["id"]),
        name=str(row["name"]),
        normalized_name=str(row["normalized_name"]),
        type=str(row["type"]),
        confidence=float(row["confidence"] or 0.0),
    )


def _entity_from_relation_row(row: sqlite3.Row) -> Entity:
    return Entity(
        id=str(row["entity_id"]),
        name=str(row["name"]),
        normalized_name=str(row["normalized_name"]),
        type=str(row["type"]),
        confidence=float(row["entity_confidence"] or 0.0),
    )


def _related_documents(graph: KnowledgeGraph, entities: Sequence[Entity]) -> list[str]:
    entity_ids = [entity.id for entity in entities]
    if not entity_ids:
        return []
    placeholders = ",".join("?" for _ in entity_ids)
    with graph._connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT document FROM mentions WHERE entity_id IN ({placeholders}) AND document != ''",
            entity_ids,
        ).fetchall()
    return [str(row["document"]) for row in rows]


def _related_chunks(graph: KnowledgeGraph, entities: Sequence[Entity]) -> list[str]:
    entity_ids = [entity.id for entity in entities]
    if not entity_ids:
        return []
    placeholders = ",".join("?" for _ in entity_ids)
    with graph._connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT chunk_id FROM mentions WHERE entity_id IN ({placeholders}) AND chunk_id != ''",
            entity_ids,
        ).fetchall()
    return [str(row["chunk_id"]) for row in rows]


def _context_summary(
    seeds: Sequence[Entity],
    neighbors: Sequence[GraphNeighbor],
    related_documents: Sequence[str],
) -> str:
    seed_text = ", ".join(entity.name for entity in seeds[:5]) or "No seed entities"
    neighbor_text = ", ".join(neighbor.entity.name for neighbor in neighbors[:6])
    doc_text = ", ".join(related_documents[:6])
    parts = [f"Seeds: {seed_text}."]
    if neighbor_text:
        parts.append(f"Neighbors: {neighbor_text}.")
    if doc_text:
        parts.append(f"Documents: {doc_text}.")
    return " ".join(parts)
