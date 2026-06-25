"""Graph-assisted retrieval built on the local SQLite knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from verilume.core.knowledge_graph import GraphContext, KnowledgeGraph
from verilume.core.schemas import LocalSource


@dataclass(frozen=True, slots=True)
class GraphRAGContext:
    seed_entities: list[str] = field(default_factory=list)
    expanded_entities: list[str] = field(default_factory=list)
    related_documents: list[str] = field(default_factory=list)
    related_chunks: list[str] = field(default_factory=list)
    graph_summary: str = ""


class GraphRAGRetriever:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def retrieve_graph_context(self, question: str) -> GraphRAGContext:
        context = self.graph.graph_context_for_query(question)
        return _to_graphrag_context(context)

    def retrieve_graph_chunks(
        self,
        question: str,
        graph_context: GraphRAGContext | None = None,
        *,
        limit: int = 5,
    ) -> list[LocalSource]:
        context = graph_context or self.retrieve_graph_context(question)
        if not context.seed_entities and not context.expanded_entities:
            return []
        entity_names = [*context.seed_entities, *context.expanded_entities]
        rows = self._mention_rows(entity_names, limit=limit)
        sources: list[LocalSource] = []
        for index, row in enumerate(rows, start=1):
            sources.append(
                LocalSource(
                    label=f"S{index}",
                    document=str(row["document"] or ""),
                    page=row["page"],
                    chunk_id=str(row["chunk_id"] or ""),
                    text=str(row["text_snippet"] or ""),
                    score=0.82,
                    metadata={
                        "source_type": "knowledge_graph",
                        "entity": str(row["name"] or ""),
                        "graph_summary": context.graph_summary,
                    },
                )
            )
        return sources

    def _mention_rows(self, entity_names: Sequence[str], *, limit: int) -> list:
        entities = []
        for name in entity_names:
            entities.extend(self.graph.search_entity(name))
        entity_ids = []
        seen = set()
        for entity in entities:
            if entity.id not in seen:
                entity_ids.append(entity.id)
                seen.add(entity.id)
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        with self.graph._connect() as conn:
            return conn.execute(
                f"""
                SELECT m.*, e.name
                FROM mentions m
                JOIN entities e ON e.id = m.entity_id
                WHERE m.entity_id IN ({placeholders})
                ORDER BY length(m.text_snippet) DESC
                LIMIT ?
                """,
                [*entity_ids, int(limit)],
            ).fetchall()


def _to_graphrag_context(context: GraphContext) -> GraphRAGContext:
    return GraphRAGContext(
        seed_entities=[entity.name for entity in context.seed_entities],
        expanded_entities=[neighbor.entity.name for neighbor in context.neighbors],
        related_documents=list(context.related_documents),
        related_chunks=list(context.related_chunks),
        graph_summary=context.summary,
    )
