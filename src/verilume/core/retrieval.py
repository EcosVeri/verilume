"""Chroma vector store access and local retrieval."""

from __future__ import annotations

from pathlib import Path

import chromadb

from verilume.core.embeddings import EmbeddingService
from verilume.core.schemas import LocalSource


def _cosine_distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


class ChromaRetriever:
    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        embeddings: EmbeddingService,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.embeddings = embeddings
        self._client = None
        self._collection = None

    @property
    def client(self):
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reconnect(self) -> None:
        self._collection = None
        self._client = None

    def reset(self) -> None:
        client = self.client
        self._collection = None
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def delete_document(self, source_path: str) -> None:
        try:
            self.collection.delete(where={"source_path": source_path})
        except Exception:
            pass

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        if not ids:
            return
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, k: int = 5, score_threshold: float = 0.35) -> list[LocalSource]:
        if not query.strip() or self.count() == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        sources: list[LocalSource] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] or {}
            score = _cosine_distance_to_score(distances[index] if index < len(distances) else None)
            if score < score_threshold:
                continue
            page_value = metadata.get("page")
            page = int(page_value) if isinstance(page_value, int) and page_value > 0 else None
            sources.append(
                LocalSource(
                    label=f"S{len(sources) + 1}",
                    document=str(metadata.get("document") or "Unknown document"),
                    page=page,
                    chunk_id=str(ids[index] if index < len(ids) else ""),
                    text=text or "",
                    score=score,
                    metadata=metadata,
                )
            )
        return sources
