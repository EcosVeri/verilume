"""Sentence-transformers embedding service."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str):
    return SentenceTransformer(model_name, device=device)


class EmbeddingService:
    """Thin wrapper around SentenceTransformer with normalized embeddings."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    @property
    def model(self):
        return _load_sentence_transformer(self.model_name, self.device)

    def embed_documents(self, texts: Iterable[str], batch_size: int = 128) -> list[list[float]]:
        values = [text or "" for text in texts]
        if not values:
            return []
        embeddings = self.model.encode(
            values,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query], batch_size=1)[0]
