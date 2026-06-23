"""Sentence-transformers embedding service."""

from __future__ import annotations

import hashlib
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

DEFAULT_CACHE_DIR = Path.home() / ".verilume" / "embedding_cache"


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=_resolve_device(device))


def _resolve_device(device: str) -> str:
    value = (device or "cpu").strip().lower()
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class EmbeddingService:
    """Thin wrapper around SentenceTransformer with normalized embeddings."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        *,
        cache_dir: Path | str | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR
        self.cache_enabled = bool(cache_enabled)

    @property
    def model(self):
        return _load_sentence_transformer(self.model_name, self.device)

    def embed_documents(self, texts: Iterable[str], batch_size: int = 128) -> list[list[float]]:
        values = [text or "" for text in texts]
        if not values:
            return []
        if self.cache_enabled:
            cached = self._embed_documents_cached(values, batch_size=batch_size)
            if cached is not None:
                return cached
        embeddings = self.model.encode(
            values,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query], batch_size=1)[0]

    def _embed_documents_cached(
        self,
        values: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]] | None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None

        results: list[list[float] | None] = [None] * len(values)
        misses: list[tuple[int, str, Path]] = []

        for index, text in enumerate(values):
            path = self._cache_path(text)
            cached = _read_embedding_cache(path)
            if cached is None:
                misses.append((index, text, path))
            else:
                results[index] = cached

        if misses:
            embeddings = self.model.encode(
                [text for _index, text, _path in misses],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            for (index, _text, path), embedding in zip(misses, embeddings, strict=False):
                vector = [float(value) for value in embedding]
                results[index] = vector
                _write_embedding_cache(path, vector)

        if any(item is None for item in results):
            return None
        return [item for item in results if item is not None]

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.blake2b(
            f"{self.model_name}\0{self.device}\0{text}".encode("utf-8", errors="ignore"),
            digest_size=24,
        ).hexdigest()
        return self.cache_dir / f"{digest}.pkl"


def _read_embedding_cache(path: Path) -> list[float] | None:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _write_embedding_cache(path: Path, vector: list[float]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
            pickle.dump(vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
