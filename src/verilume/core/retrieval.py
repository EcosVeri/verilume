"""Chroma vector store access with fast hybrid retrieval.

Features:
- Dense semantic search through Chroma embeddings.
- Lightweight BM25-style lexical scoring over candidate chunks.
- Reciprocal-rank fusion of dense + lexical results.
- Backward-compatible .search(...) method used by existing app code.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import chromadb

from verilume.core.embeddings import EmbeddingService
from verilume.core.schemas import LocalSource

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}")


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


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
        *,
        settings=None,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.embeddings = embeddings
        self._client = None
        self._collection = None
        self._all_items_cache: tuple[str, list[dict]] | None = None
        self._lexical_index_cache: tuple[str, dict] | None = None
        self.rrf_constant = max(1, int(getattr(settings, "rrf_constant", 60)))
        self.rrf_dense_weight = max(0.0, float(getattr(settings, "rrf_dense_weight", 1.0)))
        self.rrf_lexical_weight = max(0.0, float(getattr(settings, "rrf_lexical_weight", 1.0)))
        self.rrf_semantic_boost = max(0.0, float(getattr(settings, "rrf_semantic_boost", 0.25)))
        self.rrf_score_scale = max(0.0, float(getattr(settings, "rrf_score_scale", 28.0)))

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

    def close(self, *, clear_system_cache: bool = False) -> None:
        client = self._client
        self._collection = None
        self._client = None
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass
        if clear_system_cache:
            try:
                clear_cache = getattr(client, "clear_system_cache", None)
                if callable(clear_cache):
                    clear_cache()
            except Exception:
                pass

    def reset(self) -> None:
        self.invalidate_caches(drop_disk=True)
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
            self.invalidate_caches(drop_disk=True)
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
        self.invalidate_caches(drop_disk=True)

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.35,
        *,
        mode: str = "hybrid",
        dense_k: int | None = None,
        lexical_k: int | None = None,
        document_filter: str | None = None,
    ) -> list[LocalSource]:
        """Return local sources using dense, lexical, or hybrid retrieval.

        ``document_filter`` restricts candidates to chunks whose metadata
        ``document`` equals the given filename — used when the question is
        known to be about one specific indexed document (e.g. a suggested
        prompt generated from that document).
        """
        if not query.strip() or self.count() == 0:
            return []
        mode = (mode or "hybrid").lower()
        if mode == "dense":
            return self._dense_search(
                query, k=k, score_threshold=score_threshold, document_filter=document_filter
            )
        if mode in {"bm25", "lexical"}:
            return self._lexical_search(
                query, k=k, score_threshold=score_threshold, document_filter=document_filter
            )
        return self._hybrid_search(
            query,
            k=k,
            score_threshold=score_threshold,
            dense_k=dense_k or max(k * 6, 30),
            lexical_k=lexical_k or max(k * 6, 30),
            document_filter=document_filter,
        )

    def sample_sources_by_document(
        self,
        *,
        chunks_per_document: int = 2,
        limit_documents: int | None = None,
    ) -> list[LocalSource]:
        """Return representative indexed chunks grouped by source document."""
        if self.count() == 0:
            return []
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in self._all_items():
            metadata = item.get("metadata") or {}
            document = str(metadata.get("document") or "Unknown document")
            grouped[document].append(item)

        sources: list[LocalSource] = []
        document_items = list(grouped.items())
        if limit_documents is not None:
            document_items = document_items[: max(0, int(limit_documents))]
        for document_index, (_document, items) in enumerate(document_items, start=1):
            ranked_items = sorted(
                items,
                key=lambda item: (
                    _metadata_page(item.get("metadata") or {}),
                    str(item.get("id") or ""),
                ),
            )
            for chunk_index, item in enumerate(ranked_items[: max(1, int(chunks_per_document))], start=1):
                source = _local_source_from_raw(
                    text=item.get("document") or "",
                    metadata=item.get("metadata") or {},
                    chunk_id=str(item.get("id") or ""),
                    score=1.0,
                    rank=(document_index * 1000) + chunk_index,
                    retrieval="corpus",
                )
                sources.append(source)
        return _relabel(sources)

    def _dense_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        document_filter: str | None = None,
    ) -> list[LocalSource]:
        query_embedding = self.embeddings.embed_query(query)
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": max(1, k),
            "include": ["documents", "metadatas", "distances"],
        }
        if document_filter:
            query_kwargs["where"] = {"document": document_filter}
        result = self.collection.query(**query_kwargs)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        sources: list[LocalSource] = []
        for index, text in enumerate(documents):
            score = _cosine_distance_to_score(distances[index] if index < len(distances) else None)
            if score < score_threshold:
                continue
            source = _local_source_from_raw(
                text=text or "",
                metadata=metadatas[index] or {},
                chunk_id=str(ids[index] if index < len(ids) else ""),
                score=score,
                rank=index + 1,
                retrieval="dense",
            )
            sources.append(source)
        return _relabel(sources)

    def _lexical_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        document_filter: str | None = None,
    ) -> list[LocalSource]:
        index = self._lexical_index()
        if document_filter:
            index = [
                item
                for item in index
                if str((item.get("metadata") or {}).get("document") or "") == document_filter
            ]
        ranked = _bm25_rank(query, index)
        sources: list[LocalSource] = []
        for rank, (item, score) in enumerate(ranked[: max(k, 1)], start=1):
            normalized = min(1.0, score / 8.0) if score > 0 else 0.0
            if normalized < max(0.05, score_threshold * 0.45):
                continue
            sources.append(
                _local_source_from_raw(
                    text=item["document"],
                    metadata=item["metadata"],
                    chunk_id=item["id"],
                    score=normalized,
                    rank=rank,
                    retrieval="bm25",
                )
            )
        return _relabel(sources)

    def _hybrid_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        dense_k: int,
        lexical_k: int,
        document_filter: str | None = None,
    ) -> list[LocalSource]:
        dense = self._dense_search(
            query,
            k=dense_k,
            score_threshold=max(0.0, score_threshold * 0.5),
            document_filter=document_filter,
        )
        lexical = self._lexical_search(
            query,
            k=lexical_k,
            score_threshold=max(0.0, score_threshold * 0.3),
            document_filter=document_filter,
        )

        by_key: dict[str, LocalSource] = {}
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for rank, source in enumerate(dense, start=1):
            key = _source_key(source)
            by_key[key] = source
            ranks[key]["dense"] = rank
        for rank, source in enumerate(lexical, start=1):
            key = _source_key(source)
            if key not in by_key:
                by_key[key] = source
            else:
                by_key[key].score = max(by_key[key].score, source.score)
                by_key[key].metadata = {**by_key[key].metadata, **source.metadata}
            ranks[key]["bm25"] = rank

        fused: list[tuple[LocalSource, float]] = []
        for key, source in by_key.items():
            r = ranks[key]
            dense_rrf = 1.0 / (self.rrf_constant + r["dense"]) if "dense" in r else 0.0
            bm25_rrf = 1.0 / (self.rrf_constant + r["bm25"]) if "bm25" in r else 0.0
            fused_score = self.rrf_score_scale * (
                self.rrf_dense_weight * dense_rrf
                + self.rrf_lexical_weight * bm25_rrf
            ) + self.rrf_semantic_boost * float(source.score or 0.0)
            source.metadata = dict(source.metadata or {})
            source.metadata["hybrid_score"] = fused_score
            source.metadata["retrieval"] = "+".join(sorted(r))
            source.score = max(float(source.score or 0.0), min(1.0, fused_score / 1.2))
            if source.score >= score_threshold:
                fused.append((source, fused_score))

        ranked_sources = [item for item, _ in sorted(fused, key=lambda pair: pair[1], reverse=True)]
        return _relabel(ranked_sources[:k])

    def _all_items(self) -> list[dict]:
        total = self.count()
        signature = self._collection_signature(total)
        if self._all_items_cache and self._all_items_cache[0] == signature:
            return self._all_items_cache[1]

        items = []
        batch_size = 1000
        for offset in range(0, total, batch_size):
            try:
                result = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
            except TypeError:
                result = self.collection.get(include=["documents", "metadatas"])
            except Exception:
                return []
            ids = result.get("ids", []) or []
            docs = result.get("documents", []) or []
            metas = result.get("metadatas", []) or []
            for index, doc in enumerate(docs):
                document = doc or ""
                metadata = metas[index] or {}
                combined = " ".join(
                    [
                        str(metadata.get("document", "")),
                        str(metadata.get("document_title", "")),
                        str(metadata.get("authors", "")),
                        str(metadata.get("keywords", "")),
                        str(metadata.get("abstract", "")),
                        str(metadata.get("section_heading", "")),
                        str(metadata.get("document_kind", "")),
                        document,
                    ]
                )
                items.append(
                    {
                        "id": str(ids[index] if index < len(ids) else ""),
                        "document": document,
                        "metadata": metadata,
                        "tokens": _tokens(combined),
                    }
                )
            if len(docs) < batch_size:
                break

        self._all_items_cache = (signature, items)
        return items

    def invalidate_caches(self, *, drop_disk: bool = False) -> None:
        self._all_items_cache = None
        self._lexical_index_cache = None
        if drop_disk:
            try:
                self._lexical_index_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _invalidate_cache(self) -> None:
        self.invalidate_caches(drop_disk=False)

    def _lexical_index(self) -> dict:
        total = self.count()
        signature = self._collection_signature(total)
        if self._lexical_index_cache and self._lexical_index_cache[0] == signature:
            return self._lexical_index_cache[1]

        persisted = self._load_persisted_lexical_index(signature)
        if persisted is not None:
            self._lexical_index_cache = (signature, persisted)
            return persisted

        items = self._all_items()
        index = _build_bm25_index(items)
        self._lexical_index_cache = (signature, index)
        self._write_persisted_lexical_index(signature, index)
        return index

    def refresh_lexical_index(self) -> dict:
        self.invalidate_caches(drop_disk=True)
        return self._lexical_index()

    @property
    def _lexical_index_path(self) -> Path:
        return self.chroma_dir / ".lexical_index.pkl"

    def _collection_signature(self, total: int | None = None) -> str:
        total = self.count() if total is None else total
        digest = hashlib.blake2b(digest_size=20)
        digest.update(str(total).encode("utf-8"))
        if total <= 0:
            return digest.hexdigest()
        try:
            peek = self.collection.peek(limit=min(64, total))
        except Exception:
            try:
                peek = self.collection.get(limit=min(64, total), include=["metadatas"])
            except Exception:
                peek = {}
        for key in ("ids", "metadatas"):
            digest.update(repr(peek.get(key, [])).encode("utf-8", errors="ignore"))
        return digest.hexdigest()

    def _load_persisted_lexical_index(self, signature: str) -> dict | None:
        try:
            with self._lexical_index_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict) or payload.get("signature") != signature:
            return None
        index = payload.get("index")
        return index if isinstance(index, dict) else None

    def _write_persisted_lexical_index(self, signature: str, index: dict) -> None:
        temp_path: Path | None = None
        try:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile("wb", delete=False, dir=self.chroma_dir) as handle:
                pickle.dump(
                    {"signature": signature, "index": index},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                temp_path = Path(handle.name)
            temp_path.replace(self._lexical_index_path)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass


def _build_bm25_index(items: list[dict]) -> dict:
    docs_tokens = [item.get("tokens") or _tokens(item["document"]) for item in items]
    lengths = [len(tokens) for tokens in docs_tokens]
    avgdl = sum(lengths) / max(1, len(lengths))
    df = Counter()
    postings = defaultdict(list)
    for index, tokens in enumerate(docs_tokens):
        counts = Counter(tokens)
        df.update(counts.keys())
        for term, frequency in counts.items():
            postings[term].append((index, frequency))
    return {
        "items": items,
        "avgdl": avgdl,
        "df": df,
        "lengths": lengths,
        "n_docs": len(items),
        "postings": postings,
    }


def _bm25_rank(query: str, items: list[dict] | dict) -> list[tuple[dict, float]]:
    q_terms = _tokens(query)
    if not q_terms:
        return []

    if isinstance(items, dict):
        return _bm25_rank_index(q_terms, items)

    if not items:
        return []
    return _bm25_rank_index(q_terms, _build_bm25_index(items))


def _bm25_rank_index(q_terms: list[str], index: dict) -> list[tuple[dict, float]]:
    items = index.get("items") or []
    if not items:
        return []
    avgdl = float(index.get("avgdl") or 0.0)
    df = index.get("df") or {}
    lengths = index.get("lengths") or []
    n_docs = int(index.get("n_docs") or len(items))
    postings = index.get("postings") or {}
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = defaultdict(float)
    for term, query_frequency in Counter(q_terms).items():
        document_frequency = int(df.get(term, 0) or 0)
        if document_frequency <= 0:
            continue
        idf = math.log(1 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5))
        for document_index, term_frequency in postings.get(term, ()):
            document_length = lengths[document_index] if document_index < len(lengths) else 0
            denom = term_frequency + k1 * (1 - b + b * document_length / max(1.0, avgdl))
            scores[document_index] += (
                query_frequency * idf * (term_frequency * (k1 + 1)) / denom
            )
    return sorted(
        ((items[index], score) for index, score in scores.items() if score > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )


def _local_source_from_raw(
    *,
    text: str,
    metadata: dict,
    chunk_id: str,
    score: float,
    rank: int,
    retrieval: str,
) -> LocalSource:
    page_value = metadata.get("page")
    page = int(page_value) if isinstance(page_value, int) and page_value > 0 else None
    meta = dict(metadata or {})
    meta["retrieval"] = retrieval
    meta["retrieval_rank"] = rank
    return LocalSource(
        label="S0",
        document=str(metadata.get("document") or "Unknown document"),
        page=page,
        chunk_id=chunk_id,
        text=text or "",
        score=float(score),
        metadata=meta,
    )


def _metadata_page(metadata: dict) -> int:
    value = metadata.get("page")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return 999999


def _source_key(source: LocalSource) -> str:
    return source.chunk_id or f"{source.document}:{source.page}:{source.text[:80]}"


def _relabel(sources: Iterable[LocalSource]) -> list[LocalSource]:
    values = list(sources)
    for index, source in enumerate(values, start=1):
        source.label = f"S{index}"
    return values
