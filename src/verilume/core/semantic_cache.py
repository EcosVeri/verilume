"""Persistent semantic answer cache keyed by evidence context."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from verilume.core.evidence import EvidencePolicy, FactType, QueryUnderstanding
from verilume.core.query_preprocessing import normalize_query
from verilume.core.schemas import LocalSource, RAGResponse, WebSource


SEMANTIC_CACHE_VERSION = 1
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.88

_QUERY_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "define",
    "describe",
    "did",
    "do",
    "does",
    "explain",
    "for",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "please",
    "show",
    "summarise",
    "summarize",
    "tell",
    "the",
    "to",
    "total",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


@dataclass(slots=True)
class CachedAnswer:
    query_hash: str
    normalized_query: str
    answer: str
    local_sources: list[dict[str, Any]]
    web_sources: list[dict[str, Any]]
    model_answer: str
    evidence_scores: dict[str, float]
    timestamp: str
    policy: str
    diagnostics: dict[str, Any]
    document_fingerprint: str
    web_enabled: bool
    generation_backend: str
    model_name: str
    web_provider: str = ""
    confidence: str = ""
    used_web: bool = False
    cache_version: int = SEMANTIC_CACHE_VERSION

    def is_fresh(
        self,
        now: datetime,
        ttl_seconds: int,
        current_document_fingerprint: str,
    ) -> bool:
        if self.document_fingerprint != current_document_fingerprint:
            return False

        if ttl_seconds <= 0:
            return True

        cached_at = _parse_timestamp(self.timestamp)
        if cached_at is None:
            return False

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        return (now - cached_at).total_seconds() <= ttl_seconds

    def to_rag_response(self) -> RAGResponse:
        diagnostics = copy.deepcopy(self.diagnostics or {})
        diagnostics["cache_hit"] = True
        diagnostics["semantic_cache_hit"] = True
        diagnostics["semantic_cache_policy"] = self.policy
        diagnostics["semantic_cache_timestamp"] = self.timestamp

        local_sources = [_local_source_from_dict(item) for item in self.local_sources]
        web_sources = [_web_source_from_dict(item) for item in self.web_sources]
        confidence = self.confidence or str(diagnostics.get("confidence") or "cached")

        return RAGResponse(
            answer=self.answer,
            local_sources=local_sources,
            web_sources=web_sources,
            used_web=bool(self.used_web or web_sources),
            confidence=confidence,
            diagnostics=diagnostics,
            resolved_query=str(diagnostics.get("resolved_query") or "") or None,
            original_query=str(diagnostics.get("original_query") or "") or None,
        )


class SemanticCache:
    """Small JSON-backed cache for expensive evidence-ranked answers."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = Path(cache_path).expanduser()

    def lookup(
        self,
        question: str,
        *,
        policy: str | EvidencePolicy,
        document_fingerprint: str,
        web_enabled: bool,
        generation_backend: str,
        model_name: str,
        web_provider: str = "",
    ) -> CachedAnswer | None:
        normalized = normalize_cache_query(question)
        query_hash = _query_hash(normalized)
        policy_value = _policy_value(policy)
        entries = [
            entry
            for entry in self._read()
            if entry.policy == policy_value
            and entry.document_fingerprint == document_fingerprint
            and entry.web_enabled == bool(web_enabled)
            and entry.generation_backend == generation_backend
            and entry.model_name == model_name
            and entry.web_provider == web_provider
        ]

        for entry in entries:
            if entry.query_hash == query_hash:
                return entry

        best_entry: CachedAnswer | None = None
        best_score = 0.0
        for entry in entries:
            score = _query_similarity(normalized, entry.normalized_query)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
            best_entry.diagnostics = copy.deepcopy(best_entry.diagnostics or {})
            best_entry.diagnostics["semantic_cache_similarity"] = round(best_score, 4)
            return best_entry

        return None

    def store(
        self,
        question: str,
        response: RAGResponse,
        *,
        policy: str | EvidencePolicy,
        document_fingerprint: str,
        web_enabled: bool,
        generation_backend: str,
        model_name: str,
        web_provider: str = "",
    ) -> None:
        if not _is_cacheable_response(response):
            return

        normalized = normalize_cache_query(question)
        diagnostics = copy.deepcopy(response.diagnostics or {})
        diagnostics.setdefault("original_query", response.original_query or question)
        if response.resolved_query:
            diagnostics.setdefault("resolved_query", response.resolved_query)
        diagnostics["semantic_cache_stored"] = True
        diagnostics["confidence"] = response.confidence

        entry = CachedAnswer(
            query_hash=_query_hash(normalized),
            normalized_query=normalized,
            answer=response.answer,
            local_sources=[_local_source_to_dict(item) for item in response.local_sources],
            web_sources=[_web_source_to_dict(item) for item in response.web_sources],
            model_answer=str(diagnostics.get("model_answer") or ""),
            evidence_scores=_evidence_scores_from_diagnostics(diagnostics),
            timestamp=datetime.now(timezone.utc).isoformat(),
            policy=_policy_value(policy),
            diagnostics=diagnostics,
            document_fingerprint=document_fingerprint,
            web_enabled=bool(web_enabled),
            generation_backend=generation_backend,
            model_name=model_name,
            web_provider=web_provider,
            confidence=response.confidence,
            used_web=response.used_web,
        )

        entries = self._read()
        entries = [
            item
            for item in entries
            if not (
                item.query_hash == entry.query_hash
                and item.policy == entry.policy
                and item.document_fingerprint == entry.document_fingerprint
                and item.web_enabled == entry.web_enabled
                and item.generation_backend == entry.generation_backend
                and item.model_name == entry.model_name
                and item.web_provider == entry.web_provider
            )
        ]
        entries.append(entry)
        self._write(entries[-250:])

    def clear(self) -> None:
        try:
            self.cache_path.unlink()
        except FileNotFoundError:
            return

    def clear_for_document_change(self) -> None:
        self.clear()

    def _read(self) -> list[CachedAnswer]:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

        raw_entries = payload.get("entries", payload if isinstance(payload, list) else [])
        entries: list[CachedAnswer] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(CachedAnswer(**_cached_answer_defaults(item)))
            except TypeError:
                continue
        return entries

    def _write(self, entries: list[CachedAnswer]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SEMANTIC_CACHE_VERSION,
            "entries": [asdict(entry) for entry in entries],
        }
        self.cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def semantic_cache_ttl_seconds(
    query_understanding: QueryUnderstanding,
    settings: Any,
) -> int:
    fact_type = query_understanding.fact_type
    if fact_type in {FactType.DYNAMIC, FactType.NEWS} or query_understanding.requires_date_reconciliation:
        return int(getattr(settings, "semantic_cache_current_ttl_seconds", 3600))
    if fact_type == FactType.LOCAL_DOCUMENT or query_understanding.evidence_policy == EvidencePolicy.LOCAL_ONLY:
        return int(getattr(settings, "semantic_cache_local_ttl_seconds", 0))
    if fact_type in {FactType.PERSON_LOOKUP, FactType.COMPANY_LOOKUP}:
        return int(getattr(settings, "semantic_cache_entity_ttl_seconds", 604800))
    return int(getattr(settings, "semantic_cache_stable_ttl_seconds", 604800))


def document_fingerprint(settings: Any) -> str:
    """Fingerprint document state without requiring Chroma internals."""

    hasher = hashlib.sha256()
    for name in ("collection_name", "chroma_dir", "docs_dir", "manifest_path"):
        value = str(getattr(settings, name, "") or "")
        hasher.update(name.encode("utf-8"))
        hasher.update(value.encode("utf-8"))

    manifest_path = Path(getattr(settings, "manifest_path", "") or "").expanduser()
    if manifest_path.exists():
        _hash_file_metadata_and_content(hasher, manifest_path)

    docs_dir = Path(getattr(settings, "docs_dir", "") or "").expanduser()
    if docs_dir.exists():
        for path in sorted(item for item in docs_dir.rglob("*") if item.is_file()):
            _hash_file_metadata(hasher, path)

    return hasher.hexdigest()


def normalize_cache_query(question: str) -> str:
    normalized = normalize_query(question).canonical
    if not normalized:
        normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    return normalized


def _is_cacheable_response(response: RAGResponse) -> bool:
    if not response.answer.strip():
        return False
    if response.confidence in {"needs-token", "model-selection-warning", "generation-error", "clarification"}:
        return False
    if response.confidence == "low" and not response.local_sources and not response.web_sources:
        return False
    return True


def _cached_answer_defaults(item: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "local_sources": [],
        "web_sources": [],
        "model_answer": "",
        "evidence_scores": {},
        "diagnostics": {},
        "web_provider": "",
        "confidence": "",
        "used_web": False,
        "cache_version": SEMANTIC_CACHE_VERSION,
    }
    value = dict(defaults)
    value.update(item)
    return value


def _query_hash(normalized_query: str) -> str:
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def _policy_value(policy: str | EvidencePolicy) -> str:
    if isinstance(policy, EvidencePolicy):
        return policy.value
    return str(policy or "").strip()


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _query_similarity(left: str, right: str) -> float:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    sequence = SequenceMatcher(None, left, right).ratio()
    if not left_tokens or not right_tokens:
        return sequence

    shared = left_tokens & right_tokens
    jaccard = len(shared) / max(1, len(left_tokens | right_tokens))
    overlap = len(shared) / max(1, min(len(left_tokens), len(right_tokens)))
    return max(sequence, jaccard, overlap * 0.96)


def _meaningful_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
    return {token for token in tokens if token not in _QUERY_STOPWORDS and len(token) > 1}


def _evidence_scores_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in diagnostics.get("ranked_evidence") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("source_type") or "")
        if not label:
            continue
        try:
            scores[label] = float(item.get("final_score") or item.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
    return scores


def _local_source_to_dict(source: LocalSource) -> dict[str, Any]:
    return {
        "label": source.label,
        "document": source.document,
        "page": source.page,
        "chunk_id": source.chunk_id,
        "text": source.text,
        "score": source.score,
        "metadata": copy.deepcopy(source.metadata),
    }


def _web_source_to_dict(source: WebSource) -> dict[str, Any]:
    return {
        "label": source.label,
        "title": source.title,
        "url": source.url,
        "content": source.content,
        "score": source.score,
        "published_date": source.published_date,
        "metadata": copy.deepcopy(source.metadata),
    }


def _local_source_from_dict(item: dict[str, Any]) -> LocalSource:
    return LocalSource(
        label=str(item.get("label") or ""),
        document=str(item.get("document") or ""),
        page=item.get("page"),
        chunk_id=str(item.get("chunk_id") or ""),
        text=str(item.get("text") or ""),
        score=float(item.get("score") or 0.0),
        metadata=dict(item.get("metadata") or {}),
    )


def _web_source_from_dict(item: dict[str, Any]) -> WebSource:
    return WebSource(
        label=str(item.get("label") or ""),
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        content=str(item.get("content") or ""),
        score=item.get("score"),
        published_date=item.get("published_date"),
        metadata=dict(item.get("metadata") or {}),
    )


def _hash_file_metadata(hasher: Any, path: Path) -> None:
    try:
        stat = path.stat()
    except OSError:
        return
    hasher.update(str(path).encode("utf-8"))
    hasher.update(str(stat.st_size).encode("utf-8"))
    hasher.update(str(int(stat.st_mtime)).encode("utf-8"))


def _hash_file_metadata_and_content(hasher: Any, path: Path) -> None:
    _hash_file_metadata(hasher, path)
    try:
        hasher.update(path.read_bytes())
    except OSError:
        return
