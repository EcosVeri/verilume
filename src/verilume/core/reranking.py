"""Fast lexical/semantic reranking with optional cross-encoder refinement.

The default path is dependency-light and fast: it blends the upstream semantic
score with query term coverage and phrase/title matches. A cross-encoder can
still be enabled for slower, higher-cost refinement of the already-small
candidate pool.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence, TypeVar

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency path
    CrossEncoder = None  # type: ignore

from verilume.core.schemas import LocalSource, WebSource

T = TypeVar("T", LocalSource, WebSource)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']{1,}")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "answer",
    "are",
    "based",
    "can",
    "contains",
    "does",
    "document",
    "documents",
    "file",
    "files",
    "find",
    "for",
    "from",
    "have",
    "how",
    "into",
    "local",
    "more",
    "my",
    "of",
    "on",
    "say",
    "search",
    "source",
    "sources",
    "the",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "your",
}


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str):
    if CrossEncoder is None:
        raise RuntimeError("sentence-transformers CrossEncoder is not available")
    return CrossEncoder(model_name, device=device, local_files_only=True)


def _source_text(source: LocalSource | WebSource, max_chars: int = 1600) -> str:
    if isinstance(source, LocalSource):
        metadata = source.metadata or {}
        metadata_text = "\n".join(
            str(metadata.get(key, ""))
            for key in (
                "document_title",
                "authors",
                "keywords",
                "abstract",
                "section_heading",
                "document_kind",
            )
        )
        text = f"{source.document}\n{metadata_text}\n{source.text}"
    else:
        text = f"{source.title}\n{source.content}"
    return " ".join(text.split())[:max_chars]


def query_terms(query: str) -> list[str]:
    """Meaningful terms used by fast reranking and evidence filtering."""

    terms: list[str] = []
    for match in _TOKEN_RE.finditer(query or ""):
        term = match.group(0).lower().strip("'")
        if len(term) < 3 or term in _STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _source_title(source: LocalSource | WebSource) -> str:
    if isinstance(source, LocalSource):
        return source.document
    return source.title


def _fast_rerank_score(
    query: str,
    source: LocalSource | WebSource,
    *,
    semantic_weight: float = 0.52,
    lexical_weight: float = 0.48,
    phrase_bonus_full: float = 0.28,
    phrase_bonus_partial: float = 0.16,
    mismatch_penalty: float = 0.55,
    mismatch_threshold: float = 0.72,
    single_match_penalty: float = 0.78,
    single_match_threshold: float = 0.78,
) -> tuple[float, int, float]:
    terms = query_terms(query)
    source_text = _source_text(source, max_chars=2600)
    haystack = _normalize_text(f"{_source_title(source)} {source_text}")
    title = _normalize_text(_source_title(source))
    haystack_terms = set(query_terms(haystack))
    title_terms = set(query_terms(title))
    base_score = max(0.0, min(1.0, float(source.score or 0.0)))

    if not terms:
        return base_score, 0, 0.0

    term_weights = {term: _term_weight(term) for term in terms}
    total_weight = sum(term_weights.values()) or 1.0
    matched = [term for term in terms if term in haystack_terms]
    title_matches = [term for term in terms if term in title_terms]
    matched_weight = sum(term_weights[term] for term in matched)
    title_weight = sum(term_weights[term] for term in title_matches)
    coverage = matched_weight / total_weight
    title_coverage = title_weight / total_weight

    normalized_query = _normalize_text(query)
    phrase_bonus = 0.0
    if normalized_query and normalized_query in haystack:
        phrase_bonus = phrase_bonus_full
    elif len(terms) >= 2 and " ".join(terms[:2]) in haystack:
        phrase_bonus = phrase_bonus_partial

    lexical_score = min(1.0, coverage * 0.72 + title_coverage * 0.22 + phrase_bonus)
    total_weight = max(0.0001, semantic_weight + lexical_weight)
    score = (base_score * semantic_weight + lexical_score * lexical_weight) / total_weight

    if not matched and base_score < mismatch_threshold:
        score *= mismatch_penalty
    if len(matched) == 1 and len(terms) >= 3 and base_score < single_match_threshold:
        score *= single_match_penalty

    return max(0.0, min(1.0, score)), len(matched), coverage


def _term_weight(term: str) -> float:
    length_bonus = min(1.1, max(0.0, (len(term) - 4) * 0.12))
    digit_bonus = 0.35 if any(ch.isdigit() for ch in term) else 0.0
    uncommon_bonus = 0.45 if len(term) >= 10 else 0.0
    return 1.0 + length_bonus + digit_bonus + uncommon_bonus


def fast_rerank_sources(
    query: str,
    sources: Sequence[T],
    top_k: int,
    *,
    semantic_weight: float = 0.52,
    lexical_weight: float = 0.48,
    phrase_bonus_full: float = 0.28,
    phrase_bonus_partial: float = 0.16,
    mismatch_penalty: float = 0.55,
    mismatch_threshold: float = 0.72,
    single_match_penalty: float = 0.78,
    single_match_threshold: float = 0.78,
) -> list[T]:
    values = list(sources)
    scored: list[tuple[float, int, T]] = []
    for index, source in enumerate(values):
        score, overlap, coverage = _fast_rerank_score(
            query,
            source,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            phrase_bonus_full=phrase_bonus_full,
            phrase_bonus_partial=phrase_bonus_partial,
            mismatch_penalty=mismatch_penalty,
            mismatch_threshold=mismatch_threshold,
            single_match_penalty=single_match_penalty,
            single_match_threshold=single_match_threshold,
        )
        source.metadata = dict(source.metadata or {})
        source.metadata["fast_rerank_score"] = round(score, 4)
        source.metadata["query_overlap"] = overlap
        source.metadata["query_coverage"] = round(coverage, 4)
        source.score = max(float(source.score or 0.0), score)
        scored.append((score, -index, source))
    ranked = [source for _score, _index, source in sorted(scored, reverse=True)]
    return ranked[:top_k]


def rerank_sources(
    query: str,
    sources: Sequence[T],
    *,
    model_name: str = "BAAI/bge-reranker-base",
    device: str = "cpu",
    top_k: int = 8,
    enabled: bool = True,
    semantic_weight: float = 0.52,
    lexical_weight: float = 0.48,
    phrase_bonus_full: float = 0.28,
    phrase_bonus_partial: float = 0.16,
    mismatch_penalty: float = 0.55,
    mismatch_threshold: float = 0.72,
    single_match_penalty: float = 0.78,
    single_match_threshold: float = 0.78,
) -> list[T]:
    """Rerank LocalSource/WebSource objects with a cross encoder.

    The reranker score is stored in source.metadata["rerank_score"]. The public
    source.score is also blended upward so existing UI confidence code benefits.
    """
    values = fast_rerank_sources(
        query,
        sources,
        top_k=max(top_k, len(sources)),
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        phrase_bonus_full=phrase_bonus_full,
        phrase_bonus_partial=phrase_bonus_partial,
        mismatch_penalty=mismatch_penalty,
        mismatch_threshold=mismatch_threshold,
        single_match_penalty=single_match_penalty,
        single_match_threshold=single_match_threshold,
    )
    if not enabled or not query.strip() or len(values) <= 1:
        return values[:top_k]

    try:
        model = _load_cross_encoder(model_name, device)
        candidate_pool = values[: min(len(values), max(top_k * 2, 12))]
        pairs = [(query, _source_text(source)) for source in candidate_pool]
        scores = model.predict(pairs)
    except Exception:
        return values[:top_k]

    for source, score in zip(candidate_pool, scores, strict=False):
        value = float(score)
        source.metadata = dict(source.metadata or {})
        source.metadata["rerank_score"] = value
        # Blend normalized old score with bounded reranker score for UI compatibility.
        bounded = 1.0 / (1.0 + pow(2.718281828, -value))
        source.score = max(float(source.score or 0.0), bounded)

    ranked = sorted(
        candidate_pool,
        key=lambda item: float((item.metadata or {}).get("rerank_score", item.score or 0.0)),
        reverse=True,
    )
    seen = {id(source) for source in ranked}
    ranked.extend(source for source in values if id(source) not in seen)
    return ranked[:top_k]


def rerank_local_sources(
    query: str,
    sources: Sequence[LocalSource],
    *,
    model_name: str,
    device: str,
    top_k: int,
    enabled: bool,
    semantic_weight: float = 0.52,
    lexical_weight: float = 0.48,
    phrase_bonus_full: float = 0.28,
    phrase_bonus_partial: float = 0.16,
    mismatch_penalty: float = 0.55,
    mismatch_threshold: float = 0.72,
    single_match_penalty: float = 0.78,
    single_match_threshold: float = 0.78,
) -> list[LocalSource]:
    ranked = rerank_sources(
        query,
        sources,
        model_name=model_name,
        device=device,
        top_k=top_k,
        enabled=enabled,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        phrase_bonus_full=phrase_bonus_full,
        phrase_bonus_partial=phrase_bonus_partial,
        mismatch_penalty=mismatch_penalty,
        mismatch_threshold=mismatch_threshold,
        single_match_penalty=single_match_penalty,
        single_match_threshold=single_match_threshold,
    )
    for index, source in enumerate(ranked, start=1):
        source.label = f"S{index}"
    return ranked


def rerank_web_sources(
    query: str,
    sources: Sequence[WebSource],
    *,
    model_name: str,
    device: str,
    top_k: int,
    enabled: bool,
    semantic_weight: float = 0.52,
    lexical_weight: float = 0.48,
    phrase_bonus_full: float = 0.28,
    phrase_bonus_partial: float = 0.16,
    mismatch_penalty: float = 0.55,
    mismatch_threshold: float = 0.72,
    single_match_penalty: float = 0.78,
    single_match_threshold: float = 0.78,
) -> list[WebSource]:
    if not enabled:
        ranked = list(sources)[:top_k]
        for index, source in enumerate(ranked, start=1):
            source.label = f"W{index}"
        return ranked

    ranked = rerank_sources(
        query,
        sources,
        model_name=model_name,
        device=device,
        top_k=top_k,
        enabled=enabled,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        phrase_bonus_full=phrase_bonus_full,
        phrase_bonus_partial=phrase_bonus_partial,
        mismatch_penalty=mismatch_penalty,
        mismatch_threshold=mismatch_threshold,
        single_match_penalty=single_match_penalty,
        single_match_threshold=single_match_threshold,
    )
    for index, source in enumerate(ranked, start=1):
        source.label = f"W{index}"
    return ranked
