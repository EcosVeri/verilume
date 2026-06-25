"""Evidence extraction, ranking, freshness, and conflict resolution.

This module is intentionally lightweight and dependency-free. It gives the RAG
orchestrator structured objects for source ranking and diagnostics while leaving
heavy semantic reranking to verilume.core.reranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Sequence
from urllib.parse import urlparse

from verilume.core.schemas import LocalSource, WebSource


class QueryType(str, Enum):
    LOCAL = "local"
    CURRENT = "current"
    TIME_SENSITIVE = "current"
    RESEARCH = "research"
    GENERAL = "general"


class FactType(str, Enum):
    STABLE = "stable_fact"
    DYNAMIC = "dynamic_fact"
    NEWS = "news"
    LOCAL_DOCUMENT = "local_document"
    SCIENTIFIC = "scientific_explanation"
    PERSON_LOOKUP = "person_lookup"
    COMPANY_LOOKUP = "company_lookup"


class EvidencePolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    LOCAL_PLUS_MODEL = "local_plus_model"
    LOCAL_PLUS_WEB = "local_plus_web"
    LOCAL_MODEL_WEB = "local_model_web"
    WEB_ONLY = "web_only"


class EvidenceSourceType(str, Enum):
    LOCAL_CHUNK = "local"
    LOCAL = "local"
    WEB = "web"
    AI_KNOWLEDGE = "ai_knowledge"
    AI = "ai_knowledge"


EvidenceKind = EvidenceSourceType


class EvidenceAuthority(str, Enum):
    OFFICIAL = "official"
    INTERNAL = "internal"
    UNIVERSITY = "university"
    SCIENTIFIC = "scientific_paper"
    WIKIPEDIA = "wikipedia"
    NEWS = "news"
    BLOG = "blog"
    SOCIAL_MEDIA = "social_media"
    MODEL = "model"
    STANDARD = "standard"


@dataclass(slots=True)
class QueryUnderstanding:
    primary_type: QueryType
    types: list[QueryType]
    fact_type: FactType = FactType.STABLE
    evidence_policy: EvidencePolicy = EvidencePolicy.LOCAL_MODEL_WEB
    entity_terms: list[str] = field(default_factory=list)
    local_file_question: bool = False
    requires_web_validation: bool = False
    requires_date_reconciliation: bool = False
    time_sensitive_question: bool = False
    personal_company_entity_lookup: bool = False
    ai_knowledge_allowed_as_final: bool = True


@dataclass(slots=True, init=False)
class EvidenceItem:
    source_type: EvidenceSourceType
    title: str
    content: str
    semantic_relevance_score: float
    citation_label: str
    source: Any = None
    url: str = ""
    document_date: date | None = None
    page: int | None = None
    authority: EvidenceAuthority = EvidenceAuthority.STANDARD
    metadata: dict[str, Any] = field(default_factory=dict)
    is_ai_knowledge: bool = False
    _document_date_text: str | None = None

    def __init__(
        self,
        *,
        source_type: EvidenceSourceType | str | None = None,
        kind: EvidenceSourceType | str | None = None,
        title: str,
        content: str,
        semantic_relevance_score: float | None = None,
        score: float | None = None,
        citation_label: str | None = None,
        label: str | None = None,
        source: Any = None,
        url: str = "",
        document_date: date | str | None = None,
        published_date: str | None = None,
        page: int | None = None,
        authority: EvidenceAuthority | str | None = None,
        metadata: dict[str, Any] | None = None,
        is_ai_knowledge: bool | None = None,
    ) -> None:
        resolved_type = _coerce_source_type(source_type or kind)
        resolved_score = semantic_relevance_score if semantic_relevance_score is not None else score
        resolved_label = citation_label if citation_label is not None else label
        resolved_date = published_date if published_date is not None else document_date

        self.source_type = resolved_type
        self.title = title
        self.content = content
        self.semantic_relevance_score = float(resolved_score or 0.0)
        self.citation_label = resolved_label or ""
        self.source = source
        self.url = url or ""
        self.document_date = _parse_document_date(resolved_date)
        self.page = page
        self.authority = _coerce_authority(authority) if authority else _infer_authority(
            resolved_type,
            self.url,
            metadata,
        )
        self.metadata = dict(metadata or {})
        self.is_ai_knowledge = (
            resolved_type == EvidenceSourceType.AI_KNOWLEDGE
            if is_ai_knowledge is None
            else bool(is_ai_knowledge)
        )
        self._document_date_text = _normalize_date_text(resolved_date)

    def citation(self) -> str:
        return f"[{self.citation_label}]" if self.citation_label else "[AI]"

    @classmethod
    def from_ai_knowledge(cls, content: str) -> EvidenceItem:
        return cls(
            source_type=EvidenceSourceType.AI_KNOWLEDGE,
            title="AI knowledge",
            content=content.strip(),
            semantic_relevance_score=0.35,
            citation_label="AI",
            authority=EvidenceAuthority.STANDARD,
            is_ai_knowledge=True,
        )

    @property
    def kind(self) -> EvidenceKind:
        return self.source_type

    @property
    def label(self) -> str:
        return self.citation_label

    @label.setter
    def label(self, value: str) -> None:
        self.citation_label = value

    @property
    def score(self) -> float:
        return self.semantic_relevance_score

    @score.setter
    def score(self, value: float) -> None:
        self.semantic_relevance_score = float(value)

    @property
    def published_date(self) -> str | None:
        return self._document_date_text

    @published_date.setter
    def published_date(self, value: str | None) -> None:
        self._document_date_text = _normalize_date_text(value)
        self.document_date = _parse_document_date(value)


@dataclass(slots=True)
class DateReconciliation:
    freshness_note: str = ""
    local_is_older_than_web: bool = False
    newest_year: int | None = None


@dataclass(slots=True)
class ConflictResolution:
    winner: EvidenceSourceType | None
    evidence_note: str = ""
    source_agreement: str = "unknown"
    confidence: str = "medium"
    should_disclose_conflict: bool = False


@dataclass(slots=True)
class FinalAnswerPayload:
    generator_instructions: str
    evidence_badge: str = ""
    citations: list[str] = field(default_factory=list)


_CURRENT_MARKERS = (
    "latest", "current", "recent", "today", "now", "this year", "2026", "2025",
    "price", "prices", "stock price", "share price", "exchange rate", "weather",
    "forecast", "schedule", "deadline", "population", "gdp", "inflation",
    "unemployment", "interest rate", "law", "directive", "regulation", "reach regulation", "eu reach",
    "ceo", "president", "prime minister",
    "secretary of state", "foreign secretary", "foreign minister", "defence minister",
    "defense minister", "minister of defence", "minister of defense", "finance minister",
    "interior minister", "king", "queen",
    "monarch", "weather", "schedule", "deadline", "news", "breaking", "resign",
    "resigned", "resignation", "updated", "newest", "most recent",
)
_DYNAMIC_FACT_MARKERS = (
    "population", "gdp", "gross domestic product", "stock price", "share price",
    "market cap", "exchange rate", "weather", "forecast", "temperature", "schedule",
    "timetable", "deadline", "law", "regulation", "directive", "tax rate", "inflation",
    "unemployment", "interest rate", "price", "prices", "latest paper", "new paper",
    "newest paper", "recent paper", "current ceo", "ceo of", "president of",
    "prime minister of", "minister of", "head of", "director of",
)
_LOCAL_MARKERS = (
    "local file", "local files", "uploaded", "indexed", "document", "documents", "doc", "docs",
    "knowledge base", "in my files", "in the files", "which file", "which document",
    "database", "data base",
)
_RESEARCH_MARKERS = (
    "paper", "papers", "scientific", "study", "studies", "publication", "review",
    "research", "doi", "arxiv", "journal", "conference",
)
_SCIENTIFIC_EXPLANATION_MARKERS = (
    "explain", "what is", "define", "bayes", "hmc", "spectral analysis",
    "regression", "model", "algorithm", "method", "factor", "theorem",
)
_AUTHORITY_MARKERS = (
    "gov", ".edu", "public.lu", "europa.eu", "who.int", "oecd.org", "worldbank.org",
    "nature.com", "science.org", "arxiv.org", "doi.org", "reuters", "apnews", "bbc",
    "gouvernement.lu", "government",
)
_ENTITY_LOOKUP_MARKERS = (
    "who is", "who's", "ceo", "president", "prime minister", "secretary of state",
    "foreign secretary", "foreign minister", "minister", "founder", "owner", "chair",
    "director", "head of",
)
_ENTITY_STATEMENT_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "does",
    "find",
    "for",
    "from",
    "how",
    "into",
    "latest",
    "local",
    "near",
    "news",
    "online",
    "please",
    "search",
    "show",
    "tell",
    "the",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _normalize_date_text(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _parse_document_date(value: date | str | None) -> date | None:
    if isinstance(value, date):
        return value
    text = _normalize_date_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 1, 1)
    match = _DATE_RE.search(text)
    if match:
        return date(int(match.group(1)), 1, 1)
    return None


def _coerce_source_type(value: EvidenceSourceType | str | None) -> EvidenceSourceType:
    if isinstance(value, EvidenceSourceType):
        return value
    if not value:
        return EvidenceSourceType.LOCAL_CHUNK
    text = str(value).strip().lower()
    if text in {"local", "local_chunk"}:
        return EvidenceSourceType.LOCAL_CHUNK
    if text in {"web", "online"}:
        return EvidenceSourceType.WEB
    if text in {"ai", "ai_knowledge", "model"}:
        return EvidenceSourceType.AI_KNOWLEDGE
    return EvidenceSourceType.LOCAL_CHUNK


def _coerce_authority(value: EvidenceAuthority | str) -> EvidenceAuthority:
    if isinstance(value, EvidenceAuthority):
        return value
    text = str(value).strip().lower()
    for authority in EvidenceAuthority:
        if text == authority.value:
            return authority
    return EvidenceAuthority.STANDARD


def _infer_authority(
    source_type: EvidenceSourceType,
    url: str,
    metadata: dict[str, Any] | None,
) -> EvidenceAuthority:
    if source_type == EvidenceSourceType.LOCAL_CHUNK:
        return EvidenceAuthority.INTERNAL
    if source_type == EvidenceSourceType.AI_KNOWLEDGE:
        return EvidenceAuthority.MODEL

    domain = urlparse(url).netloc.lower()
    metadata_text = " ".join(str(value) for value in (metadata or {}).values()).lower()
    haystack = f"{domain} {url.lower()} {metadata_text}"
    if any(marker in haystack for marker in ("gov", "gouv", "gouvernement", "government", "europa.eu", "public.lu", "who.int", "oecd.org", "worldbank.org")):
        return EvidenceAuthority.OFFICIAL
    if any(marker in haystack for marker in (".edu", ".ac.", "university", "universite", "uni.lu", "harvard.edu", "mit.edu")):
        return EvidenceAuthority.UNIVERSITY
    if any(marker in haystack for marker in ("nature.com", "science.org", "arxiv.org", "doi.org", "springer", "sciencedirect", "journal", "pubmed")):
        return EvidenceAuthority.SCIENTIFIC
    if "wikipedia.org" in haystack:
        return EvidenceAuthority.WIKIPEDIA
    if any(marker in haystack for marker in ("reuters", "apnews", "bbc", "financialtimes", "ft.com", "guardian")):
        return EvidenceAuthority.NEWS
    if any(marker in haystack for marker in ("medium.com", "substack", "blog", "wordpress")):
        return EvidenceAuthority.BLOG
    if any(marker in haystack for marker in ("linkedin.com", "facebook.com", "x.com", "twitter.com", "instagram.com", "tiktok.com")):
        return EvidenceAuthority.SOCIAL_MEDIA
    return EvidenceAuthority.STANDARD


def _coerce_query_understanding(question_or_query: str | QueryUnderstanding) -> QueryUnderstanding:
    if isinstance(question_or_query, QueryUnderstanding):
        return question_or_query
    return classify_question(question_or_query)


def _is_local_source(item: EvidenceItem) -> bool:
    return item.source_type.value == EvidenceSourceType.LOCAL_CHUNK.value


def _is_web_source(item: EvidenceItem) -> bool:
    return item.source_type == EvidenceSourceType.WEB


def _is_ai_source(item: EvidenceItem) -> bool:
    return item.source_type.value == EvidenceSourceType.AI_KNOWLEDGE.value


def _normalized_answer(answer: str | None) -> str:
    text = (answer or "").lower()
    text = re.sub(r"\[[sw]\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_question(question: str) -> QueryUnderstanding:
    lower = (question or "").lower()
    types: list[QueryType] = []
    local = any(marker in lower for marker in _LOCAL_MARKERS)
    historical_age = _looks_like_historical_age_question(lower)
    dynamic = _looks_like_dynamic_fact(lower) and not historical_age
    current = (any(marker in lower for marker in _CURRENT_MARKERS) or dynamic) and not historical_age
    research = any(marker in lower for marker in _RESEARCH_MARKERS)
    entity_lookup = any(marker in lower for marker in _ENTITY_LOOKUP_MARKERS) or _looks_like_entity_statement(question)
    fact_type = _classify_fact_type(lower, local=local, current=current, research=research, entity_lookup=entity_lookup)
    policy = _evidence_policy_for_fact_type(fact_type, local=local, current=current)
    entity_terms = _entity_terms_from_question(question) if entity_lookup else []

    if local:
        types.append(QueryType.LOCAL)
    if current:
        types.append(QueryType.TIME_SENSITIVE)
    if research:
        types.append(QueryType.RESEARCH)
    if not types:
        types.append(QueryType.GENERAL)

    primary = (
        QueryType.TIME_SENSITIVE
        if current
        else QueryType.LOCAL
        if local
        else QueryType.RESEARCH
        if research
        else QueryType.GENERAL
    )
    return QueryUnderstanding(
        primary_type=primary,
        types=types,
        fact_type=fact_type,
        evidence_policy=policy,
        entity_terms=entity_terms,
        local_file_question=local,
        requires_web_validation=current or research or entity_lookup,
        requires_date_reconciliation=current,
        time_sensitive_question=current,
        personal_company_entity_lookup=entity_lookup,
        ai_knowledge_allowed_as_final=not current,
    )


def _evidence_policy_for_fact_type(
    fact_type: FactType,
    *,
    local: bool,
    current: bool,
) -> EvidencePolicy:
    if fact_type == FactType.LOCAL_DOCUMENT:
        return EvidencePolicy.LOCAL_ONLY
    if fact_type in {FactType.DYNAMIC, FactType.NEWS} or current:
        return EvidencePolicy.WEB_ONLY
    if fact_type in {FactType.PERSON_LOOKUP, FactType.COMPANY_LOOKUP, FactType.SCIENTIFIC}:
        return EvidencePolicy.LOCAL_MODEL_WEB
    if local:
        return EvidencePolicy.LOCAL_PLUS_MODEL
    return EvidencePolicy.LOCAL_MODEL_WEB


def _entity_terms_from_question(question: str) -> list[str]:
    text = re.sub(r"\s+", " ", (question or "").strip().strip(".,;:!?"))
    text = re.sub(
        r"^\s*(?:who\s+is|who's|tell\s+me\s+about|search\s+for|find|profile\s+of)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", text)
    blocked = _ENTITY_STATEMENT_STOPWORDS | {
        "is",
        "are",
        "was",
        "were",
        "current",
        "latest",
        "profile",
        "person",
        "company",
    }
    terms = [word.strip("'’").lower() for word in words if len(word.strip("'’")) > 1]
    return [term for term in terms if term not in blocked]


def _looks_like_dynamic_fact(lower: str) -> bool:
    if not lower:
        return False
    if any(marker in lower for marker in _DYNAMIC_FACT_MARKERS):
        if _looks_like_stable_definition(lower):
            return False
        return True
    return False


def _looks_like_stable_definition(lower: str) -> bool:
    if re.search(
        r"\b(?:population|gdp|gross domestic product|stock price|share price|exchange rate|weather|forecast|inflation|unemployment|interest rate|price|prices)\b",
        lower,
    ):
        return False
    return bool(
        re.search(r"\b(?:what is|what are|define|explain|describe)\b", lower)
        and not re.search(r"\b(?:current|latest|recent|today|now|newest|most recent|this year)\b", lower)
    )


def _classify_fact_type(
    lower: str,
    *,
    local: bool,
    current: bool,
    research: bool,
    entity_lookup: bool,
) -> FactType:
    if local:
        return FactType.LOCAL_DOCUMENT
    if "news" in lower or any(marker in lower for marker in ("breaking", "resigned", "resignation")):
        return FactType.NEWS
    if current:
        return FactType.DYNAMIC
    if entity_lookup:
        if any(marker in lower for marker in ("company", "ceo", "founder", "owner", "chair", "director")):
            return FactType.COMPANY_LOOKUP
        return FactType.PERSON_LOOKUP
    if research or any(marker in lower for marker in _SCIENTIFIC_EXPLANATION_MARKERS):
        return FactType.SCIENTIFIC
    return FactType.STABLE


def _looks_like_historical_age_question(lower: str) -> bool:
    return lower.startswith("how old") and any(
        marker in lower
        for marker in (
            "became",
            "become",
            "took office",
            "assumed office",
            "when he",
            "when she",
            "when they",
        )
    )


def _looks_like_entity_statement(question: str) -> bool:
    text = re.sub(r"\s+", " ", (question or "").strip().strip(".,;:!?"))
    if not text or " " not in text:
        return False
    if re.search(r"[?]", question or ""):
        return False
    if re.search(r"\b(?:is|are|was|were|has|have|had|do|does|did|should|could|would|will)\b", text, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’-]+", text)
    if not 2 <= len(words) <= 5:
        return False
    normalized = [word.strip("'’").lower() for word in words]
    if any(len(word) <= 2 or word in _ENTITY_STATEMENT_STOPWORDS for word in normalized):
        return False
    return True


def evidence_from_sources(
    *,
    local_sources: Sequence[LocalSource],
    web_sources: Sequence[WebSource],
    ai_answer: str | None = None,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for source in local_sources:
        items.append(
            EvidenceItem(
                source_type=EvidenceSourceType.LOCAL_CHUNK,
                citation_label=source.label,
                title=source.document,
                content=source.text,
                semantic_relevance_score=float(source.score or 0.0),
                source=source,
                page=source.page,
                document_date=(source.metadata or {}).get("document_date"),
                authority=EvidenceAuthority.INTERNAL,
                metadata=dict(source.metadata or {}),
            )
        )
    for source in web_sources:
        items.append(
            EvidenceItem(
                source_type=EvidenceSourceType.WEB,
                citation_label=source.label,
                title=source.title,
                content=source.content,
                semantic_relevance_score=float(source.score or 0.0)
                if source.score is not None
                else 0.5,
                source=source,
                url=source.url,
                document_date=source.published_date,
                metadata=dict(source.metadata or {}),
            )
        )
    if ai_answer and ai_answer.strip():
        items.append(EvidenceItem.from_ai_knowledge(ai_answer))
    return items


def rank_evidence(
    items: Sequence[EvidenceItem],
    question_or_query: str | QueryUnderstanding,
    today: date | None = None,
    settings: Any | None = None,
) -> list[EvidenceItem]:
    query = _coerce_query_understanding(question_or_query)
    retrieval_weight = float(getattr(settings, "evidence_retrieval_weight", 0.30))
    cross_encoder_weight = float(getattr(settings, "evidence_cross_encoder_weight", 0.25))
    entity_weight = float(getattr(settings, "evidence_entity_weight", 0.20))
    authority_weight = float(getattr(settings, "evidence_authority_weight", 0.10))
    freshness_weight = float(getattr(settings, "evidence_freshness_weight", 0.10))
    ai_consistency_weight = float(getattr(settings, "evidence_ai_consistency_weight", 0.05))
    total_weight = max(
        0.0001,
        retrieval_weight
        + cross_encoder_weight
        + entity_weight
        + authority_weight
        + freshness_weight
        + ai_consistency_weight,
    )
    ranked: list[EvidenceItem] = []
    for item in items:
        retrieval_score = _retrieval_component(item)
        authority_score = _authority_component(item, settings)
        cross_encoder_score = _semantic_component(item)
        entity_score = _entity_match_component(item, query)
        freshness_score = _freshness_component(item, query, today, settings)
        ai_consistency_score = _ai_consistency_component(item, query)
        score = (
            retrieval_weight * retrieval_score
            + authority_weight * authority_score
            + cross_encoder_weight * cross_encoder_score
            + entity_weight * entity_score
            + freshness_weight * freshness_score
            + ai_consistency_weight * ai_consistency_score
        ) / total_weight
        if query.local_file_question and _is_local_source(item):
            score += 0.12
        if query.fact_type == FactType.DYNAMIC and _is_ai_source(item):
            score -= 0.35
        if query.fact_type in {FactType.PERSON_LOOKUP, FactType.COMPANY_LOOKUP} and _is_local_source(item):
            score += _lookup_local_relevance_adjustment(item)
        if query.fact_type == FactType.SCIENTIFIC and item.authority in {
            EvidenceAuthority.UNIVERSITY,
            EvidenceAuthority.SCIENTIFIC,
        }:
            score += 0.06
        item.metadata = dict(item.metadata or {})
        item.metadata.update(
            {
                "evidence_retrieval_score": round(retrieval_score, 4),
                "evidence_authority_score": round(authority_score, 4),
                "evidence_cross_encoder_score": round(cross_encoder_score, 4),
                "evidence_semantic_score": round(cross_encoder_score, 4),
                "evidence_entity_match_score": round(entity_score, 4),
                "evidence_freshness_score": round(freshness_score, 4),
                "evidence_ai_consistency_score": round(ai_consistency_score, 4),
                "evidence_score": round(max(0.0, min(1.0, score)), 4),
                "evidence_fact_type": query.fact_type.value,
                "evidence_policy": query.evidence_policy.value,
                "evidence_authority": item.authority.value,
            }
        )
        item.score = max(0.0, min(1.0, score))
        ranked.append(item)
    return sorted(ranked, key=lambda x: x.score, reverse=True)


def _retrieval_component(item: EvidenceItem) -> float:
    metadata = item.metadata or {}
    candidates = [
        metadata.get("retrieval_score"),
        metadata.get("hybrid_score"),
        metadata.get("fast_rerank_score"),
        item.semantic_relevance_score,
    ]
    for value in candidates:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score > 1.0:
            score = score / (1.0 + score)
        return max(0.0, min(1.0, score))
    return 0.0


def _semantic_component(item: EvidenceItem) -> float:
    metadata = item.metadata or {}
    for key in ("rerank_score", "cross_encoder_score", "semantic_similarity", "fast_rerank_score"):
        try:
            value = float(metadata.get(key))
        except (TypeError, ValueError):
            continue
        if key == "rerank_score":
            value = 1.0 / (1.0 + pow(2.718281828, -value))
        return max(0.0, min(1.0, value))
    return max(0.0, min(1.0, float(item.semantic_relevance_score or 0.0)))


def _entity_match_component(item: EvidenceItem, query: QueryUnderstanding) -> float:
    if query.fact_type not in {FactType.PERSON_LOOKUP, FactType.COMPANY_LOOKUP}:
        return 0.65
    terms = [term for term in query.entity_terms if term]
    if not terms:
        return 0.65
    haystack = re.sub(r"\s+", " ", f"{item.title} {item.content}".lower())
    joined = " ".join(terms)
    reversed_joined = " ".join(reversed(terms))
    if joined in haystack or reversed_joined in haystack:
        return 1.0
    present = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", haystack))
    if present == len(terms):
        return 0.9
    if present:
        return max(0.15, present / max(1, len(terms)))
    return 0.0


def _ai_consistency_component(item: EvidenceItem, query: QueryUnderstanding) -> float:
    if _is_ai_source(item):
        if query.fact_type in {FactType.DYNAMIC, FactType.NEWS}:
            return 0.05
        if query.local_file_question:
            return 0.10
        return 0.70
    return 0.65


def _authority_component(item: EvidenceItem, settings: Any | None = None) -> float:
    if _is_official_evidence(item, settings):
        return 1.0
    return {
        EvidenceAuthority.INTERNAL: 0.90,
        EvidenceAuthority.UNIVERSITY: 0.95,
        EvidenceAuthority.SCIENTIFIC: 0.95,
        EvidenceAuthority.WIKIPEDIA: 0.70,
        EvidenceAuthority.NEWS: 0.80,
        EvidenceAuthority.BLOG: 0.40,
        EvidenceAuthority.SOCIAL_MEDIA: 0.25,
        EvidenceAuthority.MODEL: 0.60,
        EvidenceAuthority.STANDARD: 0.55,
        EvidenceAuthority.OFFICIAL: 1.00,
    }.get(item.authority, 0.55)


def _freshness_component(
    item: EvidenceItem,
    query: QueryUnderstanding,
    today: date | None,
    settings: Any | None = None,
) -> float:
    if query.fact_type not in {FactType.DYNAMIC, FactType.NEWS} and not query.requires_date_reconciliation:
        return 0.65
    if _is_ai_source(item):
        return 0.05
    if not item.document_date:
        return 0.45 if _is_web_source(item) else 0.25
    today = today or date.today()
    freshness_decay_days = int(getattr(settings, "evidence_freshness_decay_days", 365))
    age_days = max(0, (today - item.document_date).days)
    return max(0.0, min(1.0, 1.0 - (age_days / max(1, freshness_decay_days))))


def _lookup_local_relevance_adjustment(item: EvidenceItem) -> float:
    text = f"{item.title} {item.content}".lower()
    if any(marker in text for marker in ("passport", "passp", "identity card", "date of birth", "place of birth")):
        return -0.18
    if any(marker in text for marker in ("thesis", "prof", "professor", "supervisor", "advisor", "university", "publication", "cv")):
        return 0.08
    return 0.0


def _is_official_evidence(item: EvidenceItem, settings: Any | None = None) -> bool:
    if item.authority == EvidenceAuthority.OFFICIAL:
        return True
    markers = tuple(getattr(settings, "evidence_official_domains", _AUTHORITY_MARKERS))
    haystack = f"{item.title} {item.url}".lower()
    return any(str(marker).lower() in haystack for marker in markers)


def reconcile_dates(
    items: Sequence[EvidenceItem],
    question_or_query: str | QueryUnderstanding,
) -> DateReconciliation:
    query = _coerce_query_understanding(question_or_query)
    years: list[tuple[int, EvidenceSourceType]] = []
    for item in items:
        year_candidates = []
        if item.document_date:
            year_candidates.append(str(item.document_date.year))
        haystack = " ".join([item.published_date or "", item.title, item.content[:500]])
        year_candidates.extend(_DATE_RE.findall(haystack))
        for value in year_candidates:
            try:
                years.append((int(value), item.source_type))
            except ValueError:
                pass
    if not years:
        return DateReconciliation(
            freshness_note="No explicit source dates detected."
            if query.requires_date_reconciliation
            else "Freshness not required.",
            local_is_older_than_web=False,
            newest_year=None,
        )
    newest = max(year for year, _kind in years)
    newest_web = max((year for year, kind in years if kind == EvidenceSourceType.WEB), default=None)
    newest_local = max(
        (year for year, kind in years if kind.value == EvidenceSourceType.LOCAL_CHUNK.value),
        default=None,
    )
    local_older = bool(newest_web and newest_local and newest_local < newest_web)
    note = f"Newest visible evidence year: {newest}."
    if local_older:
        note += " Newer web evidence was found than local evidence."
    return DateReconciliation(note, local_older, newest)


def resolve_evidence_conflicts(
    question: str,
    ranked_evidence: Sequence[EvidenceItem],
    *,
    local_answer: str | None = None,
    ai_knowledge_answer: str | None = None,
    web_answer: str | None = None,
    query: QueryUnderstanding | None = None,
    reconciliation: DateReconciliation | None = None,
) -> ConflictResolution:
    query = query or classify_question(question)
    reconciliation = reconciliation or reconcile_dates(ranked_evidence, query)
    if not ranked_evidence:
        return ConflictResolution(None, "No evidence available.", "none", "low")
    kinds = [item.source_type for item in ranked_evidence]
    should_disclose_conflict = bool(
        reconciliation.local_is_older_than_web
        and local_answer
        and web_answer
        and _normalized_answer(local_answer) != _normalized_answer(web_answer)
    )
    top = ranked_evidence[0]
    if query.requires_date_reconciliation and EvidenceSourceType.WEB in kinds:
        winner = EvidenceSourceType.WEB
        note = "Newer reliable web evidence wins for current or time-sensitive facts."
        confidence = "high" if ranked_evidence[0].source_type == EvidenceSourceType.WEB else "medium"
    elif query.local_file_question and any(kind.value == EvidenceSourceType.LOCAL_CHUNK.value for kind in kinds):
        winner = EvidenceSourceType.LOCAL_CHUNK
        note = "Local files win for private or uploaded-document facts."
        confidence = "high"
    elif top.source_type == EvidenceSourceType.LOCAL_CHUNK:
        winner = EvidenceSourceType.LOCAL_CHUNK
        note = "Ranked local evidence is the strongest available evidence."
        confidence = "high" if top.score >= 0.70 else "medium"
    elif top.source_type == EvidenceSourceType.WEB:
        winner = EvidenceSourceType.WEB
        note = "Ranked web evidence is the strongest available evidence."
        confidence = "high" if top.score >= 0.72 else "medium"
    else:
        winner = EvidenceSourceType.AI_KNOWLEDGE
        note = "AI knowledge is the strongest available evidence; it is not externally verified."
        confidence = "medium" if query.ai_knowledge_allowed_as_final else "low"
    labels = {item.label for item in ranked_evidence[:4] if item.label and item.label != "AI"}
    agreement = "multiple sources" if len(labels) > 1 else "single source"
    if reconciliation.local_is_older_than_web:
        agreement = "freshness conflict"
    return ConflictResolution(winner, note, agreement, confidence, should_disclose_conflict)


def verified_evidence_for_generation(
    question: str,
    ranked_evidence: Sequence[EvidenceItem],
) -> list[EvidenceItem]:
    query = classify_question(question)
    items = list(ranked_evidence)
    if query.requires_date_reconciliation:
        web_items = [item for item in items if _is_web_source(item)]
        if web_items:
            return web_items
        local_items = [item for item in items if _is_local_source(item)]
        if local_items:
            return local_items
    if query.local_file_question:
        local_items = [item for item in items if _is_local_source(item)]
        if local_items:
            return local_items
    verified = [item for item in items if not item.is_ai_knowledge]
    return verified or items


def build_final_answer_payload(question: str, ranked_evidence: Sequence[EvidenceItem]) -> FinalAnswerPayload:
    verified = verified_evidence_for_generation(question, ranked_evidence)
    query = classify_question(question)
    blocks = []
    citations = [item.citation() for item in verified if item.label and item.label != "AI"]
    if query.requires_date_reconciliation and any(_is_web_source(item) for item in verified):
        evidence_badge = "Current web verified"
    elif any(_is_local_source(item) for item in verified):
        evidence_badge = "Local evidence grounded"
    elif any(_is_web_source(item) for item in verified):
        evidence_badge = "Web verified"
    else:
        evidence_badge = "AI knowledge only"

    for item in verified[:10]:
        content = " ".join((item.content or "").split())[:1200]
        date = f"\nPublished/visible date: {item.published_date}" if item.published_date else ""
        url = f"\nURL: {item.url}" if item.url else ""
        blocks.append(
            f"{item.citation()} kind={item.kind.value}; score={item.score:.3f}\nTitle: {item.title}{url}{date}\nEvidence: {content}"
        )
    instructions = (
        "Use only the verified evidence below. These items are already ranked from strongest to weakest. Use these citations exactly.\n\n"
        + "\n\n".join(blocks)
    )
    return FinalAnswerPayload(
        evidence_badge=evidence_badge,
        citations=citations,
        generator_instructions=instructions,
    )
