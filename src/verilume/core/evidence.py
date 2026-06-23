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
    STANDARD = "standard"


@dataclass(slots=True)
class QueryUnderstanding:
    primary_type: QueryType
    types: list[QueryType]
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
    "price", "prices", "law", "directive", "regulation", "reach regulation", "eu reach",
    "ceo", "president", "prime minister",
    "secretary of state", "foreign secretary", "foreign minister", "defence minister",
    "defense minister", "minister of defence", "minister of defense", "finance minister",
    "interior minister", "king", "queen",
    "monarch", "weather", "schedule", "deadline", "news", "breaking", "resign",
    "resigned", "resignation", "updated", "newest", "most recent",
)
_LOCAL_MARKERS = (
    "local file", "local files", "uploaded", "indexed", "document", "documents",
    "knowledge base", "in my files", "in the files", "which file", "which document",
)
_RESEARCH_MARKERS = (
    "paper", "papers", "scientific", "study", "studies", "publication", "review",
    "research", "doi", "arxiv", "journal", "conference",
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
    if text == EvidenceAuthority.OFFICIAL.value:
        return EvidenceAuthority.OFFICIAL
    if text == EvidenceAuthority.INTERNAL.value:
        return EvidenceAuthority.INTERNAL
    return EvidenceAuthority.STANDARD


def _infer_authority(
    source_type: EvidenceSourceType,
    url: str,
    metadata: dict[str, Any] | None,
) -> EvidenceAuthority:
    if source_type == EvidenceSourceType.LOCAL_CHUNK:
        return EvidenceAuthority.INTERNAL
    if source_type == EvidenceSourceType.AI_KNOWLEDGE:
        return EvidenceAuthority.STANDARD

    domain = urlparse(url).netloc.lower()
    metadata_text = " ".join(str(value) for value in (metadata or {}).values()).lower()
    haystack = f"{domain} {url.lower()} {metadata_text}"
    if any(marker in haystack for marker in _AUTHORITY_MARKERS):
        return EvidenceAuthority.OFFICIAL
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
    current = any(marker in lower for marker in _CURRENT_MARKERS) and not historical_age
    research = any(marker in lower for marker in _RESEARCH_MARKERS)
    entity_lookup = any(marker in lower for marker in _ENTITY_LOOKUP_MARKERS) or _looks_like_entity_statement(question)

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
        local_file_question=local,
        requires_web_validation=current or research or entity_lookup,
        requires_date_reconciliation=current,
        time_sensitive_question=current,
        personal_company_entity_lookup=entity_lookup,
        ai_knowledge_allowed_as_final=not current,
    )


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
    authority_boost = float(getattr(settings, "evidence_authority_boost", 0.35))
    freshness_boost = float(getattr(settings, "evidence_freshness_boost", 0.1))
    freshness_decay_days = int(getattr(settings, "evidence_freshness_decay_days", 365))
    ranked: list[EvidenceItem] = []
    for item in items:
        score = float(item.score or 0.0)
        if _is_local_source(item):
            score += 0.25
            if query.local_file_question:
                score += 0.35
        elif _is_web_source(item):
            score += 0.15
            if query.requires_date_reconciliation:
                score += 0.35
            if _is_official_evidence(item, settings):
                score += authority_boost
            if item.published_date:
                score += freshness_boost
            if today and item.document_date:
                age_days = max(0, (today - item.document_date).days)
                if age_days <= freshness_decay_days:
                    score += freshness_boost * 0.5
        elif _is_ai_source(item):
            score += 0.05
            if not query.ai_knowledge_allowed_as_final:
                score -= 0.5
        item.metadata = dict(item.metadata or {})
        item.metadata["evidence_score"] = round(score, 4)
        item.score = score
        ranked.append(item)
    return sorted(ranked, key=lambda x: x.score, reverse=True)


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
    if query.requires_date_reconciliation and EvidenceSourceType.WEB in kinds:
        winner = EvidenceSourceType.WEB
        note = "Newer reliable web evidence wins for current or time-sensitive facts."
        confidence = "high" if ranked_evidence[0].source_type == EvidenceSourceType.WEB else "medium"
    elif query.local_file_question and any(kind.value == EvidenceSourceType.LOCAL_CHUNK.value for kind in kinds):
        winner = EvidenceSourceType.LOCAL_CHUNK
        note = "Local files win for private or uploaded-document facts."
        confidence = "high"
    elif EvidenceSourceType.WEB in kinds:
        winner = EvidenceSourceType.WEB
        note = "Web evidence is the strongest available evidence."
        confidence = "medium"
    else:
        winner = EvidenceSourceType.AI_KNOWLEDGE
        note = "Only AI knowledge was available; it is not externally verified."
        confidence = "low"
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
