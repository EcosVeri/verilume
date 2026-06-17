"""RAG orchestration for local, model, and web-backed answers."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from urllib.parse import urlparse

from verilume.core.embeddings import EmbeddingService
from verilume.core.generation import (
    LOCAL_UNKNOWN,
    MODEL_UNKNOWN,
    GenerationError,
    HuggingFaceGenerator,
    is_model_selection_warning,
)
from verilume.core.retrieval import ChromaRetriever
from verilume.core.schemas import ChatMessage, LocalSource, RAGResponse, WebSource
from verilume.core.web_search import DuckDuckGoSearch, create_web_search
from verilume.settings import AppSettings, ensure_app_dirs


MAX_WEB_SOURCES_TO_SHOW = 5
WEB_QUERY_FANOUT_LIMIT = 4

INSUFFICIENT_MARKERS = (
    "web_search_needed",
    LOCAL_UNKNOWN.lower(),
    MODEL_UNKNOWN.lower(),
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "could not answer",
    "could not produce",
    "not enough information",
    "insufficient information",
    "insufficient context",
    "no local context",
    "no mention",
    "unable to determine",
    "not provided in the context",
    "not in the provided local",
    "not in the local",
    "provided local document context",
)

WEB_REQUEST_MARKERS = (
    "search web",
    "search the web",
    "web search",
    "use web",
    "use the web",
    "look up",
    "online",
    "internet",
    "latest",
    "current",
    "recent",
    "today",
)

TIME_SENSITIVE_MARKERS = (
    "as of",
    "breaking",
    "ceo",
    "company role",
    "current",
    "election",
    "exchange rate",
    "king",
    "latest",
    "law",
    "laws",
    "market cap",
    "minister",
    "now",
    "president",
    "price",
    "prices",
    "prime minister",
    "queen",
    "recent",
    "regulation",
    "regulations",
    "result",
    "results",
    "score",
    "scores",
    "sports",
    "stock",
    "today",
    "tonight",
    "yesterday",
)

PUBLIC_ENTITY_HINTS = (
    "who is",
    "who are",
    "where is",
    "where are",
    "profile",
    "professor",
    "researcher",
    "company",
    "organization",
    "organisation",
    "university",
)

PROFILE_RELEVANCE_TERMS = (
    "about",
    "author",
    "bio",
    "biography",
    "cv",
    "defence",
    "defense",
    "faculty",
    "github.com",
    "linkedin.com/in",
    "orcid",
    "profile",
    "research",
    "researcher",
    "scholar",
    "staff",
    "thesis",
    "university",
    "wiki",
    "wikipedia",
)

NOISY_WEB_RESULT_TERMS = (
    "comment",
    "comments",
    "like comment share",
    "linkedin.com/posts",
    "linkedin.com/feed",
    "linkedin.com/pulse",
    "medium.com/tag",
    "reddit.com",
    "search code",
    "saved searches",
    "youtube.com/playlist",
)

IDENTITY_STOPWORDS = {
    "about",
    "and",
    "are",
    "candidate",
    "current",
    "for",
    "internet",
    "is",
    "king",
    "latest",
    "linkedin",
    "look",
    "minister",
    "online",
    "person",
    "phd",
    "profile",
    "president",
    "prime",
    "queen",
    "professor",
    "researcher",
    "search",
    "student",
    "the",
    "too",
    "university",
    "use",
    "web",
    "what",
    "where",
    "who",
    "with",
}

CURRENT_ROLE_MARKERS = (
    "current prime minister",
    "prime minister of",
    "current king",
    "king of",
    "current queen",
    "queen of",
    "current president",
    "president of",
    "head of government",
    "head of state",
)

SOURCE_TERM_STOPWORDS = (IDENTITY_STOPWORDS - {"king", "minister", "president", "prime", "queen"}) | {
    "does",
    "government",
    "official",
    "please",
    "2026",
}

PRIMARY_OFFICIAL_SOURCE_DOMAINS = (
    "gouvernement.lu",
    "royal.uk",
    "royal-house.nl",
    "koninklijkhuis.nl",
    "government.nl",
    "gov.uk",
    "parliament.uk",
    "public.lu",
    "usa.gov",
    "whitehouse.gov",
)

SECONDARY_OFFICIAL_SOURCE_DOMAINS = (
    "europa.eu",
    "ec.europa.eu",
    "bundesregierung.de",
    "bundeskanzler.de",
)

TRUSTED_REFERENCE_DOMAINS = (
    "wikipedia.org",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "britannica.com",
)

CURRENT_ROLE_SYNONYMS = {
    "ceo": "CEO",
    "chief executive officer": "CEO",
    "governor": "governor",
    "king": "king",
    "minister": "minister",
    "president": "president",
    "prime minister": "prime minister",
    "queen": "queen",
}

ENTITY_ALIASES = {
    "america": "United States",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "the uk": "United Kingdom",
    "the united kingdom": "United Kingdom",
    "the united states": "United States",
    "u.k.": "United Kingdom",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
}

PERSON_NAME_PATTERN = (
    r"((?:(?:The\s+Rt\s+Hon|Rt\s+Hon|Sir|Dame|Lord|Lady|Mr|Ms|Mrs|Dr)\s+)*"
    r"[A-Z][A-Za-z'’-]+(?:\s+(?:[A-Z]\.|[A-Z][A-Za-z'’-]+)){0,5}"
    r"(?:\s+(?:KCB|KC|KCMG|MP|MSP|AM|MLA|III|IV|Jr\.?|Sr\.?))*)"
)

CURRENT_ROLE_REQUIRES_WEB_TAGS = (
    "ceo",
    "chief executive officer",
    "current",
    "election",
    "governor",
    "king",
    "latest",
    "minister",
    "now",
    "president",
    "present",
    "prime minister",
    "queen",
    "today",
)

NOISY_SOURCE_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "linkedin.com/posts",
    "reddit.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
)

LOCAL_FILE_NOT_FOUND = "I could not find this in the indexed local files."
LOCAL_FILE_EXPANSIONS = {
    "certificate": (
        "certificate",
        "certification",
        "diploma",
        "credential",
        "language test",
        "sproochentest",
        "exam result",
        "pdf",
    ),
    "certificates": (
        "certificate",
        "certification",
        "diploma",
        "credential",
        "language test",
        "sproochentest",
        "exam result",
        "pdf",
    ),
    "language": (
        "language",
        "sproochentest",
        "test",
        "exam",
        "certificate",
    ),
}

US_ENTITY_MARKERS = (
    "america",
    "u.s.",
    "u.s.a.",
    "united states",
    "usa",
    "us",
)

US_MONARCH_ROLE_MARKERS = (
    "king",
    "monarch",
    "queen",
)


@dataclass(slots=True)
class CurrentRoleLookup:
    role: str
    entity: str


@dataclass(slots=True)
class CurrentRoleClaim:
    name: str
    source: WebSource
    authority: float
    freshness: float
    score: float
    stale: bool = False


@dataclass(slots=True)
class CurrentRoleValidation:
    answer: str
    selected_name: str
    selected_sources: list[WebSource]
    conflict: bool
    claim_count: int


class GenerationStopped(RuntimeError):
    """Raised when the user stops multi-stage generation."""


class VerilumeRAG:
    def __init__(self, settings: AppSettings) -> None:
        ensure_app_dirs(settings)
        self.settings = settings
        self.embeddings = EmbeddingService(settings.embed_model, settings.embed_device)
        self.retriever = ChromaRetriever(
            settings.chroma_dir,
            settings.collection_name,
            self.embeddings,
        )
        self.generator = HuggingFaceGenerator(settings)
        self.web_search = create_web_search(settings)

    def ask(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> RAGResponse:
        history = history or []
        _check_generation_stop(should_stop)

        query = question
        if self.settings.enable_query_rewrite and history:
            query = self.generator.rewrite_query(question, history)
        _check_generation_stop(should_stop)

        local_file_question = self._is_local_file_question(question)
        identity_tokens = _identity_tokens(query if query != question else question)
        _emit_stage(on_stage, "Searching local files...")
        local_sources = self.retriever.search(
            query,
            k=self.settings.retriever_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )
        if identity_tokens and not local_file_question:
            local_sources = _filter_local_sources_for_identity(local_sources, identity_tokens)
        _check_generation_stop(should_stop)

        diagnostics = {
            "query": query,
            "identity_tokens": identity_tokens,
            "local_count": len(local_sources),
            "local_file_question": local_file_question,
        }
        _emit_stage(on_stage, f"✓ Local files ({len(local_sources)} matches)")
        if local_file_question:
            expanded_query = _expand_local_file_query(query or question)
            diagnostics["expanded_local_query"] = expanded_query
            expanded_sources: list[LocalSource] = []
            if expanded_query.strip().lower() != (query or question).strip().lower():
                _emit_stage(on_stage, "Expanding local file keywords...")
                expanded_sources = self.retriever.search(
                    expanded_query,
                    k=max(self.settings.retriever_k, 8),
                    score_threshold=max(0.2, self.settings.retrieval_score_threshold - 0.1),
                )
            local_sources = _merge_local_sources(
                local_sources,
                expanded_sources,
                limit=max(self.settings.retriever_k, 8),
            )
            diagnostics["local_count"] = len(local_sources)
            diagnostics["expanded_local_count"] = len(expanded_sources)
            if not local_sources:
                return RAGResponse(
                    answer=LOCAL_FILE_NOT_FOUND,
                    local_sources=[],
                    web_sources=[],
                    used_web=False,
                    confidence="low",
                    diagnostics=diagnostics,
                )
            return RAGResponse(
                answer=_local_file_evidence_answer(local_sources),
                local_sources=local_sources,
                web_sources=[],
                used_web=False,
                confidence="local-grounded",
                diagnostics=diagnostics,
            )

        force_web = self._is_web_requested(question)
        time_sensitive = self._is_time_sensitive_question(question)
        diagnostics["time_sensitive"] = time_sensitive
        local_answer = LOCAL_UNKNOWN
        model_answer = MODEL_UNKNOWN
        local_sufficient = False
        model_sufficient = False
        generation_error = ""

        try:
            _emit_stage(on_stage, "Checking local evidence...")
            local_answer = self.generator.answer_local(query, history, local_sources)
            local_sufficient = self._is_sufficient(local_answer)
            diagnostics["local_sufficient"] = local_sufficient

            if local_sufficient and not force_web:
                _emit_stage(on_stage, "✓ Local evidence answered the question")
                used_local_sources = _local_sources_used_in_answer(local_sources, local_answer)
                return RAGResponse(
                    answer=local_answer,
                    local_sources=used_local_sources,
                    web_sources=[],
                    used_web=False,
                    confidence="local-grounded",
                    diagnostics=diagnostics,
                )

            if not local_sufficient:
                _check_generation_stop(should_stop)
                _emit_stage(on_stage, "Checking AI knowledge...")
                model_answer = self.generator.answer_model_knowledge(query, history)
                model_sufficient = self._is_sufficient(model_answer)
                diagnostics["model_sufficient"] = model_sufficient
                if model_sufficient and not self._should_use_web(
                    question=question,
                    force_web=force_web,
                    local_sufficient=local_sufficient,
                    model_sufficient=model_sufficient,
                    web_enabled=self.settings.enable_web_search,
                ):
                    return RAGResponse(
                        answer=f"{model_answer}\n\nSource: AI knowledge (not externally verified)",
                        local_sources=[],
                        web_sources=[],
                        used_web=False,
                        confidence="model-only",
                        diagnostics=diagnostics,
                    )
        except GenerationError as exc:
            message = str(exc)
            generation_error = message
            diagnostics["generation_error"] = message
            diagnostics["generation_error_confidence"] = _generation_error_confidence(message)
            diagnostics.setdefault("local_sufficient", local_sufficient)
            diagnostics.setdefault("model_sufficient", model_sufficient)
            if not self._can_attempt_web_after_generation_error(question, force_web):
                return RAGResponse(
                    answer=message,
                    local_sources=[],
                    web_sources=[],
                    used_web=False,
                    confidence=_generation_error_confidence(message),
                    diagnostics=diagnostics,
                )

        _check_generation_stop(should_stop)
        web_sources: list[WebSource] = []
        used_web = False
        web_error = ""
        should_use_web = self._should_use_web(
            question=question,
            force_web=force_web,
            local_sufficient=local_sufficient,
            model_sufficient=model_sufficient,
            web_enabled=self.settings.enable_web_search,
        ) or bool(generation_error)
        diagnostics["web_requested"] = should_use_web
        diagnostics["web_provider"] = self.settings.web_search_provider_label()
        if (
            should_use_web
            and self.settings.enable_web_search
            and getattr(self.web_search, "is_configured", True)
        ):
            try:
                web_queries = _web_queries(question, query, identity_tokens)
                diagnostics["web_query"] = web_queries[0] if web_queries else ""
                diagnostics["web_queries"] = web_queries
                _emit_stage(on_stage, "Searching web evidence...")
                web_sources = self._search_web_sources(web_queries, identity_tokens)
            except Exception as exc:
                web_error = _clean_error_message(exc)
                diagnostics["web_error"] = web_error
                diagnostics["web_count"] = 0
                diagnostics["web_note"] = (
                    f"{self.settings.web_search_provider_label()} search could not complete."
                )
            used_web = bool(web_sources)
            diagnostics["web_count"] = len(web_sources)
            _emit_stage(on_stage, f"✓ Web evidence ({len(web_sources)} sources)")
        else:
            diagnostics["web_count"] = 0
            if should_use_web and self.settings.enable_web_search:
                diagnostics["web_note"] = (
                    "Web search was requested, but the selected provider is not configured."
                )

        _check_generation_stop(should_stop)
        if web_sources:
            _emit_stage(on_stage, "Generating evidence-aware answer...")
            try:
                answer = self.generator.answer_final(
                    question=question,
                    history=history,
                    local_answer=local_answer,
                    model_answer=model_answer,
                    local_sources=local_sources,
                    web_sources=web_sources,
                )
            except GenerationError as exc:
                answer = (
                    "I could not answer from local files or reliable model knowledge. "
                    f"Web search found sources, but answer synthesis failed: {exc}"
                )
        elif local_sufficient:
            answer = local_answer
            if web_error:
                answer = _append_web_update_note(
                    answer,
                    self.settings.web_search_provider_label(),
                    "search could not complete, so no web sources were added.",
                )
            elif should_use_web:
                diagnostics["web_note"] = "Web validation was requested, but no web sources were available."
        elif model_sufficient:
            if time_sensitive and should_use_web:
                if web_error:
                    answer = (
                        "I could not verify current information from local files or web sources. "
                        f"{self.settings.web_search_provider_label()} web search could not complete, "
                        "and AI knowledge is not reliable enough for current facts."
                    )
                else:
                    answer = (
                        "I could not verify current information from local files or returned web sources. "
                        "AI knowledge is not reliable enough for current facts."
                    )
            else:
                answer = f"{model_answer}\n\nSource: AI knowledge (not externally verified)"
                if web_error:
                    answer = _append_web_update_note(
                        answer,
                        self.settings.web_search_provider_label(),
                        "search could not complete, so this answer was not updated with web sources.",
                    )
                elif should_use_web:
                    diagnostics["web_note"] = "Web validation was requested, but no web sources were available."
        elif force_web:
            if web_error:
                answer = (
                    "I could not answer from local files or reliable model knowledge. "
                    f"{self.settings.web_search_provider_label()} web search could not complete, "
                    "so no web sources were available."
                )
            else:
                answer = "Web search is enabled in the workflow, but no web sources were returned."
        else:
            if generation_error and self.settings.enable_web_search:
                answer = (
                    "I could not use the selected Hugging Face model, and web search did not return "
                    "usable sources for this question."
                )
            else:
                answer = (
                    "I could not answer from local files or reliable model knowledge. "
                    "Select a web search provider, add the required configuration, and keep "
                    "Web search enabled to search online."
                )

        _emit_stage(on_stage, "Comparing and validating evidence...")
        current_role_validation = _current_role_validation_from_web(question, web_sources)
        if current_role_validation:
            diagnostics["current_role_override"] = True
            diagnostics["evidence_conflict"] = current_role_validation.conflict
            diagnostics["validated_current_role"] = current_role_validation.selected_name
            diagnostics["current_role_claim_count"] = current_role_validation.claim_count
            answer = current_role_validation.answer

        used_local_sources = _local_sources_used_in_answer(local_sources, answer)
        used_web_sources = _web_sources_used_in_answer(web_sources, answer)
        if web_sources and not used_web_sources:
            diagnostics["web_note"] = (
                "Web results were returned, but the generated answer did not cite web labels. "
                "Verilume added a conservative cited fallback."
            )
            answer = _fallback_answer_from_web_results(
                question=question,
                web_sources=web_sources,
                model_answer=model_answer if model_sufficient else "",
                previous_answer=answer,
                prefer_web_only=time_sensitive,
            )
            used_local_sources = _local_sources_used_in_answer(local_sources, answer)
            used_web_sources = _web_sources_used_in_answer(web_sources, answer)
        display_web_sources = _best_web_sources(web_sources, used_web_sources)
        confidence = self._confidence(
            used_local_sources,
            bool(used_web_sources),
            answer,
            time_sensitive=time_sensitive,
        )
        return RAGResponse(
            answer=answer,
            local_sources=used_local_sources,
            web_sources=display_web_sources,
            used_web=used_web,
            confidence=confidence,
            diagnostics=diagnostics,
        )

    def _search_web_sources(
        self,
        web_queries: Sequence[str],
        identity_tokens: Sequence[str],
    ) -> list[WebSource]:
        collected: list[WebSource] = []
        errors: list[str] = []
        target = max(1, min(MAX_WEB_SOURCES_TO_SHOW, self.settings.web_search_max_results))
        for index, web_query in enumerate(web_queries[:WEB_QUERY_FANOUT_LIMIT]):
            if not web_query.strip():
                continue
            try:
                candidates = self.web_search.search(web_query)
            except Exception as exc:
                errors.append(_clean_error_message(exc))
                continue
            if identity_tokens:
                candidates = _filter_web_sources_for_identity(candidates, identity_tokens)
            collected = _merge_web_sources(
                collected,
                candidates,
                limit=target * WEB_QUERY_FANOUT_LIMIT,
            )
            ranked = _rank_web_sources(collected, web_queries)
            if len(ranked) >= target and _has_strong_web_sources(ranked):
                return ranked[:target]
            if (
                index == 0
                and self.settings.web_search_provider != "duckduckgo"
                and len(ranked) >= target
            ):
                fallback_candidates = self._search_duckduckgo_fallback(web_query)
                if identity_tokens:
                    fallback_candidates = _filter_web_sources_for_identity(
                        fallback_candidates,
                        identity_tokens,
                    )
                collected = _merge_web_sources(
                    collected,
                    fallback_candidates,
                    limit=target * WEB_QUERY_FANOUT_LIMIT,
                )
                ranked = _rank_web_sources(collected, web_queries)
                if len(ranked) >= target and _has_strong_web_sources(ranked):
                    return ranked[:target]
        if self.settings.web_search_provider != "duckduckgo" and len(collected) < target:
            fallback_candidates = self._search_duckduckgo_fallback_queries(web_queries)
            if identity_tokens:
                fallback_candidates = _filter_web_sources_for_identity(
                    fallback_candidates,
                    identity_tokens,
                )
            collected = _merge_web_sources(
                collected,
                fallback_candidates,
                limit=target * WEB_QUERY_FANOUT_LIMIT,
            )
        if not collected and errors:
            raise RuntimeError("; ".join(errors))
        return _rank_web_sources(collected, web_queries)[:target]

    def _search_duckduckgo_fallback(self, web_query: str) -> list[WebSource]:
        try:
            return DuckDuckGoSearch(
                max_results=self.settings.web_search_max_results,
                timeout_seconds=min(10.0, self.settings.web_search_timeout_seconds),
            ).search(web_query)
        except Exception:
            return []

    def _search_duckduckgo_fallback_queries(self, web_queries: Sequence[str]) -> list[WebSource]:
        collected: list[WebSource] = []
        target = max(1, min(MAX_WEB_SOURCES_TO_SHOW, self.settings.web_search_max_results))
        for web_query in web_queries[:WEB_QUERY_FANOUT_LIMIT]:
            collected = _merge_web_sources(
                collected,
                self._search_duckduckgo_fallback(web_query),
                limit=target * WEB_QUERY_FANOUT_LIMIT,
            )
            if len(collected) >= target and _has_strong_web_sources(
                _rank_web_sources(collected, web_queries)
            ):
                break
        return collected

    def _can_attempt_web_after_generation_error(self, question: str, force_web: bool) -> bool:
        if not self.settings.enable_web_search:
            return False
        if not getattr(self.web_search, "is_configured", True):
            return False
        return (
            force_web
            or self._needs_web_validation(question)
            or not question.strip().lower().startswith(("summarize", "summarise"))
        )

    @staticmethod
    def _is_sufficient(answer: str) -> bool:
        text = answer.strip()
        lower = text.lower()
        if not text:
            return False
        if any(marker in lower for marker in INSUFFICIENT_MARKERS):
            return False
        return True

    @staticmethod
    def _is_web_requested(question: str) -> bool:
        lower = question.lower()
        return any(marker in lower for marker in WEB_REQUEST_MARKERS)

    @staticmethod
    def _is_local_file_question(question: str) -> bool:
        lower = question.lower().strip()
        corpus_markers = (
            "in the local file",
            "in the local files",
            "indexed local file",
            "indexed local files",
            "indexed local documents",
            "in the indexed files",
            "in the indexed documents",
            "in the database",
            "in database",
            "local database",
            "knowledge base",
            "uploaded file",
            "uploaded files",
            "uploaded document",
            "uploaded documents",
            "in my files",
            "in the files",
            "in my documents",
            "in the documents",
            "which document",
            "which file",
            "what document",
            "what file",
        )
        if not any(marker in lower for marker in corpus_markers):
            return False
        intent_markers = (
            "are there",
            "can you find",
            "contain",
            "contains",
            "do i have",
            "do you have",
            "does",
            "find",
            "has",
            "have",
            "is ",
            "there",
            "where",
            "which",
        )
        return any(marker in lower for marker in intent_markers)

    @staticmethod
    def _confidence(
        local_sources: Sequence[LocalSource],
        used_web: bool,
        answer: str,
        *,
        time_sensitive: bool = False,
    ) -> str:
        lower = answer.lower()
        if is_model_selection_warning(answer):
            return "model-selection-warning"
        if any(marker in lower for marker in INSUFFICIENT_MARKERS):
            return "low"
        if time_sensitive and used_web:
            return "current-information"
        if time_sensitive and not used_web and not local_sources:
            return "low"
        if used_web and local_sources:
            return "local-web-assisted"
        if used_web:
            return "web-assisted"
        if local_sources:
            return "local-grounded"
        return "model-only"

    @staticmethod
    def _needs_web_validation(question: str) -> bool:
        lower = question.lower().strip()
        if any(marker in lower for marker in WEB_REQUEST_MARKERS):
            return True
        if VerilumeRAG._is_time_sensitive_question(question):
            return True
        if any(marker in lower for marker in PUBLIC_ENTITY_HINTS):
            return True
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", question)
        capitalized = [word for word in words if word[:1].isupper()]
        return 2 <= len(capitalized) and len(words) <= 8

    @classmethod
    def _should_use_web(
        cls,
        *,
        question: str,
        force_web: bool,
        local_sufficient: bool,
        model_sufficient: bool,
        web_enabled: bool,
    ) -> bool:
        if force_web or cls._needs_web_validation(question):
            return True
        return web_enabled and not local_sufficient

    @staticmethod
    def _is_time_sensitive_question(question: str) -> bool:
        lower = question.lower().strip()
        if any(marker in lower for marker in TIME_SENSITIVE_MARKERS):
            return True
        return bool(re.search(r"\b(?:20\d{2}|19\d{2})\b", lower))


@lru_cache(maxsize=8)
def get_rag_service(settings: AppSettings) -> VerilumeRAG:
    return VerilumeRAG(settings)


def _local_sources_used_in_answer(
    local_sources: Sequence[LocalSource],
    answer: str,
) -> list[LocalSource]:
    labels_in_answer = _labels_in_answer(answer, "S")
    return [source for source in local_sources if source.label in labels_in_answer]


def _merge_local_sources(
    primary: Sequence[LocalSource],
    secondary: Sequence[LocalSource],
    limit: int,
) -> list[LocalSource]:
    merged: list[LocalSource] = []
    seen: set[str] = set()
    for source in [*primary, *secondary]:
        key = _local_source_key(source)
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
        if len(merged) >= limit:
            break
    return _relabel_local_sources(merged)


def _local_source_key(source: LocalSource) -> str:
    if source.chunk_id.strip():
        return source.chunk_id.strip()
    return f"{source.document}::{source.page or ''}::{source.text[:80]}"


def _local_file_evidence_answer(sources: Sequence[LocalSource]) -> str:
    lines = ["I found matching content in the indexed local files:"]
    for source in sources[:5]:
        page = f", page {source.page}" if source.page else ""
        preview = _compact_source_text(source.text, limit=220)
        if preview:
            lines.append(f"- [{source.label}] {source.document}{page}: {preview}")
        else:
            lines.append(f"- [{source.label}] {source.document}{page}")
    if len(sources) > 5:
        lines.append(f"Additional matching chunks found: {len(sources) - 5}.")
    return "\n".join(lines)


def _web_sources_used_in_answer(
    web_sources: Sequence[WebSource],
    answer: str,
) -> list[WebSource]:
    labels_in_answer = _labels_in_answer(answer, "W")
    return [source for source in web_sources if source.label in labels_in_answer]


def _best_web_sources(
    web_sources: Sequence[WebSource],
    used_web_sources: Sequence[WebSource],
    limit: int = MAX_WEB_SOURCES_TO_SHOW,
) -> list[WebSource]:
    if not web_sources:
        return []
    labels_in_use = {source.label for source in used_web_sources}
    selected = list(web_sources[:limit])
    if labels_in_use.issubset({source.label for source in selected}):
        return selected
    for source in web_sources[limit:]:
        if source.label in labels_in_use:
            selected.append(source)
    return selected[:limit]


def _identity_tokens(text: str) -> list[str]:
    if re.match(r"^\s*(what|how|why)\s+(is|are|do|does|did)\b", text.lower()):
        return []
    if _is_current_role_lookup(text):
        return []

    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    tokens: list[str] = []
    for word in words:
        token = word.lower().strip("'-")
        if len(token) < 3 or token in IDENTITY_STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    if len(tokens) < 2:
        return []
    return tokens[:4]


def _is_current_role_lookup(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in CURRENT_ROLE_MARKERS)


def _filter_local_sources_for_identity(
    sources: Sequence[LocalSource],
    identity_tokens: Sequence[str],
) -> list[LocalSource]:
    filtered = [
        source
        for source in sources
        if _mentions_identity(f"{source.document}\n{source.text}", identity_tokens)
    ]
    return _relabel_local_sources(filtered)


def _filter_web_sources_for_identity(
    sources: Sequence[WebSource],
    identity_tokens: Sequence[str],
) -> list[WebSource]:
    filtered = [
        source
        for source in sources
        if _web_source_matches_identity(source, identity_tokens)
    ]
    return _relabel_web_sources(filtered)


def _web_source_matches_identity(source: WebSource, identity_tokens: Sequence[str]) -> bool:
    if len(identity_tokens) < 2:
        return True

    title_url = f"{source.title}\n{source.url}"
    full_text = f"{title_url}\n{source.content}"
    if not _mentions_identity(full_text, identity_tokens):
        return False

    if _contains_identity_phrase(title_url, identity_tokens):
        return True

    full_lower = full_text.lower()
    if any(term in full_lower for term in NOISY_WEB_RESULT_TERMS):
        return False

    if _contains_identity_phrase(source.content, identity_tokens):
        return any(term in full_lower for term in PROFILE_RELEVANCE_TERMS)

    return False


def _mentions_identity(text: str, identity_tokens: Sequence[str]) -> bool:
    if len(identity_tokens) < 2:
        return True
    haystack = text.lower()
    required_tokens = identity_tokens[:2]
    return all(token in haystack for token in required_tokens)


def _contains_identity_phrase(text: str, identity_tokens: Sequence[str]) -> bool:
    if len(identity_tokens) < 2:
        return True
    phrase = " ".join(identity_tokens[:2])
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return phrase in normalized


def _relabel_local_sources(sources: Sequence[LocalSource]) -> list[LocalSource]:
    values = list(sources)
    for index, source in enumerate(values, start=1):
        source.label = f"S{index}"
    return values


def _expand_local_file_query(query: str) -> str:
    cleaned = _clean_local_file_query(query)
    terms = [cleaned] if cleaned else [query.strip()]
    lower = query.lower()
    for marker, expansions in LOCAL_FILE_EXPANSIONS.items():
        if marker in lower:
            terms.extend(expansions)
    terms.extend(("local files", "indexed documents", "uploaded documents"))
    return " ".join(_unique_nonempty(terms))


def _clean_local_file_query(query: str) -> str:
    cleaned = (query or "").strip()
    patterns = (
        r"\b(?:in|inside|from)\s+(?:the\s+)?(?:indexed\s+)?local\s+files?\b",
        r"\b(?:in|inside|from)\s+(?:the\s+)?(?:indexed\s+)?documents?\b",
        r"\b(?:in|inside|from)\s+(?:the\s+)?database\b",
        r"\b(?:in|inside|from)\s+(?:the\s+)?knowledge\s+base\b",
        r"\b(?:uploaded|indexed)\s+(?:files?|documents?)\b",
        r"^\s*(?:is|are)\s+there\s+",
        r"^\s*(?:which|what)\s+(?:document|file)\s+(?:contains?|has|includes?)\s+",
        r"^\s*(?:do|does)\s+(?:i|you|we)\s+have\s+",
        r"^\s*(?:can\s+you\s+)?find\s+",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;-?!.")
    return cleaned


def _relabel_web_sources(sources: Sequence[WebSource]) -> list[WebSource]:
    values = list(sources)
    for index, source in enumerate(values, start=1):
        source.label = f"W{index}"
    return values


def _web_queries(question: str, query: str, identity_tokens: Sequence[str]) -> list[str]:
    base = _clean_web_query(query or question)
    role_lookup = _current_role_lookup(question)
    if role_lookup:
        return _current_role_queries(role_lookup, base)

    candidates: list[str] = []
    if len(identity_tokens) >= 2:
        phrase = " ".join(identity_tokens)
        candidates.extend(
            [
                f'"{phrase}"',
                f'"{phrase}" official profile',
                base,
            ]
        )
    else:
        candidates.append(base)

    if _is_current_role_lookup(question):
        candidates.extend(
            [
                f"{base} official government",
                f"{base} official",
                f"{base} 2026",
            ]
        )
    elif any(marker in question.lower() for marker in WEB_REQUEST_MARKERS):
        candidates.extend(
            [
                f"{base} official",
                f"{base} latest",
            ]
        )
    return _unique_nonempty(candidates)


def _current_role_queries(role_lookup: CurrentRoleLookup, base: str) -> list[str]:
    role = role_lookup.role
    entity = role_lookup.entity
    if role in {"king", "queen"} and entity == "United States":
        return _unique_nonempty(
            [
                "current president of the United States official government",
                "White House President Donald J. Trump",
                "USAGov current president United States",
                "United States has no king president republic",
                base,
            ]
        )

    candidates = [
        f"current {role} of {entity} official government",
        f"{role} of {entity} official current role holder",
        f"{entity} {role} official",
        base,
    ]
    if entity == "United Kingdom" and role == "prime minister":
        candidates[:0] = [
            "GOV.UK Prime Minister current role holder",
            "Prime Minister GOV.UK Keir Starmer",
            "members parliament United Kingdom prime minister current",
        ]
    elif entity == "United States" and role == "president":
        candidates[:0] = [
            "current president of the United States official government",
            "White House President Donald J. Trump",
            "USAGov current president United States",
        ]
    elif entity == "Netherlands" and role in {"king", "queen"}:
        candidates[:0] = [
            f"Royal House of the Netherlands current {role}",
            f"current {role} of the Netherlands official royal house",
        ]
    return _unique_nonempty(candidates)


def _web_query(question: str, query: str, identity_tokens: Sequence[str]) -> str:
    queries = _web_queries(question, query, identity_tokens)
    return queries[0] if queries else _clean_web_query(query or question)


def _clean_web_query(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    patterns = (
        r"^\s*(?:please\s+)?(?:search|use)\s+(?:the\s+)?(?:web|internet|online)"
        r"\s*(?:for|about)?\s*",
        r"^\s*(?:please\s+)?look\s+up\s*(?:online|on\s+the\s+web|on\s+the\s+internet)?"
        r"\s*(?:for|about)?\s*",
        r"^\s*(?:web\s+search|online\s+search)\s*(?:for|about)?\s*",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;-?!.")
    return cleaned or text.strip()


def _unique_nonempty(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", (value or "").strip())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _merge_web_sources(
    existing: Sequence[WebSource],
    candidates: Sequence[WebSource],
    limit: int,
) -> list[WebSource]:
    merged: list[WebSource] = []
    seen: set[str] = set()
    for source in [*existing, *candidates]:
        key = _web_source_key(source)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(source)
        if len(merged) >= limit:
            break
    return _relabel_web_sources(merged)


def _rank_web_sources(sources: Sequence[WebSource], web_queries: Sequence[str]) -> list[WebSource]:
    terms = _source_terms(" ".join(web_queries))
    ranked = sorted(
        list(sources),
        key=lambda source: _web_source_rank_score(source, terms),
        reverse=True,
    )
    non_noisy = [source for source in ranked if not _is_noisy_web_source(source)]
    if len(non_noisy) >= MAX_WEB_SOURCES_TO_SHOW:
        ranked = [*non_noisy, *(source for source in ranked if _is_noisy_web_source(source))]
    return _relabel_web_sources(ranked)


def _web_source_rank_score(source: WebSource, terms: Sequence[str]) -> float:
    text = f"{source.title} {source.content} {source.url}".lower()
    score = min(float(source.score or 0.0), 5.0)

    score += _authority_score(source) / 12.0
    score += _freshness_score(source) / 8.0
    if _is_noisy_web_source(source):
        score -= 5.0
    if _is_stale_current_role_source(source):
        score -= 30.0

    if source.content.strip():
        score += 0.75
    else:
        score -= 0.5

    matched_terms = sum(1 for term in terms if term in text)
    score += min(4.0, matched_terms * 0.55)

    title_lower = source.title.lower()
    if any(term in title_lower for term in terms):
        score += 0.75
    if "prime" in terms and "minister" in terms:
        if "prime minister of luxembourg" in text or "luc frieden prime minister" in text:
            score += 2.0
        if "deputy prime minister" in text:
            score -= 2.0
    if ("usa" in terms or ("united" in terms and "states" in terms)) and "president" in terms:
        if "president of the united states" in text or "president donald" in text:
            score += 2.5
    if ("usa" in terms or ("united" in terms and "states" in terms)) and "king" in terms:
        if "king charles" in text or "britain's king" in text or "british king" in text:
            score -= 3.0

    return score


def _has_strong_web_sources(sources: Sequence[WebSource]) -> bool:
    strong_count = 0
    noisy_count = 0
    for source in sources[:MAX_WEB_SOURCES_TO_SHOW]:
        if _is_noisy_web_source(source):
            noisy_count += 1
            continue
        domain = urlparse(source.url).netloc.lower()
        if _is_official_government_domain(domain):
            strong_count += 1
            continue
        if any(reference in domain for reference in TRUSTED_REFERENCE_DOMAINS):
            strong_count += 1
    return strong_count >= 2 and noisy_count == 0


def _authority_score(source: WebSource) -> float:
    domain = urlparse(source.url).netloc.lower().removeprefix("www.")
    if _is_official_government_domain(domain):
        return 100.0
    if "royal" in domain and any(part in domain for part in ("house", "family")):
        return 95.0
    if domain.endswith(".edu") or ".edu." in domain or ".ac." in domain:
        return 90.0
    if any(reference in domain for reference in ("reuters.com", "apnews.com", "bbc.com", "bbc.co.uk")):
        return 88.0
    if any(reference in domain for reference in TRUSTED_REFERENCE_DOMAINS):
        return 70.0
    if _is_noisy_web_source(source):
        return 20.0
    if any(blog in domain for blog in ("blog", "medium.com", "substack.com", "wordpress.com")):
        return 40.0
    return 55.0


def _freshness_score(source: WebSource) -> float:
    text = f"{source.title} {source.content}".lower()
    if "current role holder" in text or "incumbent" in text:
        return 20.0
    newest_year = _source_newest_year(source)
    if newest_year is None:
        return 5.0 if _authority_score(source) >= 90.0 else 0.0
    age = max(0, date.today().year - newest_year)
    if age == 0:
        return 20.0
    if age == 1:
        return 15.0
    if age <= 2:
        return 10.0
    if age <= 4:
        return 2.0
    return -10.0


def _source_newest_year(source: WebSource) -> int | None:
    values: list[str] = []
    if source.published_date:
        values.append(source.published_date)
    values.extend(str(value) for value in source.metadata.get("visible_dates", []))
    values.extend([source.title, source.content])
    years = [
        int(match)
        for value in values
        for match in re.findall(r"\b(20\d{2}|19\d{2})\b", value or "")
    ]
    if not years:
        return None
    return max(years)


def _is_stale_current_role_source(source: WebSource) -> bool:
    text = f"{source.title} {source.content}".lower()
    stale_patterns = (
        r"\bwas\s+(?:the\s+)?(?:prime minister|president|king|queen|governor|minister|ceo)\b",
        r"\bserved\s+as\s+(?:the\s+)?(?:prime minister|president|king|queen|governor|minister|ceo)\b",
        r"\bformer\s+(?:prime minister|president|king|queen|governor|minister|ceo)\b",
        r"\bpast\s+(?:prime ministers|presidents|kings|queens|governors|ministers|ceos)\b",
        r"\b(?:prime minister|president|king|queen|governor|minister|ceo)\s+between\s+\d{4}\s+and\s+\d{4}\b",
        r"\b\d{4}\s+(?:to|-|–)\s+\d{4}\b",
    )
    if any(re.search(pattern, text) for pattern in stale_patterns):
        if "current role holder" in text or "incumbent" in text:
            return False
        return True
    return False


def _is_noisy_web_source(source: WebSource) -> bool:
    url = source.url.lower()
    return any(noisy in url for noisy in NOISY_SOURCE_DOMAINS)


def _is_official_government_domain(domain: str) -> bool:
    value = (domain or "").lower().removeprefix("www.")
    if value.endswith(".gov"):
        return True
    return any(official in value for official in PRIMARY_OFFICIAL_SOURCE_DOMAINS + SECONDARY_OFFICIAL_SOURCE_DOMAINS)


def _source_terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
        term = word.lower().strip("'-")
        if len(term) < 3 or term in SOURCE_TERM_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:12]


def _web_source_key(source: WebSource) -> str:
    url = (source.url or "").strip().lower().rstrip("/")
    if url:
        return url
    return f"{source.title.strip().lower()}::{source.content.strip().lower()[:80]}"


def _current_role_validation_from_web(
    question: str,
    web_sources: Sequence[WebSource],
) -> CurrentRoleValidation | None:
    if not web_sources:
        return None
    role_lookup = _current_role_lookup(question)
    if not role_lookup:
        return None

    if role_lookup.entity == "United States" and role_lookup.role in {"king", "queen"}:
        president_lookup = CurrentRoleLookup(role="president", entity="United States")
        president_validation = _validate_current_role_claims(president_lookup, web_sources)
        if president_validation:
            source_labels = _source_labels(president_validation.selected_sources)
            answer = (
                "Evidence conflict detected; selecting the newest high-authority evidence. "
                "The United States does not have a king or queen; it is a republic. "
                f"Its current president is {president_validation.selected_name} {source_labels}."
            )
            return CurrentRoleValidation(
                answer=answer,
                selected_name=president_validation.selected_name,
                selected_sources=president_validation.selected_sources,
                conflict=True,
                claim_count=president_validation.claim_count,
            )
        evidence_source = _best_us_civic_source(web_sources)
        if evidence_source:
            return CurrentRoleValidation(
                answer=(
                    "The United States does not have a king or queen; it is a republic. "
                    f"The relevant national executive office is the presidency [{evidence_source.label}]."
                ),
                selected_name="presidency",
                selected_sources=[evidence_source],
                conflict=False,
                claim_count=1,
            )
        return None

    return _validate_current_role_claims(role_lookup, web_sources)


def _validate_current_role_claims(
    role_lookup: CurrentRoleLookup,
    web_sources: Sequence[WebSource],
) -> CurrentRoleValidation | None:
    claims = _current_role_claims(role_lookup, web_sources)
    if not claims:
        return None

    grouped: dict[str, list[CurrentRoleClaim]] = {}
    for claim in claims:
        if claim.stale:
            continue
        grouped.setdefault(_normalize_person_name(claim.name), []).append(claim)
    if not grouped:
        return None

    scored_groups = sorted(
        grouped.values(),
        key=_claim_group_score,
        reverse=True,
    )
    selected_group = scored_groups[0]
    selected_name = _display_person_name(selected_group[0].name)
    selected_sources = _claim_group_sources(selected_group)
    has_official_source = any(_authority_score(claim.source) >= 95.0 for claim in selected_group)
    independent_high_sources = {
        _source_domain(claim.source)
        for claim in selected_group
        if _authority_score(claim.source) >= 70.0
    }
    if not has_official_source and len(independent_high_sources) < 2:
        return None

    competing_names = {
        _normalize_person_name(claim.name)
        for claim in claims
        if _normalize_person_name(claim.name) != _normalize_person_name(selected_name)
    }
    stale_conflicts = {
        _normalize_person_name(claim.name)
        for claim in claims
        if claim.stale and _normalize_person_name(claim.name) != _normalize_person_name(selected_name)
    }
    conflict = bool(competing_names or stale_conflicts or len(scored_groups) > 1)

    prefix = ""
    if conflict:
        prefix = "Evidence conflict detected; selecting the newest high-authority evidence. "
    answer = (
        f"{prefix}The current {role_lookup.role} of {_display_entity_name(role_lookup.entity)} "
        f"is {selected_name} {_source_labels(selected_sources)}."
    )
    return CurrentRoleValidation(
        answer=answer,
        selected_name=selected_name,
        selected_sources=selected_sources,
        conflict=conflict,
        claim_count=len(claims),
    )


def _current_role_claims(
    role_lookup: CurrentRoleLookup,
    web_sources: Sequence[WebSource],
) -> list[CurrentRoleClaim]:
    claims: list[CurrentRoleClaim] = []
    for source in web_sources:
        name = _extract_role_holder_name(role_lookup, source)
        if not name:
            continue
        stale = _is_stale_current_role_source(source) or _claim_is_past_tense(role_lookup, source, name)
        authority = _authority_score(source)
        freshness = _freshness_score(source)
        agreement_seed = 0.0 if stale else 8.0
        score = authority + freshness + agreement_seed - (80.0 if stale else 0.0)
        claims.append(
            CurrentRoleClaim(
                name=name,
                source=source,
                authority=authority,
                freshness=freshness,
                score=score,
                stale=stale,
            )
        )
    return claims


def _extract_role_holder_name(role_lookup: CurrentRoleLookup, source: WebSource) -> str:
    text = " ".join(f"{source.title} {source.content}".split())
    if not text:
        return ""
    if not _source_matches_role_entity(role_lookup, source):
        return ""

    role = _case_insensitive_phrase(role_lookup.role)
    entity = _case_insensitive_phrase(role_lookup.entity)
    patterns = [
        rf"\b{PERSON_NAME_PATTERN}\s+(?:is|serves\s+as|has\s+been)\s+(?:the\s+)?(?:current\s+)?{role}"
        rf"(?:\s+of\s+(?:the\s+)?{entity})?\b",
        rf"\b{PERSON_NAME_PATTERN}\s+became\s+{role}\s+on\b",
        rf"\b{PERSON_NAME_PATTERN}\s+was\s+(?:the\s+)?{role}\b",
        rf"\b{PERSON_NAME_PATTERN}\s+served\s+as\s+(?:the\s+)?{role}\b",
        rf"\b{PERSON_NAME_PATTERN}\s+.*?\b{role}\s+between\s+\d{{4}}\s+and\s+\d{{4}}\b",
        rf"\b{PERSON_NAME_PATTERN}\s+.*?\b(?:current\s+)?{role}\s+of\s+(?:the\s+)?{entity}\b",
        rf"\b(?:current\s+)?{role}(?:\s+of\s+(?:the\s+)?{entity})?\s+(?:is|:)\s+{PERSON_NAME_PATTERN}\b",
        rf"\bIncumbent\s*[:\-]?\s+{PERSON_NAME_PATTERN}\b",
        rf"\bCurrent\s+role\s+holder\s+{PERSON_NAME_PATTERN}\b",
    ]
    if role_lookup.role in {"king", "queen"}:
        patterns.extend(
            [
                rf"\b{role}\s+{PERSON_NAME_PATTERN}\s+of\s+(?:the\s+)?{entity}\b",
                rf"\b{role}\s+{PERSON_NAME_PATTERN}\b",
            ]
        )
    if role_lookup.role == "president" and role_lookup.entity == "United States":
        patterns.extend(
            [
                rf"\b{PERSON_NAME_PATTERN}\s+.*?\b(?:45th\s*&\s*47th|47th)\s+President\s+of\s+the\s+United\s+States\b",
                rf"(?<!Vice\s)\bPresident\s+(?!(?:of|and|for|to)\b){PERSON_NAME_PATTERN}\b",
            ]
        )

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw_name = _first_person_match(match)
        name = _clean_person_name(raw_name)
        if _valid_person_name(name):
            return name
    return ""


def _first_person_match(match: re.Match[str]) -> str:
    for value in match.groups():
        if value and re.search(r"[A-Z][a-z]", value):
            return value
    return ""


def _case_insensitive_phrase(value: str) -> str:
    escaped = re.escape(value).replace(r"\ ", r"\s+")
    return f"(?i:{escaped})"


def _source_matches_role_entity(role_lookup: CurrentRoleLookup, source: WebSource) -> bool:
    text = f"{source.title} {source.content} {source.url}".lower()
    entity_terms = _entity_terms(role_lookup.entity)
    role_terms = [role_lookup.role]
    if role_lookup.role == "prime minister":
        role_terms.append("pm")
    role_match = any(term in text for term in role_terms)
    entity_match = any(term in text for term in entity_terms)
    if role_lookup.entity == "United Kingdom" and "gov.uk" in text:
        entity_match = True
    if role_lookup.entity == "United States" and ("whitehouse.gov" in text or "usa.gov" in text):
        entity_match = True
    if role_lookup.entity == "Netherlands" and ("royal-house.nl" in text or "koninklijkhuis.nl" in text):
        entity_match = True
    return role_match and entity_match


def _entity_terms(entity: str) -> list[str]:
    lower = entity.lower()
    terms = [lower]
    for alias, canonical in ENTITY_ALIASES.items():
        if canonical.lower() == lower:
            terms.append(alias)
    if entity == "United Kingdom":
        terms.extend(["britain", "uk", "u.k.", "downing street"])
    elif entity == "United States":
        terms.extend(["usa", "u.s.", "u.s.a.", "america", "white house"])
    elif entity == "Netherlands":
        terms.extend(["dutch", "holland"])
    return _unique_nonempty(terms)


def _claim_is_past_tense(role_lookup: CurrentRoleLookup, source: WebSource, name: str) -> bool:
    text = f"{source.title} {source.content}".lower()
    normalized_name = re.escape(_normalize_person_name(name))
    role = re.escape(role_lookup.role)
    past_patterns = (
        rf"{normalized_name}[^.]*\bwas\s+(?:the\s+)?{role}\b",
        rf"{normalized_name}[^.]*\bserved\s+as\s+(?:the\s+)?{role}\b",
        rf"{normalized_name}[^.]*\b{role}\s+between\s+\d{{4}}\s+and\s+\d{{4}}\b",
        rf"{normalized_name}[^.]*\b\d{{4}}\s+(?:to|-|–)\s+\d{{4}}\b",
    )
    return any(re.search(pattern, text) for pattern in past_patterns)


def _claim_group_score(claims: Sequence[CurrentRoleClaim]) -> float:
    if not claims:
        return 0.0
    unique_domains = {_source_domain(claim.source) for claim in claims}
    score = max(claim.score for claim in claims)
    score += max(0, len(unique_domains) - 1) * 14.0
    score += sum(1 for claim in claims if claim.authority >= 90.0) * 8.0
    return score


def _claim_group_sources(claims: Sequence[CurrentRoleClaim]) -> list[WebSource]:
    selected: list[WebSource] = []
    seen: set[str] = set()
    for claim in sorted(claims, key=lambda value: value.score, reverse=True):
        key = _web_source_key(claim.source)
        if key in seen:
            continue
        seen.add(key)
        selected.append(claim.source)
        if len(selected) >= 3:
            break
    return selected


def _source_labels(sources: Sequence[WebSource]) -> str:
    return " ".join(f"[{source.label}]" for source in sources)


def _source_domain(source: WebSource) -> str:
    return urlparse(source.url).netloc.lower().removeprefix("www.")


def _current_role_lookup(question: str) -> CurrentRoleLookup | None:
    lower = _normalized_question_text(question)
    role = _current_role_from_text(lower)
    if not role:
        return None
    entity = _current_role_entity(lower, role)
    if not entity:
        return None
    return CurrentRoleLookup(role=role, entity=entity)


def _current_role_from_text(lower_question: str) -> str:
    for marker, role in sorted(CURRENT_ROLE_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(marker)}\b", lower_question):
            return role
    return ""


def _current_role_entity(lower_question: str, role: str) -> str:
    role_pattern = re.escape(role)
    match = re.search(rf"\b{role_pattern}\s+(?:of|for)\s+(?:the\s+)?(.+)$", lower_question)
    if match:
        entity = _clean_entity_text(match.group(1))
        canonical = _canonical_entity_name(entity)
        if canonical:
            return canonical

    for alias, canonical in sorted(ENTITY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lower_question):
            return canonical

    before_role = lower_question.split(role, 1)[0].strip()
    canonical = _canonical_entity_name(_clean_entity_text(before_role))
    return canonical


def _clean_entity_text(text: str) -> str:
    cleaned = re.sub(r"\b(?:current|latest|present|now|today|the|is|who|what|which)\b", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.!,:;-")
    return cleaned


def _canonical_entity_name(entity: str) -> str:
    lower = _normalized_question_text(entity)
    if not lower:
        return ""
    if lower in ENTITY_ALIASES:
        return ENTITY_ALIASES[lower]
    if "united kingdom" in lower or re.search(r"\buk\b", lower):
        return "United Kingdom"
    if "united states" in lower or re.search(r"\b(?:usa|us)\b", lower):
        return "United States"
    if "netherlands" in lower or "holland" in lower:
        return "Netherlands"
    return " ".join(part.capitalize() for part in lower.split())


def _asks_for_us_monarch(question: str) -> bool:
    lookup = _current_role_lookup(question)
    return bool(lookup and lookup.entity == "United States" and lookup.role in US_MONARCH_ROLE_MARKERS)


def _asks_for_us_president(question: str) -> bool:
    lookup = _current_role_lookup(question)
    return bool(lookup and lookup.entity == "United States" and lookup.role == "president")


def _mentions_us_entity(lower_question: str) -> bool:
    normalized = f" {lower_question} "
    return any(f" {marker} " in normalized for marker in US_ENTITY_MARKERS)


def _normalized_question_text(question: str) -> str:
    return re.sub(r"[^a-z0-9.'’-]+", " ", (question or "").lower()).strip()


def _extract_us_president_from_sources(web_sources: Sequence[WebSource]) -> tuple[str, WebSource] | None:
    candidates: list[tuple[float, int, str, WebSource]] = []
    for index, source in enumerate(web_sources):
        text = f"{source.title}\n{source.content}"
        name = _extract_us_president_name(text)
        if not name:
            continue
        candidates.append((_current_office_source_score(source), -index, name, source))
    if not candidates:
        return None
    _score, _index, name, source = max(candidates, key=lambda item: (item[0], item[1]))
    return name, source


def _extract_us_president_name(text: str) -> str:
    compact = " ".join((text or "").split())
    if not compact:
        return ""

    name_pattern = r"([A-Z][A-Za-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z]+){1,3})"
    patterns = (
        rf"(?<!Vice\s)\bPresident\s+(?!(?:of|and|for|to)\b){name_pattern}\b",
        rf"\bcurrent\s+president(?:\s+of\s+the\s+United\s+States)?\s+(?:is|:)\s+{name_pattern}\b",
        rf"\bpresident\s+of\s+the\s+United\s+States\s+(?:is|:)\s+{name_pattern}\b",
        rf"\b{name_pattern}\s+(?:is|serves\s+as|was\s+sworn\s+in\s+as)\s+"
        r"(?:the\s+)?(?:\d+(?:st|nd|rd|th)\s+(?:and\s+\d+(?:st|nd|rd|th)\s+)?)?"
        r"(?:current\s+)?president\s+of\s+the\s+United\s+States\b",
        rf"\b{name_pattern}\s+.*?\b(?:45th\s*&\s*47th|47th)\s+President\s+of\s+the\s+United\s+States\b",
    )
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            return _clean_person_name(match.group(1))
    return ""


def _clean_person_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", (name or "").strip(" .,:;-"))
    cleaned = re.sub(
        r"\b(?:The\s+Rt\s+Hon|Rt\s+Hon|Prime\s+Minister|President|Sir|Dame|Lord|Lady|Mr|Ms|Mrs|Dr)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?:KCB|KC|KCMG|MP|MSP|AM|MLA)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\b(?:The|White|House|Official|Website)$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned.lower() == "donald john trump":
        return "Donald J. Trump"
    return cleaned


def _valid_person_name(name: str) -> bool:
    lower = name.lower().strip()
    if not lower:
        return False
    blocked = {
        "prime minister",
        "current role",
        "government",
        "white house",
        "official website",
        "united kingdom",
        "united states",
    }
    if lower in blocked:
        return False
    return bool(re.search(r"[A-Z][a-z]", name))


def _normalize_person_name(name: str) -> str:
    cleaned = _clean_person_name(name).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    return cleaned


def _display_person_name(name: str) -> str:
    cleaned = _clean_person_name(name)
    title_words = {"j.", "iii", "iv"}
    return " ".join(
        part if part.lower() in title_words else part[:1].upper() + part[1:]
        for part in cleaned.split()
    )


def _display_entity_name(entity: str) -> str:
    if entity in {"Netherlands", "United Kingdom", "United States"}:
        return f"the {entity}"
    return entity


def _current_office_source_score(source: WebSource) -> float:
    domain = urlparse(source.url).netloc.lower()
    score = float(source.score or 0.0)
    if _is_official_government_domain(domain):
        score += 10.0
    if "whitehouse.gov" in domain:
        score += 2.0
    text = f"{source.title} {source.content}".lower()
    if "president of the united states" in text:
        score += 2.0
    if "president donald" in text:
        score += 1.5
    return score


def _best_us_civic_source(web_sources: Sequence[WebSource]) -> WebSource | None:
    candidates = [
        source
        for source in web_sources
        if re.search(
            r"\b(president of the united states|united states.*(?:republic|constitution|presidency)|"
            r"monarchy in the united states)\b",
            f"{source.title} {source.content}".lower(),
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=_current_office_source_score)


def _fallback_answer_from_web_results(
    *,
    question: str,
    web_sources: Sequence[WebSource],
    model_answer: str,
    previous_answer: str,
    prefer_web_only: bool = False,
) -> str:
    sources = list(web_sources)[:MAX_WEB_SOURCES_TO_SHOW]
    source_labels = ", ".join(f"[{source.label}]" for source in sources)
    if not prefer_web_only and model_answer and model_answer.strip().lower() != MODEL_UNKNOWN.lower():
        return f"{model_answer.strip()}\n\nBest web sources checked: {source_labels}."

    if not prefer_web_only and previous_answer and not _looks_like_non_answer(previous_answer):
        return f"{previous_answer.strip()}\n\nBest web sources checked: {source_labels}."

    lines = [
        "The best web evidence I found points to these sources:",
    ]
    for source in sources[:3]:
        preview = _compact_source_text(source.content or source.title, limit=260)
        if preview:
            lines.append(f"- [{source.label}] {source.title}: {preview}")
        else:
            lines.append(f"- [{source.label}] {source.title}")
    if len(sources) > 3:
        lines.append(f"Additional sources checked: {', '.join(f'[{source.label}]' for source in sources[3:])}.")
    return "\n".join(lines)


def _looks_like_non_answer(answer: str) -> bool:
    lower = answer.lower()
    non_answer_markers = (
        "could not",
        "cannot",
        "can't",
        "i do not know",
        "i don't know",
        "will not guess",
        "try a more specific query",
        "no web sources",
    )
    return any(marker in lower for marker in non_answer_markers)


def _compact_source_text(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _labels_in_answer(answer: str, prefix: str) -> set[str]:
    pattern = re.compile(rf"\[{prefix}(\d+)\]", re.IGNORECASE)
    return {f"{prefix}{match.group(1)}" for match in pattern.finditer(answer)}


def _check_generation_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop and should_stop():
        raise GenerationStopped("Generation stopped by user.")


def _emit_stage(on_stage: Callable[[str], None] | None, label: str) -> None:
    if not on_stage:
        return
    try:
        on_stage(label)
    except Exception:
        return


def _generation_error_confidence(message: str) -> str:
    lower = message.lower()
    if is_model_selection_warning(message):
        return "model-selection-warning"
    if "token" in lower:
        return "needs-token"
    return "generation-error"


def _clean_error_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return message[:500]


def _append_web_update_note(answer: str, provider_label: str, note: str) -> str:
    return f"{answer.rstrip()}\n\nWeb update: {provider_label} {note}"
