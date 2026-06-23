"""Fast evidence-first RAG orchestration for local, model, web, and reranked answers."""

from __future__ import annotations

import copy
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date, datetime
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.parse import urlparse

from verilume.core.agents import (
    DEFAULT_NEWS_OUTLETS,
    ConversationContextAgent,
    ConversationResolution,
    IntentRouterAgent,
    QueryUnderstandingAgent,
    SearchPlan,
    _country_from_text,
    _country_phrase,
    _government_role_from_text,
    _public_topic_from_text,
    conversation_role_patterns,
    normalize_intent_text,
    requested_news_sources,
    update_state_from_answer,
)
from verilume.core.citation_verifier import CitationVerificationAgent
from verilume.core.conversation_state import ConversationState
from verilume.core.embeddings import EmbeddingService
from verilume.core.evidence import (
    build_final_answer_payload,
    classify_question,
    evidence_from_sources,
    rank_evidence,
    reconcile_dates,
    resolve_evidence_conflicts,
)
from verilume.core.generation import (
    LOCAL_UNKNOWN,
    MODEL_UNKNOWN,
    GenerationError,
    create_generator,
    is_model_selection_warning,
)
from verilume.core.query_preprocessing import normalize_query, query_variants
from verilume.core.query_interpreter import (
    QueryInterpretationAgent,
    apply_interpretation_to_state,
)
from verilume.core.reranking import query_terms, rerank_local_sources, rerank_web_sources
from verilume.core.retrieval import ChromaRetriever
from verilume.core.schemas import ChatMessage, LocalSource, RAGResponse, WebSource
from verilume.core.search_planner import SearchPlanner
from verilume.core.web_search import (
    DuckDuckGoSearch,
    boost_priority_sources,
    classify_query_domain,
    create_web_search,
    normalize_web_url_key,
)
from verilume.settings import AppSettings, ensure_app_dirs

MAX_WEB_SOURCES_TO_SHOW = 6
WEB_QUERY_FANOUT_LIMIT = 5
LOCAL_FILE_NOT_FOUND = "I could not find this in the indexed local files."
DEFAULT_RESPONSE_CACHE_TTL_SECONDS = 300.0
CURRENT_RESPONSE_CACHE_TTL_SECONDS = 60.0
WEB_NAME_TOKEN = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.\-']+"
PERSON_NAME_ACTION_BOUNDARIES = {
    "addresses",
    "addressed",
    "announces",
    "announced",
    "appoints",
    "appointed",
    "attends",
    "attended",
    "calls",
    "called",
    "congratulates",
    "congratulated",
    "gives",
    "gave",
    "holds",
    "held",
    "hosts",
    "hosted",
    "inaugurates",
    "inaugurated",
    "launches",
    "launched",
    "meets",
    "met",
    "names",
    "named",
    "opens",
    "opened",
    "receives",
    "received",
    "says",
    "said",
    "signs",
    "signed",
    "takes",
    "took",
    "urges",
    "urged",
    "visits",
    "visited",
    "welcomes",
    "welcomed",
}
PERSON_NAME_ORG_BOUNDARIES = {
    "ambassador",
    "army",
    "assembly",
    "command",
    "commander",
    "commission",
    "council",
    "delegation",
    "department",
    "deputy",
    "director",
    "embassy",
    "force",
    "forces",
    "government",
    "house",
    "minister",
    "ministry",
    "office",
    "parliament",
    "presidency",
    "secretary",
}
PERSON_NAME_SUFFIXES = {"ii", "iii", "iv", "jr", "sr"}

INSUFFICIENT_MARKERS = (
    "web_search_needed", LOCAL_UNKNOWN.lower(), MODEL_UNKNOWN.lower(), "i don't know",
    "i do not know", "cannot answer", "can't answer", "could not answer", "could not verify",
    "not enough information", "insufficient information", "insufficient context",
    "unable to determine", "not provided in the context", "not in the local",
)
WEB_REQUEST_MARKERS = (
    "search web", "search the web", "web search", "use web", "use the web", "look up",
    "online", "internet", "latest", "current", "recent", "today", "now", "news",
    "news channel", "news channels", "reuters", "ap news", "bbc", "sky news",
    "financial times", "guardian",
)
LOCAL_FILE_MARKERS = (
    "in the file", "in the document", "in my file", "in my document",
    "in the local file", "in the local files", "indexed local file", "indexed local files",
    "indexed local documents", "in the database", "local database", "knowledge base",
    "uploaded file", "uploaded files", "uploaded document", "uploaded documents", "in my files",
    "in the files", "in my documents", "which document", "which file", "what document", "what file",
)
LOCAL_FILE_EXPANSIONS = {
    "certificate": ("certificate", "certification", "diploma", "credential", "language test", "sproochentest", "exam result"),
    "language": ("language", "sproochentest", "test", "exam", "certificate"),
}
IDENTITY_STOPWORDS = {
    "about",
    "and",
    "are",
    "became",
    "become",
    "current",
    "did",
    "do",
    "does",
    "for",
    "how",
    "internet",
    "is",
    "latest",
    "look",
    "old",
    "online",
    "profile",
    "search",
    "the",
    "was",
    "web",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
}

USA_PATTERN = re.compile(r"\b(?:usa|u\.s\.a\.?|united states|u\.s\.?)\b", re.IGNORECASE)
UK_PATTERN = re.compile(r"\b(?:uk|u\.k\.?|united kingdom)\b", re.IGNORECASE)
MONARCH_ROLE_LABELS = {"King", "Queen", "Monarch"}
HEAD_OF_STATE_ROLE_BY_COUNTRY = {
    "Cameroon": "President",
    "Democratic Republic of the Congo": "President",
    "France": "President",
    "United States": "President",
}
OFFICIAL_WEB_MARKERS = (
    "whitehouse.gov",
    "state.gov",
    ".gov",
    "gov.uk",
    "gouvernement.lu",
    "public.lu",
    "europa.eu",
    "royal-house.nl",
    "usa.gov",
)
NEWS_WEB_MARKERS = (
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "news.sky.com",
    "sky.com",
    "ft.com",
    "financialtimes.com",
    "theguardian.com",
    "theguardian.co.uk",
    "bloomberg.com",
    "cnbc.com",
    "politico.com",
)
PUBLIC_OFFICE_MARKERS = (
    "president",
    "prime minister",
    "secretary of state",
    "minister of defence",
    "minister of defense",
    "defence minister",
    "defense minister",
    "foreign secretary",
    "foreign minister",
    "finance minister",
    "interior minister",
    "king",
    "queen",
    "monarch",
    "ceo",
)
IDENTITY_LOCAL_POSITIVE_MARKERS = (
    "about",
    "author",
    "bio",
    "biography",
    "cv",
    "curriculum vitae",
    "department",
    "doctoral",
    "github",
    "homepage",
    "linkedin",
    "orcid",
    "portfolio",
    "profile",
    "publication",
    "research",
    "researcher",
    "resume",
    "scholar",
    "student",
    "thesis",
    "university",
)
IDENTITY_LOCAL_STRONG_MARKERS = (
    "about",
    "author",
    "bio",
    "biography",
    "cv",
    "curriculum vitae",
    "github",
    "homepage",
    "linkedin",
    "orcid",
    "portfolio",
    "profile",
    "resume",
    "scholar",
)
IDENTITY_LOCAL_CONTEXT_MARKERS = (
    "department",
    "doctoral",
    "publication",
    "research",
    "researcher",
    "student",
    "thesis",
    "university",
)
IDENTITY_LOCAL_NEGATIVE_MARKERS = (
    "acknowledgement",
    "acknowledgements",
    "attestation",
    "attended",
    "bank",
    "bill",
    "certificate",
    "certification",
    "course notes",
    "exam",
    "invoice",
    "language test",
    "lecture notes",
    "passport",
    "payment",
    "receipt",
    "seminar",
    "sproochentest",
    "tax",
    "ticket",
    "visa",
)
SCIENTIFIC_LOCAL_POSITIVE_MARKERS = (
    "abstract",
    "article",
    "bayesian",
    "conference",
    "dissertation",
    "doi",
    "hamiltonian",
    "journal",
    "monte carlo",
    "mcmc",
    "paper",
    "publication",
    "replica exchange",
    "research",
    "thesis",
    "thermodynamic",
    "university",
)
SCIENTIFIC_LOCAL_NEGATIVE_MARKERS = (
    "certification",
    "exam",
    "prep guide",
    "programming for sas",
    "sas 9",
    "study guide",
    "training manual",
)
PUBLIC_KNOWLEDGE_MARKERS = (
    "directive",
    "eu reach",
    "reach directive",
    "reach regulation",
    "regulation",
    "toxicology",
)
PUBLIC_DEFINITION_PREFIXES = (
    "define",
    "explain",
    "identify",
    "list",
    "name",
    "tell me about",
    "what are",
    "what is",
    "which",
)


class GenerationStopped(RuntimeError):
    """Raised when the user stops multi-stage generation."""


class DiagnosticsBuilder:
    """Small helper for consistent RAG diagnostics construction."""

    def __init__(self) -> None:
        self._data: dict = {}

    def add_query_info(self, *, original: str, resolved: str, query: str) -> "DiagnosticsBuilder":
        self._data.update(
            {
                "query": query,
                "original_query": original,
                "resolved_query": resolved,
            }
        )
        return self

    def add_conversation(self, conversation) -> "DiagnosticsBuilder":
        state = conversation.state
        self._data.update(
            {
                "conversation_followup": conversation.is_followup,
                "conversation_entities": state.active_entities,
                "conversation_topics": state.active_topics,
                "conversation_country": state.active_country,
                "conversation_person": state.active_person,
                "conversation_event": state.active_event,
                "conversation_news_story": state.active_news_story,
                "conversation_intent": state.intent,
                "conversation_roles": state.roles,
                "roles": state.roles,
                "preferred_sources": state.preferred_sources,
                "news_intent": conversation.news_intent,
                "requested_sources": conversation.requested_sources,
            }
        )
        return self

    def add_search_plan(self, search_plan: SearchPlan) -> "DiagnosticsBuilder":
        self._data.update(
            {
                "search_plan": search_plan.diagnostics(),
                "search_plan_intent": search_plan.intent,
                "search_plan_preferred_sources": search_plan.preferred_sources,
                "search_plan_need_local": search_plan.need_local,
                "search_plan_need_web": search_plan.need_web,
                "search_plan_need_model": search_plan.need_model,
            }
        )
        return self

    def update(self, **values) -> "DiagnosticsBuilder":
        self._data.update(values)
        return self

    def build(self) -> dict:
        return dict(self._data)


class VerilumeRAG:
    def __init__(self, settings: AppSettings) -> None:
        ensure_app_dirs(settings)
        self.settings = settings
        self.embeddings = EmbeddingService(
            settings.embed_model,
            settings.embed_device,
            cache_dir=settings.embedding_cache_dir,
            cache_enabled=settings.embedding_cache_enabled,
        )
        self.retriever = ChromaRetriever(
            settings.chroma_dir,
            settings.collection_name,
            self.embeddings,
            settings=settings,
        )
        self.generator = create_generator(settings)
        self.web_search = create_web_search(settings)
        self.conversation_context_agent = ConversationContextAgent()
        self.intent_router = IntentRouterAgent()
        self.query_interpretation_agent = QueryInterpretationAgent(self.generator)
        self.query_understanding_agent = QueryUnderstandingAgent()
        self.search_planner = SearchPlanner()
        self.citation_verifier = CitationVerificationAgent()
        self._response_cache: dict[tuple, tuple[float, RAGResponse]] = {}

    def ask(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
        conversation_state: ConversationState | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> RAGResponse:
        history = history or []
        cache_key = _response_cache_key(question, history, conversation_state)
        if should_stop is None:
            cached = self._cached_response(cache_key)
            if cached is not None:
                _emit_stage(on_stage, "✓ Cached answer ready")
                return cached

        response = self._ask_uncached(question, history, conversation_state, should_stop, on_stage)
        if should_stop is None:
            self._store_cached_response(cache_key, response)
        return response

    def _ask_uncached(
        self,
        question: str,
        history: Sequence[ChatMessage],
        conversation_state: ConversationState | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> RAGResponse:
        _check_generation_stop(should_stop)

        original_question = question
        base_state = _merged_conversation_state_for_interpretation(
            self.conversation_context_agent.state_from_history(list(history)),
            conversation_state,
        )

        route = self.intent_router.route(original_question)
        if not route.uses_rag:
            diagnostics = (
                DiagnosticsBuilder()
                .add_query_info(
                    original=original_question,
                    resolved=original_question,
                    query=original_question,
                )
                .update(
                    query_type=route.route,
                    query_types=[route.route],
                    pipeline="intent_router",
                    **route.diagnostics,
                )
                .build()
            )
            response = RAGResponse(
                answer=route.answer,
                local_sources=[],
                web_sources=[],
                used_web=False,
                confidence=route.route,
                diagnostics=diagnostics,
            )
            return _attach_conversation_state(response, base_state, original_question, original_question)

        self.query_interpretation_agent.generator = self.generator
        _emit_stage(on_stage, "Interpreting question...")
        interpretation = self.query_interpretation_agent.interpret(
            original_question,
            list(history),
            base_state,
        )
        interpreted_state = apply_interpretation_to_state(base_state, interpretation)
        if interpretation.needs_clarification:
            diagnostics = (
                DiagnosticsBuilder()
                .add_query_info(
                    original=original_question,
                    resolved=interpretation.resolved_question,
                    query=interpretation.resolved_question,
                )
                .update(
                    query_type="clarification",
                    query_types=["clarification"],
                    pipeline="query_interpreter",
                    query_interpretation=interpretation.diagnostics,
                    interpretation_intent=interpretation.intent,
                    needs_clarification=True,
                    clarification_question=interpretation.clarification_question,
                )
                .build()
            )
            response = RAGResponse(
                interpretation.clarification_question or "Can you clarify what you mean?",
                [],
                [],
                False,
                "clarification",
                diagnostics,
            )
            return _attach_conversation_state(
                response,
                interpreted_state,
                original_question,
                interpretation.resolved_question,
            )

        question = interpretation.resolved_question or original_question
        semantic_plan = self.search_planner.plan(interpretation)
        search_plan = semantic_plan.to_legacy_plan()
        search_plan.country = interpreted_state.active_country
        search_plan.role = interpreted_state.active_role or _government_role_from_text(
            normalize_intent_text(question)
        )
        search_plan.entity = interpreted_state.active_person
        search_plan.topic = (
            interpreted_state.active_topic
            or interpreted_state.active_research_topic
            or (interpreted_state.active_topics[0] if interpreted_state.active_topics else "")
        )
        conversation = ConversationResolution(
            original_question=original_question,
            resolved_question=question,
            state=interpreted_state,
            is_followup=interpretation.is_follow_up,
            news_intent=interpretation.intent == "news",
            requested_sources=interpretation.preferred_sources,
        )

        query_understanding = self.query_understanding_agent.understand(question)
        local_file_question = (
            self._is_local_file_question(question)
            or interpretation.intent == "local_document"
        )
        query_understanding.local_file_question = local_file_question
        identity_tokens = _identity_tokens(question)

        query = question
        if (
            self.settings.enable_query_rewrite
            and question == original_question
            and _should_rewrite_query(
                question,
                history,
                min_history=self.settings.query_rewrite_min_history,
                similarity_threshold=self.settings.query_rewrite_similarity_threshold,
            )
        ):
            _emit_stage(on_stage, "Rewriting query...")
            query = self.generator.rewrite_query(question, list(history))

        time_sensitive = (
            query_understanding.requires_date_reconciliation
            or semantic_plan.freshness_required
        )
        diagnostics = (
            DiagnosticsBuilder()
            .add_query_info(original=original_question, resolved=question, query=query)
            .add_conversation(conversation)
            .add_search_plan(search_plan)
            .update(
                query_interpretation=interpretation.diagnostics,
                interpretation_intent=interpretation.intent,
                interpretation_entities=interpretation.entities,
                interpretation_search_queries=interpretation.search_queries,
                interpretation_use_local=interpretation.use_local,
                interpretation_use_web=interpretation.use_web,
                interpretation_use_model=interpretation.use_model_knowledge,
                semantic_search_plan=semantic_plan.diagnostics(),
                normalized_query=normalize_query(query).canonical,
                normalized_key_terms=list(normalize_query(query).key_terms),
                normalized_entities=list(normalize_query(query).entities),
                query_type=query_understanding.primary_type.value,
                query_types=[item.value for item in query_understanding.types],
                local_file_question=query_understanding.local_file_question,
                time_sensitive=time_sensitive,
                requires_web_validation=query_understanding.requires_web_validation,
                requires_date_reconciliation=time_sensitive,
                generation_backend=self.settings.generation_backend,
                generation_model=self.settings.active_generation_model(),
                pipeline="local_first_parallel_fallback_evidence",
            )
            .build()
        )

        force_web = self._is_web_requested(question)
        planned_web = search_plan.need_web
        web_ready = bool(self.settings.enable_web_search and getattr(self.web_search, "is_configured", True))
        current_or_web = bool(force_web or time_sensitive or (planned_web and not search_plan.need_local))
        expanded_web_queries = _web_queries(question, query, search_plan)
        if search_plan.intent == "government":
            web_queries = _dedupe_web_queries([*expanded_web_queries, *semantic_plan.search_queries])
        else:
            web_queries = _dedupe_web_queries([*semantic_plan.search_queries, *expanded_web_queries])
        diagnostics["web_queries"] = web_queries
        skip_local_retrieval = (not search_plan.need_local) or _should_skip_local_retrieval(
            question,
            query_understanding,
            local_file_question,
        )
        local_queries = _local_search_queries(query, identity_tokens, local_file_question)
        diagnostics["local_queries"] = local_queries
        diagnostics["local_retrieval_skipped"] = skip_local_retrieval
        diagnostics["local_retrieval_attempted"] = not skip_local_retrieval

        if skip_local_retrieval:
            _emit_stage(on_stage, "Skipping local retrieval for public web/model evidence...")
            local_sources = []
        else:
            _emit_stage(on_stage, "Searching local evidence...")
            local_sources = self._search_local_sources(query, identity_tokens, local_file_question)
        _check_generation_stop(should_stop)
        diagnostics["local_count"] = len(local_sources)
        diagnostics["best_local_score"] = _best_local_score(local_sources)
        strong_local = _local_evidence_looks_strong(local_sources, self.settings)
        diagnostics["local_evidence_strong"] = strong_local
        _emit_stage(on_stage, f"✓ Local retrieval ({len(local_sources)} matches)")

        if local_file_question and (
            not local_sources or _should_answer_local_file_question_directly(query)
        ):
            response = self._answer_local_file_question(query, local_sources, diagnostics, on_stage)
            return _attach_conversation_state(response, conversation.state, original_question, question)

        local_answer = LOCAL_UNKNOWN
        local_sufficient = False
        local_answer_relevant = False
        generation_error = ""
        if local_sources:
            _emit_stage(on_stage, "Checking local evidence...")
            try:
                local_answer = self.generator.answer_local(query, list(history), local_sources)
                local_answer_relevant = _local_answer_supports_question(
                    query,
                    local_answer,
                    local_sources,
                    local_file_question=local_file_question,
                )
                local_sufficient = self._is_sufficient(local_answer) and local_answer_relevant
            except GenerationError as exc:
                generation_error = str(exc)
                diagnostics["generation_error"] = _clean_error_message(exc)
                diagnostics["generation_error_confidence"] = _generation_error_confidence(generation_error)
        diagnostics["local_answer_relevant"] = local_answer_relevant
        diagnostics["local_sufficient"] = local_sufficient

        if local_sufficient and not current_or_web:
            used_local_sources = _local_sources_used_in_answer(local_sources, local_answer) or local_sources
            _add_evidence_diagnostics(
                diagnostics,
                question,
                used_local_sources,
                [],
                local_answer,
                True,
                None,
                False,
                self.settings,
            )
            _emit_stage(on_stage, "✓ Local evidence answered the question")
            response = RAGResponse(
                local_answer,
                used_local_sources,
                [],
                False,
                "local-grounded",
                diagnostics,
            )
            return _attach_conversation_state(response, conversation.state, original_question, question)

        should_use_web = _should_use_web(
            question=question,
            force_web=force_web or planned_web,
            web_enabled=self.settings.enable_web_search,
            query_understanding=query_understanding,
        ) or bool(generation_error and web_ready)
        diagnostics["web_requested"] = should_use_web
        diagnostics["web_provider"] = self.settings.web_search_provider_label()

        web_sources: list[WebSource] = []
        web_error = ""
        model_answer = MODEL_UNKNOWN
        model_sufficient = False
        model_answer_relevant = False

        if generation_error and not should_use_web:
            response = RAGResponse(
                answer=generation_error,
                local_sources=[],
                web_sources=[],
                used_web=False,
                confidence=_generation_error_confidence(generation_error),
                diagnostics=diagnostics,
            )
            return _attach_conversation_state(response, conversation.state, original_question, question)

        if should_use_web and not web_ready:
            diagnostics["web_count"] = 0
            diagnostics["web_note"] = "Web search was requested, but the selected provider is not configured."

        if should_use_web and web_ready:
            _emit_stage(on_stage, "Checking AI knowledge and web evidence...")
            skip_model_for_entity = (
                query_understanding.personal_company_entity_lookup
                and not query_understanding.requires_date_reconciliation
            )
            skip_model_for_current_role = bool(_current_public_role_context(question))
            skip_model_for_public_office = _is_public_office_query(question, query_understanding)
            skip_model_for_age_at_office = _looks_like_age_at_office_query(question)
            skip_model_for_plan = not search_plan.need_model
            skip_model = (
                skip_model_for_entity
                or skip_model_for_current_role
                or skip_model_for_public_office
                or skip_model_for_age_at_office
                or skip_model_for_plan
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                web_future = executor.submit(
                    self._search_web_sources,
                    web_queries,
                    question=question,
                    prefer_fast_public_search=(
                        skip_model_for_entity
                        or skip_model_for_current_role
                        or skip_model_for_public_office
                        or _looks_like_public_knowledge_query(question)
                        or skip_model_for_age_at_office
                        or skip_model_for_plan
                    ),
                )
                model_future: Future | None = None
                if not generation_error and not skip_model:
                    model_future = executor.submit(self.generator.answer_model_knowledge, query, list(history))
                elif skip_model_for_age_at_office:
                    diagnostics["model_skipped"] = "age-at-office query uses web evidence first"
                elif skip_model_for_entity:
                    diagnostics["model_skipped"] = "entity lookup uses web evidence first"
                elif skip_model_for_current_role and "current" in original_question.lower():
                    diagnostics["model_skipped"] = "current public role uses web evidence first"
                elif skip_model_for_public_office:
                    diagnostics["model_skipped"] = "current public office query uses web evidence first"
                elif skip_model_for_current_role:
                    diagnostics["model_skipped"] = "current public role uses web evidence first"
                elif skip_model_for_plan:
                    diagnostics["model_skipped"] = f"{search_plan.intent} plan uses source evidence first"

                try:
                    web_sources = web_future.result()
                    if identity_tokens:
                        web_sources = _filter_web_sources_for_identity(web_sources, identity_tokens)
                    web_sources = self._rerank_web(_web_rerank_query(query, web_queries), web_sources)
                    diagnostics["web_count"] = len(web_sources)
                    _emit_stage(on_stage, f"✓ Web evidence ({len(web_sources)} sources)")
                except Exception as exc:
                    web_error = _clean_error_message(exc)
                    diagnostics["web_error"] = web_error
                    diagnostics["web_count"] = 0
                    diagnostics["web_note"] = (
                        f"{self.settings.web_search_provider_label()} search could not complete."
                    )

                if model_future is not None:
                    try:
                        model_answer = model_future.result()
                        model_answer_relevant = _model_answer_supports_question(query, model_answer)
                        model_sufficient = self._is_sufficient(model_answer) and model_answer_relevant
                    except GenerationError as exc:
                        generation_error = str(exc)
                        diagnostics["model_error"] = _clean_error_message(exc)
                        diagnostics["generation_error_confidence"] = _generation_error_confidence(generation_error)
                    except Exception as exc:
                        diagnostics["model_error"] = _clean_error_message(exc)
        elif not local_sufficient and not generation_error:
            _emit_stage(on_stage, "Checking AI knowledge...")
            try:
                model_answer = self.generator.answer_model_knowledge(query, list(history))
                model_answer_relevant = _model_answer_supports_question(query, model_answer)
                model_sufficient = self._is_sufficient(model_answer) and model_answer_relevant
            except GenerationError as exc:
                generation_error = str(exc)
                diagnostics["model_error"] = _clean_error_message(exc)
                diagnostics["generation_error_confidence"] = _generation_error_confidence(generation_error)
            except Exception as exc:
                diagnostics["model_error"] = _clean_error_message(exc)
        if (
            not should_use_web
            and not local_sufficient
            and self.settings.enable_web_search
            and (generation_error or not model_sufficient)
        ):
            should_use_web = True
            diagnostics["web_requested"] = True
            diagnostics["web_reason"] = "fallback_after_model"
            if not web_ready:
                diagnostics["web_count"] = 0
                diagnostics["web_note"] = "Web search fallback is enabled, but the selected provider is not configured."
            else:
                _emit_stage(on_stage, "Checking web evidence...")
                try:
                    web_sources = self._search_web_sources(
                        web_queries,
                        question=question,
                        prefer_fast_public_search=_looks_like_public_knowledge_query(question),
                    )
                    if identity_tokens:
                        web_sources = _filter_web_sources_for_identity(web_sources, identity_tokens)
                    web_sources = self._rerank_web(_web_rerank_query(query, web_queries), web_sources)
                    diagnostics["web_count"] = len(web_sources)
                    _emit_stage(on_stage, f"✓ Web evidence ({len(web_sources)} sources)")
                except Exception as exc:
                    web_error = _clean_error_message(exc)
                    diagnostics["web_error"] = web_error
                    diagnostics["web_count"] = 0
                    diagnostics["web_note"] = (
                        f"{self.settings.web_search_provider_label()} search could not complete."
                    )
        diagnostics["model_answer_relevant"] = model_answer_relevant
        diagnostics["model_sufficient"] = model_sufficient

        _check_generation_stop(should_stop)
        _emit_stage(on_stage, "Extracting and ranking evidence...")
        ranked_evidence, resolution = _add_evidence_diagnostics(
            diagnostics,
            question,
            local_sources,
            web_sources,
            local_answer,
            local_sufficient,
            model_answer,
            model_sufficient,
            self.settings,
        )

        _emit_stage(on_stage, "Verifying citations and generating final answer...")
        answer = self._generate_final_answer(
            question,
            list(history),
            local_answer,
            model_answer,
            local_sources,
            web_sources,
            ranked_evidence,
            web_error,
            generation_error,
            should_use_web,
            force_web,
            time_sensitive,
            model_sufficient,
            local_sufficient,
        )
        current_override = _current_public_fact_answer(question, web_sources)
        if (
            current_override
            and time_sensitive
            and not _looks_like_office_start_query(question)
            and not _looks_like_age_at_office_query(question)
        ):
            answer, evidence_conflict = current_override
            diagnostics["current_role_override"] = True
            diagnostics["evidence_conflict"] = evidence_conflict

        answer = _verify_citations(answer, local_sources, web_sources)
        citation_verification = self.citation_verifier.verify(
            answer,
            question=question,
            local_sources=local_sources,
            web_sources=web_sources,
        )
        answer = citation_verification.answer
        diagnostics["citation_verification_supported"] = citation_verification.supported
        diagnostics["citation_verification_labels"] = citation_verification.cited_labels
        if citation_verification.missing_labels:
            diagnostics["citation_verification_missing_labels"] = citation_verification.missing_labels
        if citation_verification.notes:
            diagnostics["citation_verification_notes"] = citation_verification.notes
        verification = _verify_answer_against_evidence(
            answer,
            local_sources,
            web_sources,
            question,
            self.settings,
        )
        diagnostics["answer_verification_status"] = verification["status"]
        diagnostics["answer_verification_score"] = verification["score"]
        if verification.get("note"):
            diagnostics["answer_verification_note"] = verification["note"]
        if verification["status"] == "unsupported" and web_sources and (current_or_web or time_sensitive):
            retry_override = _current_public_fact_answer(question, web_sources)
            if retry_override:
                answer, evidence_conflict = retry_override
                diagnostics["current_role_override"] = True
                diagnostics["evidence_conflict"] = evidence_conflict
            else:
                answer = _fallback_answer_from_web_results(
                    web_sources=web_sources,
                    previous_answer=answer,
                )
                diagnostics["unsupported_answer_rebuilt_from_web"] = True
            verification = _verify_answer_against_evidence(
                answer,
                local_sources,
                web_sources,
                question,
                self.settings,
            )
            diagnostics["answer_verification_status"] = verification["status"]
            diagnostics["answer_verification_score"] = verification["score"]
            if verification.get("note"):
                diagnostics["answer_verification_note"] = verification["note"]
        used_local_sources = _local_sources_used_in_answer(local_sources, answer)
        used_web_sources = _web_sources_used_in_answer(web_sources, answer)
        if web_sources and not used_web_sources and (current_or_web or not used_local_sources):
            answer = _fallback_answer_from_web_results(web_sources=web_sources, previous_answer=answer)
            used_web_sources = _web_sources_used_in_answer(web_sources, answer)

        display_web_sources = _best_web_sources(web_sources, used_web_sources)
        confidence = self._confidence(
            local_sources=used_local_sources,
            used_web=bool(used_web_sources),
            answer=answer,
            evidence_confidence=resolution.confidence,
            time_sensitive=query_understanding.requires_date_reconciliation,
        )
        if (
            verification["status"] == "unsupported"
            and str(getattr(self.settings, "answer_verification_mode", "heuristic")).lower()
            in {"strict", "enforce", "enforced"}
        ):
            confidence = "low"
        _emit_stage(on_stage, "✓ Evidence-ranked answer ready")
        response = RAGResponse(
            answer,
            used_local_sources,
            display_web_sources,
            bool(used_web_sources),
            confidence,
            diagnostics,
        )
        updated_state = update_state_from_answer(
            conversation.state,
            question=original_question,
            resolved_query=question,
            answer=answer,
        )
        response.diagnostics["conversation_roles"] = updated_state.roles
        response.diagnostics["roles"] = updated_state.roles
        response.diagnostics["conversation_country"] = updated_state.active_country
        response.diagnostics["conversation_person"] = updated_state.active_person
        return _attach_conversation_state(response, updated_state, original_question, question)

    def _cached_response(self, cache_key: tuple) -> RAGResponse | None:
        cached = self._response_cache.get(cache_key)
        if not cached:
            return None
        expires_at, response = cached
        if time.monotonic() >= expires_at:
            self._response_cache.pop(cache_key, None)
            return None
        value = copy.deepcopy(response)
        value.diagnostics = dict(value.diagnostics or {})
        value.diagnostics["cache_hit"] = True
        return value

    def _store_cached_response(self, cache_key: tuple, response: RAGResponse) -> None:
        ttl = _response_cache_ttl(response)
        if ttl <= 0:
            return
        self._response_cache[cache_key] = (time.monotonic() + ttl, copy.deepcopy(response))

    def _search_local_sources(self, query: str, identity_tokens: Sequence[str], local_file_question: bool) -> list[LocalSource]:
        mode = getattr(self.settings, "retrieval_mode", "hybrid")
        if local_file_question or identity_tokens:
            mode = "bm25"
        pool_limit = max(self.settings.retriever_k * 8, int(getattr(self.settings, "reranker_top_k", self.settings.retriever_k)) * 4, 40)
        search_k = max(self.settings.retriever_k, int(getattr(self.settings, "reranker_top_k", self.settings.retriever_k)))
        sources: list[LocalSource] = []
        for local_query in _local_search_queries(query, identity_tokens, local_file_question):
            threshold = self.settings.retrieval_score_threshold
            if local_query != query:
                threshold = max(0.18, threshold - 0.12)
            sources = _merge_local_sources(
                sources,
                self.retriever.search(
                    local_query,
                    k=search_k,
                    score_threshold=threshold,
                    mode=mode,
                ),
                limit=pool_limit,
            )
        if identity_tokens and not local_file_question:
            sources = _filter_local_sources_for_identity(sources, identity_tokens)
        ranked = self._rerank_local(query, sources)
        return _filter_relevant_local_sources(
            query,
            ranked,
            identity_tokens=identity_tokens,
            local_file_question=local_file_question,
            limit=self.settings.retriever_k,
        )

    def _rerank_local(self, query: str, sources: Sequence[LocalSource]) -> list[LocalSource]:
        return rerank_local_sources(
            query,
            sources,
            model_name=getattr(self.settings, "reranker_model", "BAAI/bge-reranker-base"),
            device=getattr(self.settings, "reranker_device", self.settings.embed_device),
            top_k=getattr(self.settings, "reranker_top_k", self.settings.retriever_k),
            enabled=getattr(self.settings, "enable_reranker", True),
            semantic_weight=getattr(self.settings, "rerank_semantic_weight", 0.52),
            lexical_weight=getattr(self.settings, "rerank_lexical_weight", 0.48),
            phrase_bonus_full=getattr(self.settings, "rerank_phrase_bonus_full", 0.28),
            phrase_bonus_partial=getattr(self.settings, "rerank_phrase_bonus_partial", 0.16),
            mismatch_penalty=getattr(self.settings, "rerank_mismatch_penalty", 0.55),
            mismatch_threshold=getattr(self.settings, "rerank_mismatch_threshold", 0.72),
            single_match_penalty=getattr(self.settings, "rerank_single_match_penalty", 0.78),
            single_match_threshold=getattr(self.settings, "rerank_single_match_threshold", 0.78),
        )

    def _rerank_web(self, query: str, sources: Sequence[WebSource]) -> list[WebSource]:
        return rerank_web_sources(
            query,
            sources,
            model_name=getattr(self.settings, "reranker_model", "BAAI/bge-reranker-base"),
            device=getattr(self.settings, "reranker_device", self.settings.embed_device),
            top_k=min(MAX_WEB_SOURCES_TO_SHOW, getattr(self.settings, "reranker_top_k", MAX_WEB_SOURCES_TO_SHOW)),
            enabled=getattr(self.settings, "enable_reranker", True),
            semantic_weight=getattr(self.settings, "rerank_semantic_weight", 0.52),
            lexical_weight=getattr(self.settings, "rerank_lexical_weight", 0.48),
            phrase_bonus_full=getattr(self.settings, "rerank_phrase_bonus_full", 0.28),
            phrase_bonus_partial=getattr(self.settings, "rerank_phrase_bonus_partial", 0.16),
            mismatch_penalty=getattr(self.settings, "rerank_mismatch_penalty", 0.55),
            mismatch_threshold=getattr(self.settings, "rerank_mismatch_threshold", 0.72),
            single_match_penalty=getattr(self.settings, "rerank_single_match_penalty", 0.78),
            single_match_threshold=getattr(self.settings, "rerank_single_match_threshold", 0.78),
        )

    def _generate_final_answer(
        self,
        question,
        history,
        local_answer,
        model_answer,
        local_sources,
        web_sources,
        ranked_evidence,
        web_error,
        generation_error,
        should_use_web,
        force_web,
        time_sensitive,
        model_sufficient,
        local_sufficient,
    ) -> str:
        query_understanding = classify_question(question)
        provider_label = self.settings.web_search_provider_label()

        if web_error and local_sufficient:
            return _append_web_update_note(
                local_answer,
                provider_label,
                "search could not complete, so no web sources were added.",
            )
        needs_current_verification = _requires_current_source_verification(question, query_understanding)
        if needs_current_verification and not web_sources and not (local_answer and self._is_sufficient(local_answer)):
            if web_error:
                return _helpful_failure_answer(
                    question,
                    reason="current_web_failed",
                    provider_label=provider_label,
                    web_error=web_error,
                )
            return _helpful_failure_answer(
                question,
                reason="current_no_sources",
                provider_label=provider_label,
            )
        if needs_current_verification and web_sources and all(
            _is_unreliable_current_source(source) for source in web_sources
        ):
            return _unverified_current_web_answer(web_sources)
        if _looks_like_office_start_query(question) and web_sources:
            start_answer = _office_start_answer_from_web(question, web_sources)
            if start_answer:
                return start_answer
        if _looks_like_age_at_office_query(question) and web_sources:
            age_answer = _age_at_office_answer_from_web(question, web_sources)
            if age_answer:
                return age_answer
        if (
            query_understanding.personal_company_entity_lookup
            and web_sources
            and not query_understanding.requires_date_reconciliation
            and not local_sufficient
        ):
            return _fallback_from_ranked_evidence(ranked_evidence, question=question)
        if (
            _looks_like_scientific_local_query(question)
            and web_sources
            and not query_understanding.requires_date_reconciliation
        ):
            return _scientific_answer_from_ranked_evidence(ranked_evidence, question=question)
        if _looks_like_public_knowledge_query(question) and web_sources:
            return _public_knowledge_answer_from_web(question, web_sources)
        if _looks_like_news_query(question) and web_sources:
            return _news_answer_from_web(question, web_sources)
        if web_sources:
            try:
                return self._answer_from_verified_evidence(
                    question,
                    history,
                    local_answer,
                    model_answer,
                    ranked_evidence,
                )
            except GenerationError:
                try:
                    return self.generator.answer_final(
                        question=question,
                        history=history,
                        local_answer=local_answer,
                        model_answer=model_answer,
                        local_sources=local_sources,
                        web_sources=web_sources,
                    ).strip()
                except GenerationError:
                    return _fallback_from_ranked_evidence(ranked_evidence, question=question)
        if local_sufficient:
            return local_answer
        if model_sufficient:
            answer = f"{model_answer}\n\nSource: AI knowledge (not externally verified)"
            if web_error and should_use_web:
                return _append_web_update_note(
                    answer,
                    provider_label,
                    "search could not complete, so this answer was not updated with web sources.",
                )
            return answer
        if generation_error and not should_use_web:
            return generation_error
        if force_web or should_use_web:
            if web_error:
                return _helpful_failure_answer(
                    question,
                    reason="web_failed",
                    provider_label=provider_label,
                    web_error=web_error,
                )
            return _helpful_failure_answer(
                question,
                reason="no_relevant_sources",
                provider_label=provider_label,
                web_enabled=self.settings.enable_web_search,
                web_sources=web_sources,
            )
        if generation_error:
            return generation_error
        return _helpful_failure_answer(
            question,
            reason="no_answer",
            provider_label=provider_label,
            web_enabled=self.settings.enable_web_search,
            web_sources=web_sources,
        )

    def _answer_from_verified_evidence(
        self,
        question,
        history,
        local_answer,
        model_answer,
        ranked_evidence,
    ) -> str:
        chat = getattr(self.generator, "chat", None)
        if not callable(chat):
            raise GenerationError("Structured final synthesis is unavailable for this generator.")
        style_instruction = getattr(
            self.generator,
            "style_instruction",
            "Provide a clear, complete answer.",
        )
        payload = build_final_answer_payload(question, ranked_evidence)
        confidence = _verified_payload_confidence(payload, ranked_evidence)
        answer = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Verilume, an evidence-first AI assistant. "
                        "Use only the verified evidence provided. Do not invent facts or citations. "
                        "Cite local files as [S1], [S2] and web sources as [W1], [W2]. "
                        "AI knowledge can explain wording, but it must never override verified evidence. "
                        "Answer or respond to the user's question or statement directly first. "
                        "Put confidence after the answer as: Confidence: High, Confidence: Medium, or Confidence: Low. "
                        f"Style: {style_instruction}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Conversation history:\n{_format_history(history)}\n\n"
                        f"Evidence confidence to use unless evidence conflicts: {confidence}\n\n"
                        f"Local answer candidate:\n{local_answer}\n\n"
                        f"AI knowledge candidate:\n{model_answer}\n\n"
                        f"{payload.generator_instructions}\n\n"
                        "Write the final answer only. Prefer direct synthesis over listing evidence. "
                        "Do not make the answer only a list of sources."
                    ),
                },
            ]
        ).strip()
        if not answer:
            raise GenerationError("The final synthesis returned an empty answer.")
        if not re.search(r"\bconfidence:\s*(high|medium|low)\b", answer, flags=re.IGNORECASE):
            answer = f"{answer}\n\nConfidence: {confidence}"
        return answer

    def _answer_local_file_question(self, question: str, local_sources: list[LocalSource], diagnostics: dict, on_stage=None) -> RAGResponse:
        expanded_query = _expand_local_file_query(question)
        diagnostics["expanded_local_query"] = expanded_query
        expanded_sources: list[LocalSource] = []
        expansion_queries = _local_file_search_queries(question)
        diagnostics["local_file_search_queries"] = expansion_queries
        ranking_query = _local_file_ranking_query(question)
        pool_limit = max(self.settings.retriever_k * 8, 64)
        source_limit = max(self.settings.retriever_k, 8)
        final_sources: list[LocalSource] | None = None
        if expansion_queries:
            _emit_stage(on_stage, "Expanding local file keywords...")
            for expansion_query in expansion_queries:
                expanded_sources = _merge_local_sources(
                    expanded_sources,
                    self.retriever.search(
                        expansion_query,
                        k=max(self.settings.retriever_k, 8),
                        score_threshold=max(0.18, self.settings.retrieval_score_threshold - 0.12),
                        mode="bm25",
                    ),
                    limit=pool_limit,
                )
                candidate_sources = _rank_and_filter_local_file_sources(
                    self,
                    ranking_query,
                    _merge_local_sources(local_sources, expanded_sources, limit=pool_limit),
                    source_limit,
                )
                if _local_file_sources_are_enough(candidate_sources):
                    final_sources = candidate_sources
                    break

        if final_sources is None:
            final_sources = _rank_and_filter_local_file_sources(
                self,
                ranking_query,
                _merge_local_sources(local_sources, expanded_sources, limit=pool_limit),
                source_limit,
            )

        local_sources = final_sources
        diagnostics["local_count"] = len(local_sources)
        diagnostics["expanded_local_count"] = len(expanded_sources)
        _add_evidence_diagnostics(
            diagnostics,
            question,
            local_sources,
            [],
            _local_file_evidence_answer(local_sources) if local_sources else None,
            bool(local_sources),
            None,
            False,
            self.settings,
        )
        if not local_sources:
            return RAGResponse(LOCAL_FILE_NOT_FOUND, [], [], False, "low", diagnostics)
        return RAGResponse(_local_file_evidence_answer(local_sources), local_sources, [], False, "local-grounded", diagnostics)

    def _search_web_sources(
        self,
        web_queries: Sequence[str],
        *,
        question: str = "",
        prefer_fast_public_search: bool = False,
    ) -> list[WebSource]:
        collected: list[WebSource] = []
        errors: list[str] = []
        target = max(1, min(MAX_WEB_SOURCES_TO_SHOW, self.settings.web_search_max_results))
        query_candidates = _dedupe_web_queries(web_queries)[:WEB_QUERY_FANOUT_LIMIT]
        if not query_candidates:
            return []

        first_query = query_candidates[0]
        if prefer_fast_public_search and _looks_like_public_knowledge_query(question):
            with ThreadPoolExecutor(max_workers=self._web_search_workers(len(query_candidates))) as executor:
                futures = [
                    executor.submit(self._search_duckduckgo_fallback, web_query)
                    for web_query in query_candidates
                ]
                for future in as_completed(futures):
                    collected = _merge_web_sources(
                        collected,
                        _future_result(future, []),
                        limit=target * WEB_QUERY_FANOUT_LIMIT,
                    )
            ranked = _rank_web_sources(collected, web_queries)
            if _has_usable_web_results(ranked[:target], target) and _public_topics_covered(
                question,
                ranked[:target],
            ):
                return ranked[:target]

        if prefer_fast_public_search and self.settings.web_search_provider != "duckduckgo":
            collected = _merge_web_sources(
                collected,
                self._search_duckduckgo_fallback(first_query),
                limit=target * WEB_QUERY_FANOUT_LIMIT,
            )
            ranked = _rank_web_sources(collected, web_queries)
            if _has_usable_web_results(ranked[:target], target) and _web_results_are_answerable_for_current_role(
                question,
                ranked[:target],
            ):
                return ranked[:target]

        try:
            collected = _merge_web_sources(
                collected,
                self.web_search.search(first_query),
                limit=target * WEB_QUERY_FANOUT_LIMIT,
            )
        except Exception as exc:
            errors.append(_clean_error_message(exc))

        ranked = _rank_web_sources(collected, web_queries)
        if _has_usable_web_results(ranked[:target], target) and _web_results_are_answerable_for_current_role(
            question,
            ranked[:target],
        ):
            return ranked[:target]

        remaining_queries = query_candidates[1:]
        if remaining_queries:
            with ThreadPoolExecutor(max_workers=self._web_search_workers(len(remaining_queries))) as executor:
                futures = {
                    executor.submit(self.web_search.search, web_query): web_query
                    for web_query in remaining_queries
                }
                for future in as_completed(futures):
                    try:
                        collected = _merge_web_sources(
                            collected,
                            future.result(),
                            limit=target * WEB_QUERY_FANOUT_LIMIT,
                        )
                    except Exception as exc:
                        errors.append(_clean_error_message(exc))

            ranked = _rank_web_sources(collected, web_queries)
            if _has_usable_web_results(ranked[:target], target) and _web_results_are_answerable_for_current_role(
                question,
                ranked[:target],
            ):
                return ranked[:target]

        if self.settings.web_search_provider != "duckduckgo":
            fallback_sources = self._search_duckduckgo_fallback(first_query)
            if fallback_sources:
                collected = _merge_web_sources(
                    collected,
                    fallback_sources,
                    limit=target * WEB_QUERY_FANOUT_LIMIT,
                )
                ranked = _rank_web_sources(collected, web_queries)
                if _has_usable_web_results(ranked[:target], target) and _web_results_are_answerable_for_current_role(
                    question,
                    ranked[:target],
                ):
                    return ranked[:target]

        ranked = _rank_web_sources(collected, web_queries)
        if self.settings.web_search_provider != "duckduckgo" and (
            not collected or not _has_usable_web_results(ranked[:target], target)
        ):
            fallback_sources = self._search_duckduckgo_fallback_queries(web_queries)
            if not fallback_sources:
                for web_query in web_queries[:WEB_QUERY_FANOUT_LIMIT]:
                    fallback_sources = _merge_web_sources(
                        fallback_sources,
                        self._search_duckduckgo_fallback(web_query),
                        limit=MAX_WEB_SOURCES_TO_SHOW,
                    )
                    if len(fallback_sources) >= MAX_WEB_SOURCES_TO_SHOW:
                        break
            collected = _merge_web_sources(collected, fallback_sources, limit=target * WEB_QUERY_FANOUT_LIMIT)
        ranked = _rank_web_sources(collected, web_queries)
        if (
            self.settings.enable_aggressive_web_fallback
            and not _has_usable_web_results(ranked[:target], target)
        ):
            expanded_sources = self._search_expanded_web_sources(
                question=question,
                web_queries=query_candidates,
                limit=target * WEB_QUERY_FANOUT_LIMIT,
            )
            collected = _merge_web_sources(
                collected,
                expanded_sources,
                limit=target * WEB_QUERY_FANOUT_LIMIT,
            )
        if not collected and errors:
            raise RuntimeError("; ".join(errors))
        return _rank_web_sources(collected, web_queries)[:target]

    def _web_search_workers(self, query_count: int) -> int:
        return max(1, min(int(getattr(self.settings, "web_search_max_workers", 3)), query_count))

    def _search_expanded_web_sources(
        self,
        *,
        question: str,
        web_queries: Sequence[str],
        limit: int,
    ) -> list[WebSource]:
        collected: list[WebSource] = []
        expanded_queries = _aggressive_web_queries(question, web_queries)
        fallback_max_results = max(
            self.settings.web_search_max_results,
            int(getattr(self.settings, "web_search_fallback_max_results", 12)),
        )
        for web_query in expanded_queries[:WEB_QUERY_FANOUT_LIMIT]:
            collected = _merge_web_sources(
                collected,
                self._provider_search(web_query, max_results=fallback_max_results),
                limit=limit,
            )
            if len(collected) >= limit:
                break
        return collected

    def _provider_search(self, query: str, *, max_results: int | None = None) -> list[WebSource]:
        if max_results is None or not hasattr(self.web_search, "max_results"):
            return self.web_search.search(query)
        previous = self.web_search.max_results
        try:
            self.web_search.max_results = max(1, int(max_results))
            return self.web_search.search(query)
        finally:
            self.web_search.max_results = previous

    def _search_duckduckgo_fallback(self, web_query: str) -> list[WebSource]:
        try:
            return DuckDuckGoSearch(
                max_results=self.settings.web_search_max_results,
                timeout_seconds=min(5.0, self.settings.web_search_timeout_seconds),
            ).search(web_query)
        except Exception:
            return []

    def _search_duckduckgo_fallback_queries(self, web_queries: Sequence[str]) -> list[WebSource]:
        collected: list[WebSource] = []
        for web_query in web_queries[:WEB_QUERY_FANOUT_LIMIT]:
            collected = _merge_web_sources(collected, self._search_duckduckgo_fallback(web_query), limit=MAX_WEB_SOURCES_TO_SHOW)
            if len(collected) >= MAX_WEB_SOURCES_TO_SHOW:
                break
        return collected

    @staticmethod
    def _is_sufficient(answer: str) -> bool:
        text = (answer or "").strip()
        return bool(text) and not any(marker in text.lower() for marker in INSUFFICIENT_MARKERS)

    @staticmethod
    def _is_web_requested(question: str) -> bool:
        return any(marker in question.lower() for marker in WEB_REQUEST_MARKERS)

    @staticmethod
    def _is_local_file_question(question: str) -> bool:
        lower = question.lower().strip()
        action_markers = (
            "are there",
            "can you find",
            "contain",
            "contains",
            "do i have",
            "do my",
            "do the uploaded",
            "do you have",
            "does",
            "find",
            "has",
            "have",
            "is ",
            "mention",
            "mentions",
            "there",
            "where",
            "which",
        )
        if any(marker in lower for marker in LOCAL_FILE_MARKERS) and any(
            marker in lower for marker in action_markers
        ):
            return True
        intent_patterns = (
            r"\b(?:do|does|did|is|are|any)\b.+\b(?:my|uploaded|indexed|local)\s+(?:file|files|document|documents)\b",
            r"\b(?:which|what)\s+(?:file|files|document|documents)\b",
            r"\bin\s+(?:my\s+|the\s+)?(?:uploaded\s+|indexed\s+|local\s+)?(?:file|document)\b",
            r"\bin\s+(?:my|the\s+uploaded|uploaded|the\s+indexed|indexed|local)\s+(?:file|files|document|documents)\b",
            r"\b(?:do\s+my|does\s+my|do\s+the\s+uploaded|is\s+there).+\b(?:document|file|upload|index|local)\b",
        )
        return any(re.search(pattern, lower) for pattern in intent_patterns)

    @staticmethod
    def _confidence(*, local_sources, used_web, answer, evidence_confidence, time_sensitive) -> str:
        lower = answer.lower()
        explicit_confidence = re.search(r"\bconfidence:\s*(high|medium|low)\b", lower)
        if explicit_confidence:
            return explicit_confidence.group(1)
        if is_model_selection_warning(answer):
            return "model-selection-warning"
        if any(marker in lower for marker in INSUFFICIENT_MARKERS):
            return "low"
        if time_sensitive and used_web:
            return "current-information"
        if used_web and local_sources:
            return "local-web-assisted"
        if used_web:
            return "web-assisted"
        if local_sources:
            return "local-grounded"
        return "model-only"


@lru_cache(maxsize=8)
def get_rag_service(settings: AppSettings) -> VerilumeRAG:
    return VerilumeRAG(settings)


def _future_result(future: Future, default):
    try:
        return future.result()
    except Exception:
        return default


def _attach_conversation_state(
    response: RAGResponse,
    state: ConversationState,
    original_question: str,
    resolved_question: str,
) -> RAGResponse:
    state.last_resolved_question = resolved_question or state.last_resolved_question
    response.conversation_state = state
    response.original_query = original_question
    response.resolved_query = resolved_question
    response.diagnostics.setdefault("original_query", original_question)
    response.diagnostics.setdefault("resolved_query", resolved_question)
    response.diagnostics.setdefault("conversation_roles", state.roles)
    response.diagnostics.setdefault("roles", state.roles)
    return response


def _merged_conversation_state_for_interpretation(
    inferred: ConversationState,
    provided: ConversationState | None,
) -> ConversationState:
    if provided is None:
        return inferred
    merged = ConversationState(
        active_topic=provided.active_topic or inferred.active_topic,
        active_country=provided.active_country or inferred.active_country,
        active_person=provided.active_person or inferred.active_person,
        active_document=provided.active_document or inferred.active_document,
        active_news_story=provided.active_news_story or inferred.active_news_story,
        entities=[*provided.entities, *inferred.entities],
        roles={**inferred.roles, **provided.roles},
        preferred_sources=_unique_nonempty([*provided.preferred_sources, *inferred.preferred_sources]),
        last_answer_summary=provided.last_answer_summary or inferred.last_answer_summary,
        last_resolved_question=provided.last_resolved_question or inferred.last_resolved_question,
        active_entities=_unique_nonempty([*provided.active_entities, *inferred.active_entities]),
        active_topics=_unique_nonempty([*provided.active_topics, *inferred.active_topics]),
        active_documents=_unique_nonempty([*provided.active_documents, *inferred.active_documents]),
        active_web_sources=_unique_nonempty([*provided.active_web_sources, *inferred.active_web_sources]),
        active_dates=_unique_nonempty([*provided.active_dates, *inferred.active_dates]),
        active_role=provided.active_role or inferred.active_role,
        active_company=provided.active_company or inferred.active_company,
        active_organization=provided.active_organization or inferred.active_organization,
        active_law=provided.active_law or inferred.active_law,
        active_research_topic=provided.active_research_topic or inferred.active_research_topic,
        active_dataset=provided.active_dataset or inferred.active_dataset,
        intent=provided.intent or inferred.intent,
        expires_after=provided.expires_after or inferred.expires_after,
        active_event=provided.active_event or inferred.active_event,
    )
    return merged


def _response_cache_key(
    question: str,
    history: Sequence[ChatMessage],
    conversation_state: ConversationState | None = None,
) -> tuple:
    normalized = normalize_query(question)
    normalized_question = normalized.canonical or re.sub(
        r"\s+",
        " ",
        (question or "").strip().lower(),
    )
    state_key = ()
    if conversation_state is not None and _query_needs_context_cache_key(question):
        state_key = (
            conversation_state.active_country,
            conversation_state.active_person,
            conversation_state.active_role,
            tuple(sorted(conversation_state.roles.items())),
            tuple(conversation_state.preferred_sources),
        )
    return normalized_question, state_key


def _query_needs_context_cache_key(question: str) -> bool:
    normalized = normalize_intent_text(question)
    if re.search(r"\b(?:he|him|his|she|her|they|them|their|it|its|this|that|same|latter)\b", normalized):
        return True
    role = _government_role_from_text(normalized)
    return bool(role and not _country_from_text(question))


def _should_answer_local_file_question_directly(question: str) -> bool:
    normalized = normalize_intent_text(question)
    return bool(
        normalized.startswith(("which document", "which file", "what document", "what file"))
        or re.search(r"\b(?:contains?|mentions?|where)\b", normalized)
    )


def _response_cache_ttl(response: RAGResponse) -> float:
    diagnostics = response.diagnostics or {}
    if diagnostics.get("requires_date_reconciliation"):
        return CURRENT_RESPONSE_CACHE_TTL_SECONDS
    if response.confidence in {"needs-token", "model-selection-warning", "generation-error"}:
        return 0.0
    return DEFAULT_RESPONSE_CACHE_TTL_SECONDS


def _should_rewrite_query(
    question: str,
    history: Sequence[ChatMessage],
    *,
    min_history: int = 1,
    similarity_threshold: float = 0.92,
) -> bool:
    if not history:
        return False
    user_turns = [item.content for item in history if getattr(item, "role", "") == "user"]
    if len(user_turns) < max(0, min_history):
        return False
    normalized = re.sub(r"[^a-z0-9' ]+", " ", (question or "").lower()).strip()
    if not normalized:
        return False
    if _is_lightweight_chat(question):
        return False
    previous_user = re.sub(r"[^a-z0-9' ]+", " ", (user_turns[-1] if user_turns else "").lower()).strip()
    if previous_user and SequenceMatcher(None, normalized, previous_user).ratio() >= similarity_threshold:
        return False
    context_markers = {
        "above",
        "earlier",
        "he",
        "her",
        "him",
        "his",
        "it",
        "its",
        "previous",
        "same",
        "she",
        "that",
        "their",
        "them",
        "these",
        "they",
        "this",
        "those",
    }
    tokens = set(normalized.split())
    if tokens & context_markers:
        return True
    return len(tokens) <= 4 and normalized.startswith(("and ", "also ", "what about", "how about"))


def _is_lightweight_chat(question: str) -> bool:
    return not IntentRouterAgent().route(question).uses_rag


def _lightweight_chat_answer(question: str) -> str:
    return IntentRouterAgent().route(question).answer


def _should_use_web(
    *,
    question: str,
    force_web: bool,
    web_enabled: bool,
    query_understanding,
) -> bool:
    if not web_enabled:
        return False
    if force_web:
        return True
    if any(marker in question.lower() for marker in WEB_REQUEST_MARKERS):
        return True
    return _requires_current_source_verification(question, query_understanding)


def _should_skip_local_retrieval(question: str, query_understanding, local_file_question: bool) -> bool:
    return False


def _looks_like_public_knowledge_query(question: str) -> bool:
    lower = re.sub(r"\s+", " ", (question or "").lower()).strip()
    if not lower:
        return False
    if any(marker in lower for marker in LOCAL_FILE_MARKERS):
        return False
    if _public_topic_from_text(question):
        return True
    if not any(marker in lower for marker in PUBLIC_KNOWLEDGE_MARKERS):
        return False
    parts = [part.strip() for part in re.split(r"\?+|\n+|(?:\s+and\s+what\s+is\s+)", lower) if part.strip()]
    if not parts:
        parts = [lower]
    return any(part.startswith(PUBLIC_DEFINITION_PREFIXES) for part in parts)


def _looks_like_news_query(question: str) -> bool:
    lower = (question or "").lower()
    if any(marker in lower for marker in ("news", "reuters", "ap news", "bbc", "sky news", "financial times", "guardian")):
        return True
    return any(marker in lower for marker in ("resign", "resigned", "resignation", "breaking"))


def _requires_current_source_verification(question: str, query_understanding=None) -> bool:
    """Return True only for facts that may be stale without source evidence."""

    normalized = normalize_intent_text(question)
    if not normalized:
        return False
    if _is_stable_model_knowledge_question(question):
        return False
    if _looks_like_office_start_query(question) or _looks_like_age_at_office_query(question):
        return False
    if _looks_like_news_query(question):
        return True
    if any(
        marker in normalized
        for marker in (
            "current",
            "latest",
            "most recent",
            "newest",
            "recent",
            "today",
            "now",
            "this year",
            "2026",
            "2025",
            "weather",
            "price",
            "prices",
            "stock",
            "schedule",
            "deadline",
            "law",
            "regulation",
            "directive",
            "ceo",
        )
    ):
        return True
    if _is_public_office_query(question, query_understanding or classify_question(question)):
        return True
    return bool(getattr(query_understanding, "requires_date_reconciliation", False))


def _is_stable_model_knowledge_question(question: str) -> bool:
    normalized = normalize_intent_text(question)
    if not normalized:
        return False
    if _looks_like_news_query(question):
        return False
    if re.search(r"\b(?:latest|current|recent|today|now|this year|breaking|resigned?|resignation)\b", normalized):
        return False
    if _looks_like_scientific_local_query(question) or _looks_like_public_knowledge_query(question):
        return True
    if normalized.startswith(
        (
            "define ",
            "explain ",
            "what is ",
            "what are ",
            "how does ",
            "how do ",
            "why does ",
            "why do ",
            "how to ",
        )
    ):
        return not any(marker in normalized for marker in PUBLIC_OFFICE_MARKERS)
    return bool(
        re.search(
            r"\b(?:smallest|largest|biggest|capital|area|population|continent|ocean|mountain|river|country in europe)\b",
            normalized,
        )
    )


def _looks_like_age_at_office_query(question: str) -> bool:
    lower = (question or "").lower()
    return lower.startswith("how old") and any(
        marker in lower for marker in ("became", "become", "took office", "took power", "assumed office")
    )


def _looks_like_office_start_query(question: str) -> bool:
    lower = (question or "").lower()
    if not lower.startswith("when "):
        return False
    if not _has_office_start_marker(lower):
        return False
    return any(marker in lower for marker in PUBLIC_OFFICE_MARKERS)


def _has_office_start_marker(lower: str) -> bool:
    return any(
        marker in lower
        for marker in (
            "became",
            "become",
            "came into power",
            "came to power",
            "come into power",
            "come to power",
            "in power",
            "took office",
            "took power",
            "assumed office",
        )
    )


def _is_public_office_query(question: str, query_understanding) -> bool:
    if not getattr(query_understanding, "requires_date_reconciliation", False):
        return False
    lower = (question or "").lower()
    return any(marker in lower for marker in PUBLIC_OFFICE_MARKERS)


def _local_search_queries(
    query: str,
    identity_tokens: Sequence[str],
    local_file_question: bool,
) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (query or "").strip())
    stripped = _strip_question_prefix(cleaned)
    normalized = _normalize_scientific_query(stripped)
    candidates = [cleaned]
    if _query_variant_is_useful(cleaned, stripped):
        candidates.append(stripped)
    if _query_variant_is_useful(cleaned, normalized):
        candidates.append(normalized)

    if identity_tokens and not local_file_question:
        name = " ".join(identity_tokens)
        candidates.extend(
            [
                name,
                f"{name} profile",
                f"{name} publications",
                f"{name} research",
            ]
        )

    if _looks_like_scientific_local_query(cleaned):
        candidates.extend(_scientific_local_queries(normalized))

    return _unique_nonempty(candidates)[:6]


def _query_variant_is_useful(original: str, candidate: str) -> bool:
    original_key = re.sub(r"[^a-z0-9]+", " ", (original or "").lower()).strip()
    candidate_key = re.sub(r"[^a-z0-9]+", " ", (candidate or "").lower()).strip()
    if not candidate_key or candidate_key == original_key:
        return False
    terms = query_terms(candidate)
    return len(terms) >= 2 or _looks_like_scientific_local_query(candidate)


def _strip_question_prefix(query: str) -> str:
    value = (query or "").strip()
    patterns = (
        r"^\s*what\s+is\s+",
        r"^\s*what\s+are\s+",
        r"^\s*who\s+is\s+",
        r"^\s*tell\s+me\s+about\s+",
        r"^\s*explain\s+",
        r"^\s*define\s+",
    )
    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    return value.strip(" ?!.:;-") or query.strip()


def _normalize_scientific_query(query: str) -> str:
    value = re.sub(r"\bmonte\s*carlo\b", "Monte Carlo", query or "", flags=re.IGNORECASE)
    value = re.sub(r"\bhamiltonian\s+montecarlo\b", "Hamiltonian Monte Carlo", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value or query.strip()


def _looks_like_scientific_local_query(query: str) -> bool:
    lower = (query or "").lower()
    return any(
        marker in lower
        for marker in (
            "hamiltonian",
            "monte carlo",
            "montecarlo",
            "replica exchange",
            "thermodynamic integration",
            "bayesian",
            "mcmc",
            "paper",
            "thesis",
            "publication",
        )
    )


def _scientific_local_queries(query: str) -> list[str]:
    lower = (query or "").lower()
    candidates = [query]
    if "replica exchange" in lower and "hamiltonian" in lower:
        candidates.extend(
            [
                "Replica Exchange Hamiltonian Monte Carlo",
                "Replica Exchange Monte Carlo",
                "Hamiltonian Monte Carlo",
                "H-REMC",
            ]
        )
    elif "hamiltonian" in lower and "monte carlo" in lower:
        candidates.extend(["Hamiltonian Monte Carlo", "HMC"])
    if "thermodynamic integration" in lower:
        candidates.append("thermodynamic integration replica exchange")
    if query:
        candidates.extend(
            [
                f"{query} thesis",
                f"{query} paper",
                f"{query} publication",
                f"{query} abstract",
            ]
        )
    return candidates


def _setting(settings: AppSettings, name: str, default):
    return getattr(settings, name, default)


def _best_local_score(local_sources: Sequence[LocalSource]) -> float:
    scores = [float(getattr(source, "score", 0.0) or 0.0) for source in local_sources]
    return max(scores) if scores else 0.0


def _local_evidence_looks_strong(local_sources: Sequence[LocalSource], settings: AppSettings) -> bool:
    if not local_sources:
        return False
    strong_score = float(_setting(settings, "strong_local_score_threshold", 0.72))
    min_sources = int(_setting(settings, "strong_local_min_sources", 1))
    return len(local_sources) >= min_sources and _best_local_score(local_sources) >= strong_score


def _add_evidence_diagnostics(
    diagnostics,
    question,
    local_sources,
    web_sources,
    local_answer,
    local_sufficient,
    model_answer,
    model_sufficient,
    settings=None,
):
    query_understanding = classify_question(question)
    evidence_items = evidence_from_sources(local_sources=local_sources, web_sources=web_sources, ai_answer=model_answer if model_sufficient else None)
    ranked_evidence = rank_evidence(evidence_items, query_understanding, settings=settings)
    reconciliation = reconcile_dates(ranked_evidence, query_understanding)
    resolution = resolve_evidence_conflicts(question, ranked_evidence, local_answer=local_answer if local_sufficient else None, ai_knowledge_answer=model_answer if model_sufficient else None, web_answer=None, query=query_understanding, reconciliation=reconciliation)
    used_local = bool(local_sources and local_sufficient)
    used_model = bool(model_sufficient and _model_answer_available(model_answer))
    used_web = bool(web_sources)
    diagnostics["used_local"] = used_local
    diagnostics["used_model_knowledge"] = used_model
    diagnostics["used_web"] = used_web
    diagnostics["model_knowledge_available"] = _model_answer_available(model_answer)
    diagnostics["web_enabled"] = bool(getattr(settings, "enable_web_search", False)) if settings is not None else False
    diagnostics["evidence_streams"] = [
        stream
        for stream, enabled in (
            ("local", used_local),
            ("model_knowledge", used_model),
            ("web", used_web),
        )
        if enabled
    ]
    diagnostics["evidence_winner"] = _diagnostic_evidence_winner(
        resolution.winner,
        used_local=used_local,
        used_model=used_model,
        used_web=used_web,
    )
    diagnostics["evidence_note"] = resolution.evidence_note
    diagnostics["source_agreement"] = resolution.source_agreement
    diagnostics["freshness_note"] = reconciliation.freshness_note
    diagnostics["local_is_older_than_web"] = reconciliation.local_is_older_than_web
    return ranked_evidence, resolution


def _model_answer_available(model_answer: str | None) -> bool:
    text = (model_answer or "").strip()
    return bool(text and not any(marker in text.lower() for marker in INSUFFICIENT_MARKERS))


def _diagnostic_evidence_winner(winner, *, used_local: bool, used_model: bool, used_web: bool) -> str | None:
    if used_local and used_model and used_web:
        return "hybrid"
    if winner is None:
        return None
    value = getattr(winner, "value", str(winner))
    if value == "ai_knowledge":
        return "model_knowledge"
    return value


def _local_answer_supports_question(
    question: str,
    answer: str,
    local_sources: Sequence[LocalSource],
    *,
    local_file_question: bool,
) -> bool:
    if local_file_question:
        return True
    text = (answer or "").strip()
    if not text:
        return False
    normalized = normalize_query(question)
    query_terms = {term.lower() for term in normalized.key_terms if len(term) > 2}
    entity_terms = {
        term.lower()
        for entity in normalized.entities
        for term in re.findall(r"[a-z0-9][a-z0-9'-]*", entity.lower())
        if len(term) > 2
    }
    cited_sources = _local_sources_used_in_answer(local_sources, answer) or list(local_sources[:3])
    haystacks = [text.lower(), *[(source.text or "").lower() for source in cited_sources]]
    combined = " ".join(haystacks)
    if cited_sources and any(marker in question.lower() for marker in WEB_REQUEST_MARKERS):
        return True
    if entity_terms and not any(term in combined for term in entity_terms):
        return False
    if not query_terms:
        return True
    if len(query_terms) <= 1 and cited_sources:
        return True
    overlap = {term for term in query_terms if term in combined}
    minimum_overlap = 1 if len(query_terms) <= 3 else 2
    return len(overlap) >= minimum_overlap


def _model_answer_supports_question(question: str, answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False
    normalized = normalize_query(question)
    query_terms = [
        term.lower()
        for term in normalized.key_terms
        if len(term) > 2 and term.lower() not in {"web", "search", "online", "internet"}
    ]
    entity_terms = [
        term.lower()
        for entity in normalized.entities
        for term in re.findall(r"[a-z0-9][a-z0-9'-]*", entity.lower())
        if len(term) > 2
    ]
    if entity_terms and not any(term in text for term in entity_terms):
        return False
    if not query_terms:
        return True
    overlap = sum(1 for term in query_terms if term in text)
    if len(query_terms) <= 2:
        return overlap >= 1
    return overlap >= min(2, len(query_terms))


def _local_sources_used_in_answer(local_sources: Sequence[LocalSource], answer: str) -> list[LocalSource]:
    labels = _labels_in_answer(answer, "S")
    return [source for source in local_sources if source.label in labels]


def _web_sources_used_in_answer(web_sources: Sequence[WebSource], answer: str) -> list[WebSource]:
    labels = _labels_in_answer(answer, "W")
    return [source for source in web_sources if source.label in labels]


def _merge_local_sources(primary, secondary, limit):
    merged, seen = [], set()
    for source in [*primary, *secondary]:
        key = source.chunk_id or f"{source.document}:{source.page}:{source.text[:80]}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
        if len(merged) >= limit:
            break
    return _relabel_local_sources(merged)


def _local_file_evidence_answer(sources):
    lines = [f"Confidence: {_local_file_confidence(sources)}", "", "Matching indexed local files:"]
    for source in sources[:5]:
        page = f", page {source.page}" if getattr(source, "page", None) else ""
        preview = _compact_source_text(source.text, limit=240)
        lines.append(f"- [{source.label}] {source.document}{page}: {preview}" if preview else f"- [{source.label}] {source.document}{page}")
    if len(sources) > 5:
        lines.append(f"Additional matching chunks found: {len(sources) - 5}.")
    return "\n".join(lines)


def _local_file_confidence(sources):
    if not sources:
        return "Low"
    best = sources[0]
    metadata = best.metadata or {}
    fast_score = float(metadata.get("fast_rerank_score", best.score or 0.0) or 0.0)
    coverage = float(metadata.get("query_coverage", 0.0) or 0.0)
    overlap = int(metadata.get("query_overlap", 0) or 0)
    if (best.score >= 0.82 and overlap >= 2) or (fast_score >= 0.68 and coverage >= 0.34 and overlap >= 2):
        return "High"
    if best.score >= 0.62 or coverage >= 0.2 or overlap >= 2:
        return "Medium"
    return "Low"


def _rank_and_filter_local_file_sources(rag, ranking_query, sources, source_limit):
    ranked = rag._rerank_local(ranking_query, sources)
    return _filter_relevant_local_sources(
        ranking_query,
        ranked,
        identity_tokens=(),
        local_file_question=True,
        limit=source_limit,
    )


def _local_file_sources_are_enough(sources):
    if not sources:
        return False
    best = sources[0]
    metadata = best.metadata or {}
    fast_score = float(metadata.get("fast_rerank_score", best.score or 0.0) or 0.0)
    coverage = float(metadata.get("query_coverage", 0.0) or 0.0)
    overlap = int(metadata.get("query_overlap", 0) or 0)
    return best.score >= 0.84 or (fast_score >= 0.68 and coverage >= 0.34 and overlap >= 2)


def _merge_web_sources(existing, candidates, limit):
    best_by_key = {}
    for source in [*existing, *candidates]:
        key = _web_source_key(source)
        if not key:
            continue
        previous = best_by_key.get(key)
        if previous is None or _web_source_dedupe_score(source) > _web_source_dedupe_score(previous):
            best_by_key[key] = source
    merged = list(best_by_key.values())[:limit]
    return _relabel_web_sources(merged)


def _rank_web_sources(web_sources, web_queries):
    query_text = " ".join(web_queries)
    query_terms = _rank_terms(query_text)
    domain = classify_query_domain(query_text)
    boosted = boost_priority_sources(list(web_sources), domain)
    ranked = sorted(boosted, key=lambda source: _web_source_rank_score(source, query_terms), reverse=True)
    non_noisy = [source for source in ranked if not _is_noisy_web_source(source)]
    if len(non_noisy) >= MAX_WEB_SOURCES_TO_SHOW:
        ranked = non_noisy + [source for source in ranked if _is_noisy_web_source(source)]
    return _relabel_web_sources(ranked)


def _has_usable_web_results(web_sources, target):
    if not web_sources:
        return False
    non_noisy = [source for source in web_sources if not _is_noisy_web_source(source)]
    if len(non_noisy) >= target:
        return True
    return bool(non_noisy and any(_is_authoritative_web_source(source) for source in non_noisy))


def _web_results_are_answerable_for_current_role(question, web_sources) -> bool:
    if _is_resignation_news_query(question):
        return True
    context = _current_public_role_context(question)
    if not context:
        return True
    role = context["role"]
    for source in web_sources:
        if role == "secretary of state" and not _is_us_secretary_of_state_source(source):
            continue
        if _extract_role_candidate(source, role):
            return True
    return False


def _web_source_rank_score(source, query_terms):
    haystack = f"{source.title} {source.url} {source.content}".lower()
    domain = source.url.lower()
    title = (source.title or "").lower()
    score = 0.0
    if _is_authoritative_web_source(source):
        score += 4.0
    elif any(m in domain for m in ("university", ".edu", "uni.", "ac.")):
        score += 3.0
    elif any(m in domain for m in ("arxiv.org", "doi.org", "ieee.org", "nature.com")):
        score += 2.8
    elif any(m in domain for m in ("reuters.", "apnews.", "bbc.", "euronews.")):
        score += 2.0
    if any(term in query_terms for term in ("news", "reuters", "resign", "resigned", "resignation", "why")):
        score += _news_source_rank_boost(source, query_terms)
    overlap = sum(1 for term in query_terms if term in haystack)
    score += min(2.0, overlap * 0.25)
    if any(marker in haystack for marker in ("current role holder", "incumbent", "has been king", "is the prime minister", "is prime minister", "is the president", "is president", "is the secretary of state", "serves as secretary of state", "sworn in as secretary of state")):
        score += 0.75
    if "secretary" in query_terms and "state" in query_terms:
        if _is_us_secretary_of_state_source(source):
            score += 3.0
        elif "secretary of state" in haystack:
            score -= 2.0
    if title.startswith(("prime minister", "president ", "secretary of state", "king ")):
        score += 0.5
    if _is_authoritative_web_source(source) and title.startswith(("prime minister", "president ", "secretary of state", "king ")):
        score += 1.25
    if any(marker in haystack for marker in ("was prime minister", "former prime minister", "between 25", "between 20", "former president")):
        score -= 2.5
    if any(marker in haystack for marker in ("blog", "opinion", "editorial")):
        score -= 1.5
    if isinstance(source.score, (int, float)):
        score += min(1.0, max(0.0, float(source.score)))
    if _is_noisy_web_source(source):
        score -= 2.5
    source.score = max(float(source.score or 0.0), min(1.0, score / 6.0))
    return score


def _is_authoritative_web_source(source):
    domain = _web_source_domain(source)
    url = (source.url or "").lower()
    title = (source.title or "").lower()
    if any(marker in domain or marker in url for marker in OFFICIAL_WEB_MARKERS):
        return True
    return any(marker in title for marker in ("royal house", "white house"))


def _news_source_rank_boost(source, query_terms) -> float:
    haystack = f"{source.title} {source.url} {source.content}".lower()
    domain = _web_source_domain(source)
    score = 0.0
    if any(marker in domain for marker in NEWS_WEB_MARKERS):
        score += 2.0
    requested = {
        "reuters": ("reuters.com", "reuters"),
        "bbc": ("bbc.com", "bbc.co.uk", "bbc news"),
        "guardian": ("theguardian.com", "guardian"),
    }
    if "sky" in query_terms:
        requested["sky"] = ("news.sky.com", "sky news")
    if "financial" in query_terms or "times" in query_terms:
        requested["financial times"] = ("ft.com", "financial times")
    if "ap" in query_terms:
        requested["ap"] = ("apnews.com", "associated press", "ap news")
    for term, markers in requested.items():
        if term not in " ".join(query_terms):
            continue
        if any(marker in domain or marker in haystack for marker in markers):
            score += 3.0
    if any(marker in haystack for marker in ("resigned", "resignation", "resigns", "stepped down")):
        score += 1.0
    return score


def _is_noisy_web_source(source):
    haystack = f"{source.title} {source.url} {source.content}".lower()
    return any(term in haystack for term in ("facebook.com", "instagram.com", "linkedin.com/posts", "reddit.com", "tiktok.com", "twitter.com", "x.com", "youtube.com", "youtu.be"))


def _rank_terms(text):
    stopwords = {"about", "and", "are", "current", "for", "from", "latest", "official", "reliable", "search", "source", "the", "who"}
    return {term for term in re.findall(r"[a-z][a-z0-9'-]{2,}", text.lower()) if term not in stopwords}


def _identity_tokens(question):
    lower = (question or "").strip().lower()
    normalized = normalize_query(question)
    if (
        any(term in {"area", "population", "capital"} for term in normalized.key_terms)
        and not lower.startswith(("who ", "who is", "who's"))
    ):
        return []
    if _government_role_from_text(normalize_intent_text(question)) or _current_public_role_context(question):
        return []
    if _looks_like_scientific_local_query(question) and lower.startswith(
        ("what is", "what are", "explain", "define", "how does", "how do")
    ):
        return []
    if _looks_like_age_at_office_query(question):
        person = _age_query_person(question)
        return _unique_nonempty(_identity_words(person)) if person else []
    words = _identity_words(question)
    bare_entity = _looks_like_bare_entity_query(words)
    tokens = []
    for word in words:
        clean = word.strip("'’").lower()
        if clean in IDENTITY_STOPWORDS or len(clean) <= 2:
            continue
        if bare_entity or word[:1].isupper():
            tokens.append(clean)
    unique = list(dict.fromkeys(tokens))
    return unique if len(unique) >= 2 else []


def _identity_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", text or "")


def _looks_like_bare_entity_query(words):
    if not 2 <= len(words) <= 5:
        return False
    normalized = [word.strip("'’").lower() for word in words]
    if any(word in IDENTITY_STOPWORDS or len(word) <= 2 for word in normalized):
        return False
    leading = words[0].lower()
    return leading not in {"what", "where", "when", "why", "how", "does", "do", "is", "are"}


def _filter_local_sources_for_identity(sources, identity_tokens):
    if not identity_tokens:
        return list(sources)
    return _relabel_local_sources(
        [
            source
            for source in sources
            if _source_matches_identity(f"{source.document} {source.text}", identity_tokens)
            and _identity_local_source_is_relevant(source, identity_tokens)
        ]
    )


def _filter_relevant_local_sources(
    query,
    sources,
    *,
    identity_tokens,
    local_file_question,
    limit,
):
    values = list(sources)
    if not values:
        return []

    terms = query_terms(query)
    if not terms:
        return _relabel_local_sources(values[:limit])
    scientific_query = _looks_like_scientific_local_query(query)

    if local_file_question:
        rare_term_values = _rare_matched_terms(values, terms)
        if rare_term_values:
            rare_values = [
                source
                for source in values
                if any(term in _normalized_source_tokens(source) for term in rare_term_values)
            ]
            if rare_values:
                values = rare_values

    filtered = []
    for source in values:
        metadata = source.metadata or {}
        overlap = int(metadata.get("query_overlap", 0) or 0)
        coverage = float(metadata.get("query_coverage", 0.0) or 0.0)
        fast_score = float(metadata.get("fast_rerank_score", source.score or 0.0) or 0.0)
        retrieval = str(metadata.get("retrieval", "")).lower()
        semantic_score = float(source.score or 0.0)

        if identity_tokens and not _source_matches_identity(
            f"{source.document} {source.text}",
            identity_tokens,
        ):
            continue
        if identity_tokens and not local_file_question and not _identity_local_source_is_relevant(
            source,
            identity_tokens,
        ):
            continue
        if scientific_query and not local_file_question and not _scientific_local_source_is_relevant(
            source,
        ):
            continue

        if local_file_question:
            keep = (
                overlap >= 2
                or coverage >= 0.24
                or ("bm25" in retrieval and overlap >= 1 and fast_score >= 0.58)
                or semantic_score >= 0.82
            )
        else:
            keep = (
                overlap >= 1
                or coverage >= 0.18
                or fast_score >= 0.62
                or semantic_score >= 0.78
            )

        if keep:
            filtered.append(source)

    return _relabel_local_sources(filtered[:limit])


def _rare_matched_terms(sources, terms):
    if len(sources) <= 1:
        return []
    source_terms = [_normalized_source_tokens(source) for source in sources]
    frequencies = {
        term: sum(1 for tokens in source_terms if term in tokens)
        for term in terms
    }
    max_frequency = max(1, int(len(sources) * 0.25))
    rare_terms = [
        term
        for term, frequency in frequencies.items()
        if 0 < frequency <= max_frequency and len(term) >= 6
    ]
    return sorted(rare_terms, key=lambda term: (frequencies[term], -len(term)))


def _normalized_source_text(source):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        f"{getattr(source, 'document', '')} {getattr(source, 'title', '')} {getattr(source, 'text', '')} {getattr(source, 'content', '')}".lower(),
    )


def _normalized_source_tokens(source):
    return set(query_terms(_normalized_source_text(source)))


def _filter_web_sources_for_identity(sources, identity_tokens):
    if not identity_tokens:
        return list(sources)
    return _relabel_web_sources([s for s in sources if not _is_noisy_web_source(s) and _source_matches_identity(f"{s.title} {s.url} {s.content}", identity_tokens)])


def _source_matches_identity(text, identity_tokens):
    source_tokens = set(re.findall(r"[a-z][a-z'’-]+", _fold_text(text)))
    folded_tokens = {_fold_identity_token(token) for token in source_tokens}
    for token in identity_tokens:
        folded = _fold_identity_token(token)
        if token in source_tokens or folded in folded_tokens:
            continue
        if len(token) >= 5 and any(
            SequenceMatcher(None, folded, candidate).ratio() >= 0.84
            for candidate in folded_tokens
        ):
            continue
        return False
    return True


def _identity_local_source_is_relevant(source, identity_tokens) -> bool:
    text = _normalized_source_text(source)
    document = re.sub(r"[^a-z0-9]+", " ", getattr(source, "document", "").lower())
    contains_identity = _source_matches_identity(f"{document} {text}", identity_tokens)
    if not contains_identity:
        return False

    signal_text = _strip_negated_identity_profile_markers(text)
    has_strong_signal = any(marker in signal_text for marker in IDENTITY_LOCAL_STRONG_MARKERS)
    has_context_signal = any(marker in text for marker in IDENTITY_LOCAL_CONTEXT_MARKERS)
    has_negative_signal = any(marker in text for marker in IDENTITY_LOCAL_NEGATIVE_MARKERS)
    document_has_identity = all(_fold_identity_token(token) in document for token in identity_tokens)

    if has_negative_signal and not has_strong_signal:
        return False
    if document_has_identity:
        return True
    if has_strong_signal:
        return True
    if has_context_signal and _identity_appears_near_context_signal(source, identity_tokens):
        return True
    return False


def _strip_negated_identity_profile_markers(text: str) -> str:
    profile_markers = (
        "about",
        "bio",
        "biography",
        "cv",
        "curriculum vitae",
        "portfolio",
        "profile",
        "resume",
    )
    marker_pattern = "|".join(re.escape(marker) for marker in profile_markers)
    return re.sub(
        rf"\b(?:not|is not|isn t|isnt|without)\s+(?:a\s+|an\s+|the\s+)?"
        rf"(?:{marker_pattern})(?:\s+or\s+(?:{marker_pattern}))*\b",
        " ",
        text or "",
    )


def _identity_appears_near_context_signal(source, identity_tokens) -> bool:
    raw_text = f"{getattr(source, 'document', '')} {getattr(source, 'text', '')}"
    folded = _fold_text(raw_text)
    if not folded:
        return False
    name_pattern = r"\s+".join(re.escape(_fold_identity_token(token)) for token in identity_tokens)
    match = re.search(name_pattern, folded)
    if not match:
        return False
    start = max(0, match.start() - 140)
    end = min(len(folded), match.end() + 180)
    window = folded[start:end]
    strong_or_context = (*IDENTITY_LOCAL_STRONG_MARKERS, "doctoral researcher", "profile page")
    return any(marker in window for marker in strong_or_context)


def _scientific_local_source_is_relevant(source) -> bool:
    text = _normalized_source_text(source)
    document = re.sub(r"[^a-z0-9]+", " ", getattr(source, "document", "").lower())
    has_positive_signal = any(marker in text for marker in SCIENTIFIC_LOCAL_POSITIVE_MARKERS)
    has_negative_document_signal = any(
        marker in document for marker in SCIENTIFIC_LOCAL_NEGATIVE_MARKERS
    )
    if has_negative_document_signal and not any(
        marker in document for marker in ("paper", "publication", "thesis", "dissertation")
    ):
        return False
    return has_positive_signal or float(getattr(source, "score", 0.0) or 0.0) >= 0.86


def _fold_identity_token(token):
    return _fold_text(token).replace("ph", "f").replace("’", "'")


def _current_public_fact_answer(question, web_sources):
    if _is_resignation_news_query(question):
        return None
    context = _current_public_role_context(question)
    evidence = _current_public_fact_evidence(question, web_sources)
    if not context or not evidence:
        return None

    best_source, best_candidate = evidence[0]
    candidate = best_candidate or _extract_role_candidate(best_source, context["role"])
    if not candidate:
        return None

    candidate_names = [name for _source, name in evidence if name]
    evidence_conflict = _candidate_names_conflict(candidate_names)
    citation = f"[{best_source.label}]"

    if context["kind"] == "no-monarch":
        answer = (
            "The United States does not have a king or queen. "
            f"The current president of the United States is {candidate} {citation}."
        )
        return answer, False

    return f"{context['subject']} is {candidate} {citation}.", evidence_conflict


def _web_source_key(source):
    url = normalize_web_url_key(getattr(source, "url", "") or "")
    return url or f"{getattr(source, 'title', '').strip().lower()}::{getattr(source, 'content', '').strip().lower()[:80]}"


def _web_source_dedupe_score(source: WebSource) -> float:
    score = float(getattr(source, "score", 0.0) or 0.0)
    if _is_authoritative_web_source(source):
        score += 1.0
    if getattr(source, "content", ""):
        score += min(0.5, len(source.content) / 1000)
    if _is_noisy_web_source(source):
        score -= 1.0
    return score


def _best_web_sources(web_sources, used_web_sources, limit=MAX_WEB_SOURCES_TO_SHOW):
    if not web_sources:
        return []
    prioritized = []
    seen = set()
    for source in [*used_web_sources, *web_sources]:
        if source.label in seen:
            continue
        seen.add(source.label)
        prioritized.append(source)
        if len(prioritized) >= limit:
            break
    return prioritized


def _relabel_local_sources(sources):
    values = list(sources)
    for index, source in enumerate(values, start=1):
        source.label = f"S{index}"
    return values


def _relabel_web_sources(sources):
    values = list(sources)
    for index, source in enumerate(values, start=1):
        source.label = f"W{index}"
    return values


def _web_queries(question, query, search_plan: SearchPlan | None = None):
    base = _clean_web_query(query or question)
    candidates = _planner_query_candidates(search_plan, base)
    if not candidates and base:
        candidates.append(base)
    candidates.extend(_normalized_query_candidates(question, query, base))
    candidates.extend(_contextual_query_candidates(search_plan, base))
    candidates.extend(_news_query_candidates(question, base))
    if not candidates:
        candidates.extend(_office_start_query_candidates(question, base))
        candidates.extend(_age_at_office_query_candidates(question, base))
        candidates.extend(_current_public_query_candidates(question, base))
        candidates.extend(_public_knowledge_query_candidates(question, base))
    if not candidates:
        candidates = [base, f"{base} official government", f"{base} latest", f"{base} reliable source"]
    elif base and not _looks_like_public_knowledge_query(question) and not _looks_like_news_query(question):
        candidates.extend([base, f"{base} official government"])
    if any(term in base.lower() for term in ("paper", "scientific", "study", "research")):
        candidates.extend([f"{base} paper", f"{base} arxiv OR doi", f"{base} review"])
    return _unique_nonempty(candidates)


def _web_rerank_query(query: str, web_queries: Sequence[str]) -> str:
    normalized = normalize_query(query)
    parts = [
        query,
        normalized.canonical,
        " ".join(normalized.key_terms),
        " ".join(normalized.entities),
        *list(web_queries)[:3],
    ]
    return " ".join(part for part in parts if part).strip()


def _normalized_query_candidates(question: str, query: str, base: str) -> list[str]:
    values: list[str] = []
    for candidate in [*query_variants(question), *query_variants(query), base]:
        cleaned = _clean_web_query(candidate)
        if cleaned:
            values.append(cleaned)
    normalized = normalize_query(query or question)
    term_text = " ".join(normalized.key_terms)
    entity_text = " ".join(normalized.entities)
    if term_text and entity_text:
        values.append(f"{entity_text} {term_text}")
    if normalized.intent in {"what", "statement"} and term_text:
        values.append(f"{term_text} facts")
    return _unique_nonempty(values)


def _contextual_query_candidates(search_plan: SearchPlan | None, base: str) -> list[str]:
    if search_plan is None:
        return []
    candidates = []
    entity = search_plan.entity
    topic = search_plan.topic
    country = search_plan.country
    role = search_plan.role
    context = " ".join(value for value in (entity, role, country, topic) if value)
    if context and context.lower() not in base.lower():
        candidates.append(f"{context} {base}")
    if entity and role:
        candidates.append(f"{entity} {role}")
    if role and country:
        candidates.append(f"{role} of {_country_phrase(country)} official")
    if topic and entity:
        candidates.append(f"{entity} {topic}")
    return candidates


def _aggressive_web_queries(question: str, web_queries: Sequence[str]) -> list[str]:
    base_queries = _dedupe_web_queries([question, *web_queries, *query_variants(question)])
    expanded: list[str] = []
    for query in base_queries:
        domain = classify_query_domain(query)
        normalized = normalize_query(query)
        term_text = " ".join(normalized.key_terms)
        entity_text = " ".join(normalized.entities)
        expanded.append(query)
        if domain == "government":
            expanded.extend([f"{query} official", f"{query} government biography", f"{query} incumbent"])
        elif domain == "news":
            expanded.extend([f"{query} Reuters", f"{query} AP News", f"{query} BBC"])
        elif domain == "science":
            expanded.extend([f"{query} DOI", f"{query} arxiv", f"{query} university"])
        elif domain == "person":
            expanded.extend([f"{query} profile", f"{query} biography", f"{query} official"])
        else:
            expanded.extend([f"{query} reliable source", f"{query} official", f"{query} reference"])
        if term_text:
            expanded.append(f"{term_text} reliable source")
        if term_text and entity_text:
            expanded.append(f"{entity_text} {term_text} reference")
    return _dedupe_web_queries(expanded)


def _dedupe_web_queries(queries: Sequence[str]) -> list[str]:
    unique = []
    seen = set()
    for query in queries:
        cleaned = re.sub(r"\s+", " ", (query or "").strip())
        if not cleaned:
            continue
        key = _web_query_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _web_query_key(query: str) -> str:
    folded = _fold_text(query)
    folded = re.sub(r"\b(?:please|kindly)\b", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip(" :;-?!.")
    return folded


def _planner_query_candidates(search_plan: SearchPlan | None, base: str) -> list[str]:
    if search_plan is None:
        return []
    intent = search_plan.intent
    if intent == "person" and search_plan.entity:
        entity = search_plan.entity
        return [
            f"{entity} university profile ORCID",
            f"{entity} Google Scholar",
            f"{entity} GitHub",
            f"{entity} LinkedIn",
            f"{entity} ResearchGate",
            base,
        ]
    if intent == "government" and search_plan.country and search_plan.role:
        country = _country_phrase(search_plan.country)
        role = search_plan.role
        corrected_head_of_state = _corrected_head_of_state_query_candidates(
            search_plan.country,
            role,
            base,
        )
        if corrected_head_of_state:
            return corrected_head_of_state
        if search_plan.country == "United Kingdom" and role == "Prime Minister":
            return [
                "GOV.UK Prime Minister",
                "current prime minister of the United Kingdom official government",
                base,
            ]
        if search_plan.country == "United States" and role == "President":
            return [
                "current president of the United States official government",
                "White House President of the United States",
                base,
            ]
        return [
            f"{role} of {country} official government",
            f"current {role} {country} government",
            f"{country} {role} official biography",
            base,
        ]
    if intent == "scientific_definition":
        topic = search_plan.topic or base
        return [
            f"{topic} definition university",
            f"{topic} arxiv paper",
            f"{topic} tutorial lecture notes",
            base,
        ]
    if intent == "public_knowledge":
        topic = search_plan.topic or base
        return _unique_nonempty(
            [
                f"{topic} reliable reference",
                f"{topic} around the world",
                f"{topic} history",
                base,
            ]
        )
    if intent == "news":
        subject = search_plan.entity or search_plan.topic or base
        return [f"{source} {subject}" for source in search_plan.preferred_sources[:6]]
    return []


def _age_at_office_query_candidates(question: str, base: str) -> list[str]:
    if not _looks_like_age_at_office_query(question):
        return []
    candidates = [base]
    person = _age_query_person(question)
    role_country = _age_query_role_country(question)
    if person and role_country:
        candidates.extend(
            [
                f"{person} born {role_country} took office",
                f"{person} date of birth {role_country} president since",
                f"{person} biography {role_country}",
            ]
        )
    elif person:
        candidates.extend([f"{person} date of birth took office", f"{person} biography"])
    return candidates


def _office_start_query_candidates(question: str, base: str) -> list[str]:
    if not _looks_like_office_start_query(question):
        return []
    person = _office_start_query_person(question)
    role_country = _office_start_query_role_country(question)
    candidates = [base]
    if person and role_country:
        candidates.extend(
            [
                f"{person} {role_country} since",
                f"{person} became {role_country} date",
                f"{person} took office {role_country}",
            ]
        )
    return candidates


def _news_query_candidates(question: str, base: str) -> list[str]:
    if not _looks_like_news_query(question):
        return []
    subject = _clean_news_query_subject(base or question)
    sources = requested_news_sources(question)
    if not sources and any(marker in (question or "").lower() for marker in ("news", "resign", "resignation")):
        sources = list(DEFAULT_NEWS_OUTLETS)
    candidates = []
    for source in sources[:6]:
        candidates.append(f"{source} {subject}")
        if "resign" in subject.lower() and "why" not in subject.lower():
            candidates.append(f"{source} why {subject}")
    if not candidates:
        candidates.append(subject)
    return candidates[:8]


def _clean_news_query_subject(text: str) -> str:
    cleaned = re.sub(
        r"^\s*(?:search|check|look up)\s+(?:reuters|ap news|ap|bbc news|bbc|sky news|sky|financial times|ft|the guardian|guardian|news(?: channels?)?)\s*(?:for|about)?\s*",
        " ",
        text or "",
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(?:and\s+)?tell\s+me\s+", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" :;-?!.") or text.strip()


def _public_knowledge_query_candidates(question, base):
    candidates = []
    topics = _public_knowledge_topics(question)
    public_topic = _public_topic_from_text(question)
    if "toxicology" in topics:
        candidates.append("toxicology definition NIH")
    if "reach" in topics:
        candidates.append("REACH regulation ECHA official")
    if len(topics) == 1 and "toxicology" in topics:
        candidates.append("toxicology definition scientific discipline")
    if len(topics) == 1 and "reach" in topics:
        candidates.append("European Commission REACH Regulation chemicals")
    if public_topic:
        candidates.extend(
            [
                f"{public_topic} reliable reference",
                f"{public_topic} around the world",
                f"{public_topic} history",
            ]
        )
        lower = (question or "").lower()
        if "erupt" in lower:
            candidates.append(f"{public_topic} eruption history")
        years = re.search(r"\blast\s+(\d+)\s+years\b", lower)
        if years:
            candidates.append(f"{public_topic} erupted last {years.group(1)} years")
    if not candidates and _looks_like_public_knowledge_query(question):
        candidates.append(f"{base} official source")
    if candidates and base:
        candidates.append(base)
    return candidates


def _current_public_role_context(question):
    lower = (question or "").lower()
    if "prime minister" in lower and "luxembourg" in lower:
        return {
            "kind": "role",
            "role": "prime minister",
            "subject": "The current prime minister of Luxembourg",
        }
    if "prime minister" in lower and UK_PATTERN.search(lower):
        return {
            "kind": "role",
            "role": "prime minister",
            "subject": "The current prime minister of the United Kingdom",
        }
    if "secretary of state" in lower and not re.search(r"\b(?:state column|state variable|state =)\b", lower):
        return {
            "kind": "role",
            "role": "secretary of state",
            "subject": "The current U.S. Secretary of State",
        }
    if "king" in lower and "netherlands" in lower:
        return {
            "kind": "role",
            "role": "king",
            "subject": "The current king of the Netherlands",
        }
    if "king" in lower and USA_PATTERN.search(lower):
        return {
            "kind": "no-monarch",
            "role": "president",
            "subject": "The current president of the United States",
        }
    if "president" in lower and USA_PATTERN.search(lower):
        return {
            "kind": "role",
            "role": "president",
            "subject": "The current president of the United States",
        }
    country = _country_from_text(question)
    if country and "president" in lower:
        return {
            "kind": "role",
            "role": "president",
            "subject": f"The current president of {_country_phrase(country)}",
            "country": country,
        }
    if country and "prime minister" in lower:
        return {
            "kind": "role",
            "role": "prime minister",
            "subject": f"The current prime minister of {_country_phrase(country)}",
            "country": country,
        }
    return None


def _current_public_query_candidates(question, base):
    context = _current_public_role_context(question)
    if not context:
        return []
    role = context["role"]
    lower = (question or "").lower()
    if context["kind"] == "no-monarch":
        return [
            "current president of the United States",
            "White House President of the United States",
            "current president of the United States official government",
        ]
    if role == "president" and USA_PATTERN.search(lower):
        return [
            "current president of the United States official government",
            "White House President of the United States",
        ]
    if role == "prime minister" and UK_PATTERN.search(lower):
        return [
            "GOV.UK Prime Minister",
            "current prime minister of the United Kingdom official government",
        ]
    if role == "secretary of state":
        return [
            "current U.S. Secretary of State official state.gov",
            "U.S. Department of State Secretary of State",
            "Secretary of State United States official government",
        ]
    if role == "prime minister" and "luxembourg" in lower:
        return [
            f"{base} official government",
            "Luxembourg government prime minister",
        ]
    if role == "king" and "netherlands" in lower:
        return [
            "King of the Netherlands Royal House",
            "current king of the Netherlands official royal house",
        ]
    country = context.get("country")
    if country:
        country_text = _country_phrase(country)
        return [
            f"current {role} of {country_text} official government",
            f"{role} of {country_text} incumbent",
            f"{country_text} {role} biography",
        ]
    return []


def _corrected_head_of_state_query_candidates(
    country: str,
    role: str,
    base: str,
) -> list[str]:
    corrected_role = HEAD_OF_STATE_ROLE_BY_COUNTRY.get(country, "")
    if not corrected_role or role not in MONARCH_ROLE_LABELS:
        return []
    country_phrase = _country_phrase(country)
    role_phrase = corrected_role.lower()
    return [
        f"current {role_phrase} of {country_phrase}",
        f"{corrected_role} of {country_phrase} official government",
        f"{country_phrase} head of state official",
        base,
    ]


def _current_public_fact_evidence(question, web_sources):
    context = _current_public_role_context(question)
    if not context:
        return []

    role = context["role"]
    candidates = []
    for source in web_sources:
        if _is_noisy_web_source(source):
            continue
        if role == "secretary of state" and not _is_us_secretary_of_state_source(source):
            continue
        candidates.append((source, _extract_role_candidate(source, role)))

    official = [
        (source, name)
        for source, name in candidates
        if _is_authoritative_web_source(source) and name
    ]
    if official:
        return sorted(
            official,
            key=lambda item: _current_fact_source_score(item[0], item[1], role),
            reverse=True,
        )
    if context.get("country"):
        single_country_candidates = [
            (source, name)
            for source, name in candidates
            if name
        ]
        if single_country_candidates:
            return sorted(
                single_country_candidates,
                key=lambda item: _current_fact_source_score(item[0], item[1], role),
                reverse=True,
            )

    by_candidate = {}
    for source, name in candidates:
        key = _normalized_candidate_name(name)
        if not key:
            continue
        by_candidate.setdefault(key, []).append((source, name))

    corroborated = []
    for values in by_candidate.values():
        domains = {_web_source_domain(source) for source, _name in values if _web_source_domain(source)}
        if len(domains) >= 2:
            corroborated.extend(values)
    return sorted(
        corroborated,
        key=lambda item: _current_fact_source_score(item[0], item[1], role),
        reverse=True,
    )


def _current_fact_source_score(source, candidate, role):
    haystack = f"{source.title} {source.content}".lower()
    title = (source.title or "").lower()
    score = 0.0
    if _is_authoritative_web_source(source):
        score += 5.0
    if candidate:
        score += 1.0
    if title.startswith(role):
        score += 2.0
    if any(marker in haystack for marker in ("current role holder", "incumbent", "has been king", "is the prime minister", "is prime minister", "is the president", "is president", "is the secretary of state", "serves as secretary of state", "sworn in as secretary of state")):
        score += 1.0
    if any(marker in haystack for marker in ("was prime minister", "former prime minister", "between 25", "between 20", "former president")):
        score -= 3.0
    return score


def _is_us_secretary_of_state_source(source) -> bool:
    haystack = f"{source.title} {source.url} {source.content}".lower()
    domain = _web_source_domain(source)
    return (
        "state.gov" in domain
        or "usembassy.gov" in domain
        or "state.gov" in haystack
        or "usembassy.gov" in haystack
        or "u.s. department of state" in haystack
        or "united states department of state" in haystack
    )


def _extract_role_candidate(source, role):
    patterns = {
        "prime minister": (
            r"\bCurrent role holder\s+(?P<name>[^.]+?)\.",
            r"\bIncumbent\s+(?P<name>[^.]+?)\s+since\b",
            rf"\b(?P<name>{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{1,5}})\s+became\s+(?:the\s+)?Prime Minister\b",
            rf"\b(?P<name>{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{1,5}})\s+(?:is|serves\s+as|has\s+served\s+as)\s+(?:the\s+)?(?:current\s+)?Prime Minister(?:\s+of\s+{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{0,5}})?\b",
        ),
        "president": (
            rf"\bPresident\s+(?P<name>{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{0,5}})\b",
            rf"\b(?P<name>{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{1,5}})\s*,\s+(?:the\s+)?(?:current\s+)?President\b",
            rf"\b(?P<name>{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{1,5}})\s+(?:is|has\s+been|serves\s+as|has\s+served\s+as)\s+(?:the\s+)?(?:current\s+)?President\b",
            rf"\b(?P<name>{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{1,5}})\s+is\s+the\s+\d+(?:st|nd|rd|th)\s+and\s+\d+(?:st|nd|rd|th)\s+President\b",
        ),
        "secretary of state": (
            r"\bSecretary\s+of\s+State\s+(?P<name>[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,5})\b",
            r"\b(?P<name>[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,5})\s*:\s+[^.]{0,120}\b[Ss]ecretary\s+of\s+[Ss]tate\b",
            r"\b(?P<name>[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,5})\s+(?:was\s+sworn\s+in\s+as|is)\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)\s+)?[Ss]ecretary\s+of\s+[Ss]tate\b",
            r"\b(?P<name>[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,5})\s+serves\s+as\s+(?:the\s+)?(?:U\.S\.\s+)?[Ss]ecretary\s+of\s+[Ss]tate\b",
        ),
        "king": (
            r"\bKing\s+(?P<name>[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,4})\b",
            r"\b(?P<name>[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,4})\s+has\s+been\s+King\b",
            r"\b(?P<name>[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,4})\s+is\s+(?:the\s+)?King\b",
        ),
    }
    for text in (source.content or "", source.title or ""):
        for pattern in patterns.get(role, ()):
            match = re.search(pattern, text)
            if not match:
                continue
            candidate = _clean_person_name(match.group("name"))
            if _looks_like_person_name(candidate):
                return candidate

    title_head = re.split(r"\s+\|\s+|\s+-\s+", source.title or "", maxsplit=1)[0]
    candidate = _clean_person_name(title_head)
    if re.fullmatch(r"[A-Z][A-Z\-']+\s+[A-Z][a-z\-']+", candidate):
        surname, given = candidate.split(" ", 1)
        candidate = f"{given} {surname.title()}"
    if _looks_like_person_name(candidate):
        return candidate
    return ""


def _clean_person_name(value):
    text = (value or "").strip()
    text = re.sub(r"^(?:the\s+)?(?:(?:rt\s+hon|hon|sir|dr|mr|mrs|ms)\s+)+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:president|prime minister|secretary of state|king|queen)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:KCB|KC|MP|PC|OBE|CBE|Jr\.?|Sr\.?)\b", "", text)
    text = _trim_person_name_at_boundary(text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    text = _normalize_person_name_case(text)
    return text


def _looks_like_person_name(value):
    text = (value or "").strip()
    if not text or not re.search(r"[A-Za-z]", text):
        return False
    if re.match(r"^(?:of|the)\b", text, flags=re.IGNORECASE):
        return False
    words = [word for word in re.split(r"\s+", text) if word]
    if not 1 <= len(words) <= 5:
        return False
    if _person_name_has_boundary_term(text):
        return False
    return not re.search(
        r"\b(?:department|duties|government|house|minister|official|prime|secretary|state|united states)\b",
        text,
        re.IGNORECASE,
    )


def _trim_person_name_at_boundary(value: str) -> str:
    text = re.split(
        r"\s+(?:has\s+been|has\s+served|is|was|became|since|between|serves\s+as|served\s+as)\b",
        value or "",
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    kept = []
    for index, raw_token in enumerate(re.split(r"\s+", text.strip())):
        token = raw_token.strip(" ,.;:()[]{}")
        folded = token.lower().strip("'’")
        if not folded:
            continue
        if index > 0 and folded in PERSON_NAME_ACTION_BOUNDARIES:
            break
        if index >= 2 and folded in PERSON_NAME_ORG_BOUNDARIES:
            break
        if index >= 2 and _looks_like_person_name_org_acronym(token):
            break
        kept.append(raw_token)
    return " ".join(kept)


def _looks_like_person_name_org_acronym(token: str) -> bool:
    cleaned = token.strip(" ,.;:()[]{}")
    return (
        bool(re.fullmatch(r"[A-Z][A-Z0-9&./-]{1,}", cleaned))
        and cleaned.lower() not in PERSON_NAME_SUFFIXES
    )


def _normalize_person_name_case(value: str) -> str:
    words = []
    for word in re.split(r"\s+", value or ""):
        bare = word.strip(" ,.;:()[]{}")
        if len(bare) > 1 and bare.isupper() and bare.lower() not in PERSON_NAME_SUFFIXES:
            words.append(word.replace(bare, bare.title()))
        else:
            words.append(word)
    return " ".join(words)


def _person_name_has_boundary_term(value: str) -> bool:
    words = [word.strip(" ,.;:()[]{}").lower().strip("'’") for word in re.split(r"\s+", value or "")]
    return any(
        word in PERSON_NAME_ACTION_BOUNDARIES or word in PERSON_NAME_ORG_BOUNDARIES
        for word in words
    )


def _normalized_candidate_name(value):
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _candidate_names_conflict(names: Sequence[str]) -> bool:
    normalized = [_normalized_candidate_name(name) for name in names if _normalized_candidate_name(name)]
    unique = list(dict.fromkeys(normalized))
    if len(unique) <= 1:
        return False
    for index, left in enumerate(unique):
        for right in unique[index + 1:]:
            if not _candidate_names_compatible(left, right):
                return True
    return False


def _candidate_names_compatible(left: str, right: str) -> bool:
    left_tokens = _candidate_name_tokens(left)
    right_tokens = _candidate_name_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return True
    left_edges = (min(left_tokens), max(left_tokens)) if len(left_tokens) >= 2 else ()
    right_edges = (min(right_tokens), max(right_tokens)) if len(right_tokens) >= 2 else ()
    if left_edges and left_edges == right_edges:
        return True
    return bool(left_tokens & right_tokens) and _last_name_token(left) == _last_name_token(right)


def _candidate_name_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zÀ-ÖØ-öø-ÿ'-]+", _fold_text(value))
        if len(token) > 1 and token not in {"the", "rt", "hon", "sir", "dr", "mr", "mrs", "ms"}
    }


def _last_name_token(value: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-zÀ-ÖØ-öø-ÿ'-]+", _fold_text(value))
        if len(token) > 1
    ]
    return tokens[-1] if tokens else ""


def _unverified_current_web_answer(web_sources):
    labels = [f"[{source.label}]" for source in web_sources[:2] if getattr(source, "label", "")]
    suffix = f" The available result {' and '.join(labels)} was not authoritative enough to confirm the answer." if labels else ""
    return f"I could not verify current information from reliable web sources.{suffix}"


def _is_unreliable_current_source(source):
    haystack = f"{source.title} {source.url} {source.content}".lower()
    if _is_noisy_web_source(source):
        return True
    return any(marker in haystack for marker in ("blog", "personal-blog", "opinion", "editorial"))


def _is_resignation_news_query(question: str) -> bool:
    lower = (question or "").lower()
    return any(marker in lower for marker in ("resign", "resigned", "resignation", "stepped down"))


def _web_source_domain(source):
    return urlparse(getattr(source, "url", "") or "").netloc.lower()


def _clean_web_query(text):
    cleaned = (text or "").strip()
    patterns = (
        r"^\s*(?:please\s+)?(?:search|use)\s+(?:the\s+)?(?:web|internet|online)\s*(?:for|about)?\s*",
        r"^\s*(?:please\s+)?look\s+up\s*(?:online|on\s+the\s+web|on\s+the\s+internet)?\s*(?:for|about)?\s*",
        r"^\s*(?:web\s+search|online\s+search)\s*(?:for|about)?\s*",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" :;-?!.") or text.strip()


def _expand_local_file_query(query):
    cleaned = _clean_local_file_query(query)
    terms = [cleaned] if cleaned else [query.strip()]
    lower = query.lower()
    for marker, expansions in LOCAL_FILE_EXPANSIONS.items():
        if marker in lower:
            terms.extend(expansions)
    terms.extend(("local files", "indexed documents", "uploaded documents"))
    return " ".join(_unique_nonempty(terms))


def _local_file_search_queries(query):
    cleaned = _clean_local_file_query(query)
    lower = query.lower()
    queries = []
    specific_terms: list[str] = []
    for marker, expansions in LOCAL_FILE_EXPANSIONS.items():
        if marker not in lower:
            continue
        specific_terms.extend(_specific_expansion_terms(expansions))
    specific_terms = _unique_nonempty(specific_terms)
    if specific_terms:
        strongest = specific_terms[0]
        if cleaned and strongest.lower() not in cleaned.lower():
            queries.append(f"{cleaned} {strongest}")
        else:
            queries.append(strongest)
    return _unique_nonempty(queries)


def _local_file_ranking_query(query):
    cleaned = _clean_local_file_query(query)
    lower = query.lower()
    terms = [cleaned] if cleaned else []
    for marker, expansions in LOCAL_FILE_EXPANSIONS.items():
        if marker in lower:
            terms.extend(_specific_expansion_terms(expansions))
    return " ".join(_unique_nonempty(terms)) or query


def _specific_expansion_terms(expansions):
    return sorted(
        (term for term in expansions if len(term) >= 5),
        key=lambda term: (len(term), term),
        reverse=True,
    )[:4]


def _clean_local_file_query(query):
    cleaned = (query or "").strip()
    patterns = (
        r"\b(?:in|inside|from)\s+(?:the\s+)?(?:indexed\s+)?local\s+files?\b",
        r"\b(?:in|inside|from)\s+(?:the\s+)?(?:indexed\s+)?documents?\b",
        r"\b(?:uploaded|indexed)\s+(?:files?|documents?)\b",
        r"^\s*(?:is|are)\s+there\s+",
        r"^\s*(?:which|what)\s+(?:document|file)\s+(?:contains?|has|includes?)\s+",
        r"^\s*(?:do|does)\s+(?:i|you|we)\s+have\s+",
        r"^\s*(?:can\s+you\s+)?find\s+",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" :;-?!.")


def _unique_nonempty(values):
    unique, seen = [], set()
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


def _fallback_answer_from_web_results(*, web_sources, previous_answer):
    sources = list(web_sources)[:MAX_WEB_SOURCES_TO_SHOW]
    if previous_answer and not _looks_like_non_answer(previous_answer):
        labels = ", ".join(f"[{source.label}]" for source in sources)
        return f"{previous_answer.strip()}\n\nBest web sources checked: {labels}."
    confidence = "High" if any(_is_authoritative_web_source(source) for source in sources) else "Medium"
    best = sources[0] if sources else None
    answer = _compact_source_text(_useful_source_text(best) if best else "", limit=320)
    if answer and best:
        lines = [f"{answer} [{best.label}]", "", f"Confidence: {confidence}", "", "Sources:"]
    else:
        lines = [
            "I found web sources, but they did not contain enough extractable text for a fuller answer.",
            "",
            f"Confidence: {confidence}",
        ]
        lines.extend(("", "Sources:"))
    for source in sources[:3]:
        preview = _compact_source_text(source.content or source.title, limit=260)
        lines.append(f"- [{source.label}] {source.title}: {preview}" if preview else f"- [{source.label}] {source.title}")
    if len(sources) > 3:
        lines.append("Additional sources checked: " + ", ".join(f"[{source.label}]" for source in sources[3:]) + ".")
    return "\n".join(lines)


def _helpful_failure_answer(
    question: str,
    *,
    reason: str,
    provider_label: str,
    web_error: str = "",
    web_enabled: bool = True,
    web_sources: Sequence[WebSource] | None = None,
) -> str:
    normalized = normalize_query(question)
    variants = query_variants(question)[:4]
    lines: list[str]
    if reason == "current_web_failed":
        lines = [
            "I could not verify current information from local files or web sources.",
            f"{provider_label} web search could not complete, and AI knowledge is not reliable enough for current facts.",
        ]
    elif reason == "current_no_sources":
        lines = [
            "I could not verify current information from local files or returned web sources.",
            "AI knowledge is not reliable enough for current facts.",
        ]
    elif reason == "web_failed":
        lines = [
            "I could not answer from local files or reliable model knowledge.",
            f"{provider_label} web search could not complete, so no web sources were available.",
        ]
    elif not web_enabled:
        lines = [
            "I could not answer from local files, and web search is disabled.",
        ]
    else:
        lines = [
            "I could not answer from local files, reliable model knowledge, or returned web sources.",
        ]

    lines.extend(["", "What you can try:"])
    if not web_enabled:
        lines.append("- Enable web search or add a web search API key in the sidebar.")
    lines.append("- Rephrase with the main entity and the attribute you need.")
    if normalized.entities:
        lines.append(f"- Include the entity explicitly: {'; '.join(normalized.entities[:3])}.")
    if normalized.key_terms:
        lines.append(f"- Core terms I understood: {', '.join(normalized.key_terms[:6])}.")
    if web_error:
        lines.append(f"- Provider detail: {provider_label} search could not complete: {web_error}.")
    if web_sources:
        lines.append(f"- I checked {len(web_sources)} web source(s), but none clearly answered it.")
    if variants:
        lines.extend(["", "Search forms to try:"])
        lines.extend(f"- {variant}" for variant in variants[:3])
    return "\n".join(lines)


def _verified_payload_confidence(payload, ranked_evidence):
    if not ranked_evidence:
        return "Low"
    badge = str(getattr(payload, "evidence_badge", "") or "").lower()
    citations = list(getattr(payload, "citations", []) or [])
    best_score = max(float(getattr(item, "score", 0.0) or 0.0) for item in ranked_evidence)
    authoritative = any(
        str(getattr(item, "authority", "") or "").lower().endswith("official")
        for item in ranked_evidence
    )
    if "ai knowledge only" in badge:
        return "Low"
    if authoritative and best_score >= 0.68:
        return "High"
    if len(citations) >= 2 and best_score >= 0.62:
        return "High"
    if best_score >= 0.52 or citations:
        return "Medium"
    return "Low"


def _fallback_from_ranked_evidence(ranked_evidence, *, question):
    if not ranked_evidence:
        return "I could not find enough relevant evidence to answer.\n\nConfidence: Low"

    usable = [item for item in ranked_evidence if getattr(item, "citation_label", "") != "AI"]
    if not usable:
        usable = list(ranked_evidence)

    confidence = _extractive_confidence(usable)
    terms = query_terms(question)
    answer_lines = _extractive_answer_lines(question, usable, terms)
    lines = [*answer_lines, "", f"Confidence: {confidence}", "", "Sources:"]
    for item in usable[:4]:
        title = _compact_source_text(item.title, limit=120)
        url = f" — {item.url}" if getattr(item, "url", "") else ""
        lines.append(f"- {item.citation()} {title}{url}")
    return "\n".join(lines)


def _scientific_answer_from_ranked_evidence(ranked_evidence, *, question):
    usable = [item for item in ranked_evidence if getattr(item, "citation_label", "") != "AI"]
    if not usable:
        return _fallback_from_ranked_evidence(ranked_evidence, question=question)

    terms = query_terms(question)
    subject = _scientific_subject(question)
    confidence = _extractive_confidence(usable)
    definition = _best_scientific_fact(
        usable,
        terms,
        ("combines", "method", "algorithm", "uses", "sampling", "monte carlo", "hamiltonian"),
    )
    benefits = _best_scientific_fact(
        usable,
        terms,
        ("improve", "efficient", "exploration", "mixing", "multimodal", "robust"),
        exclude={definition},
    )
    applications = _best_scientific_fact(
        usable,
        terms,
        ("applied", "application", "model", "bayesian", "hbv", "rainfall", "runoff"),
        exclude={definition, benefits},
    )

    if definition:
        lines = [f"{subject}: {definition}"]
    else:
        lines = [f"{subject}: {_fact_from_item(usable[0], terms)}"]

    key_points = [item for item in (benefits, applications) if item]
    if key_points:
        lines.extend(("", "Key points:"))
        for point in key_points[:3]:
            lines.append(f"- {point}")

    lines.extend(("", f"Confidence: {confidence}", "", "Sources:"))
    for item in usable[:4]:
        title = _compact_source_text(item.title, limit=120)
        url = f" — {item.url}" if getattr(item, "url", "") else ""
        lines.append(f"- {item.citation()} {title}{url}")
    return "\n".join(lines)


def _public_knowledge_answer_from_web(question: str, web_sources: Sequence[WebSource]) -> str:
    topics = _public_knowledge_topics(question)
    if not topics:
        return _generic_public_knowledge_answer_from_web(question, web_sources)
    source_by_topic = {
        topic: _best_public_source_for_topic(topic, web_sources)
        for topic in topics
    }
    answered = {
        topic: source
        for topic, source in source_by_topic.items()
        if source is not None
    }
    if not answered:
        return _fallback_answer_from_web_results(web_sources=web_sources, previous_answer="")

    confidence = _public_knowledge_confidence(topics, answered)
    lines = []
    for topic in topics:
        source = answered.get(topic)
        if source is None:
            lines.append(f"{_public_topic_label(topic)}: I did not find a strong source for this part.")
            continue
        lines.append(_public_topic_answer(question, topic, source))
    lines.extend(("", f"Confidence: {confidence}"))
    return "\n\n".join(lines)


def _generic_public_knowledge_answer_from_web(question: str, web_sources: Sequence[WebSource]) -> str:
    usable = [source for source in web_sources if not _is_noisy_web_source(source)]
    if not usable:
        return _fallback_answer_from_web_results(web_sources=web_sources, previous_answer="")
    topic = _public_topic_from_text(question) or _strip_question_prefix(question)
    terms = tuple(query_terms(f"{question} {topic}")) or tuple(topic.lower().split())
    best = usable[0]
    sentence = _best_public_content_sentence(best, terms)
    confidence = "High" if any(_is_authoritative_web_source(source) for source in usable[:3]) else "Medium"
    if _looks_like_list_request(question):
        lead = _list_style_answer(topic, sentence, best)
    else:
        lead = f"{sentence} [{best.label}]"
    lines = [lead, "", f"Confidence: {confidence}", "", "Sources:"]
    for source in usable[:4]:
        title = _compact_source_text(source.title, limit=120)
        url = f" — {source.url}" if getattr(source, "url", "") else ""
        lines.append(f"- [{source.label}] {title}{url}")
    return "\n".join(lines)


def _looks_like_list_request(question: str) -> bool:
    normalized = normalize_intent_text(question)
    return normalized.startswith(("identify ", "list ", "name ", "show ", "which "))


def _list_style_answer(topic: str, sentence: str, source: WebSource) -> str:
    cleaned = _compact_source_text(sentence, limit=360)
    if re.search(r"\b(?:are|examples?|include|includes|including)\b", cleaned, flags=re.IGNORECASE):
        return f"{cleaned} [{source.label}]"
    topic_label = topic or "examples"
    return f"Relevant {topic_label} include examples described by {source.title}: {cleaned} [{source.label}]"


def _best_public_content_sentence(source: WebSource, markers: Sequence[str]) -> str:
    content = getattr(source, "content", "") or ""
    if content:
        scored = []
        for index, sentence in enumerate(_candidate_sentences(content)):
            cleaned = _clean_public_sentence(sentence)
            lower = cleaned.lower()
            if _looks_like_boilerplate_text(cleaned):
                continue
            overlap = sum(1 for marker in markers if marker in lower)
            list_signal = 2 if re.search(r"\b(?:examples?|include|includes|including)\b", lower) else 0
            if overlap or list_signal:
                scored.append((overlap * 2 + list_signal - index * 0.03, cleaned))
        if scored:
            return _compact_source_text(max(scored, key=lambda item: item[0])[1], limit=360)
    return _best_public_sentence(source, markers)


def _office_start_answer_from_web(question: str, web_sources: Sequence[WebSource]) -> str:
    person = _office_start_query_person(question) or _office_start_role_candidate(question, web_sources)
    role_country = _office_start_query_role_country(question) or _office_start_context_role_country(question)
    office = _find_office_date(web_sources, role_country)
    if office is None:
        return ""
    office_date, office_source = office
    subject = person or "They"
    office_text = role_country or "that office"
    return (
        f"{subject} became {office_text} on {_format_date(office_date)} "
        f"[{office_source.label}].\n\nConfidence: High"
    )


def _office_start_role_candidate(question: str, web_sources: Sequence[WebSource]) -> str:
    context = _current_public_role_context(question)
    if not context:
        return ""
    evidence = _current_public_fact_evidence(question, web_sources)
    if evidence:
        _source, candidate = evidence[0]
        if candidate:
            return candidate
    for source in web_sources:
        candidate = _extract_role_candidate(source, context["role"])
        if candidate:
            return candidate
    return ""


def _office_start_context_role_country(question: str) -> str:
    context = _current_public_role_context(question)
    if not context or not context.get("subject"):
        return ""
    subject = re.sub(r"^the\s+current\s+", "", context["subject"], flags=re.IGNORECASE)
    return subject[:1].upper() + subject[1:]


def _age_at_office_answer_from_web(question: str, web_sources: Sequence[WebSource]) -> str:
    person = _age_query_person(question)
    role_country = _age_query_role_country(question)
    birth = _find_birth_date(web_sources, person)
    office = _find_office_date(web_sources, role_country)
    if birth is None or office is None:
        return ""

    birth_date, birth_source = birth
    office_date, office_source = office
    age = _age_on_date(birth_date, office_date)
    labels = [f"[{birth_source.label}]"]
    if office_source.label != birth_source.label:
        labels.append(f"[{office_source.label}]")
    source_text = " ".join(labels)
    subject = person or "They"
    office_text = role_country or "that office"
    return (
        f"{subject} was {age} years old when he became {office_text} "
        f"on {_format_date(office_date)} {source_text}.\n\nConfidence: High"
    )


def _find_birth_date(web_sources: Sequence[WebSource], person: str) -> tuple[date, WebSource] | None:
    for source in web_sources:
        text = _source_haystack(source)
        if person and _fold_text(person) not in _fold_text(text):
            continue
        value = _extract_birth_date(text)
        if value:
            return value, source
    for source in web_sources:
        value = _extract_birth_date(_source_haystack(source))
        if value:
            return value, source
    return None


def _find_office_date(web_sources: Sequence[WebSource], role_country: str) -> tuple[date, WebSource] | None:
    for source in web_sources:
        text = _source_haystack(source)
        if role_country and not _role_country_matches(text, role_country):
            continue
        value = _extract_office_date(text)
        if value:
            return value, source
    for source in web_sources:
        value = _extract_office_date(_source_haystack(source))
        if value:
            return value, source
    return None


def _extract_birth_date(text: str) -> date | None:
    patterns = (
        r"\bborn\s+(?:on\s+)?(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
        r"\bdate\s+of\s+birth\s*:?\s*(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return _parse_human_date(match.group("date"))
    return None


def _extract_office_date(text: str) -> date | None:
    patterns = (
        r"\b(?:came|come)\s+(?:into|to)\s+power\s+(?:on\s+)?(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
        r"\b(?:became|assumed\s+office|took\s+office|sworn\s+in)\s+(?:on\s+)?(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
        r"\b(?:became|was\s+sworn\s+in\s+as)\s+(?:the\s+)?(?:president|prime\s+minister|minister)[^.\n]{0,80}?\s+(?:on\s+)?(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
        r"\b(?:became|was\s+appointed|was\s+named|was\s+sworn\s+in\s+as)\s+(?:the\s+)?[^.\n]{3,120}?\s+(?:on\s+)?(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
        r"\b(?:in\s+power|held\s+power)\b[^.\n]{0,80}\b(?:since|from)\s+(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
        r"\b(?:president|prime\s+minister|minister)\b[^.\n]{0,120}\b(?:since|from)\s+(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
        r"\b(?:since|from)\s+(?P<date>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b[^.\n]{0,120}\b(?:president|prime\s+minister|minister)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return _parse_human_date(match.group("date"))
    return None


def _parse_human_date(value: str) -> date | None:
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _age_on_date(birth_date: date, event_date: date) -> int:
    age = event_date.year - birth_date.year
    if (event_date.month, event_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def _format_date(value: date) -> str:
    return value.strftime("%d %B %Y").lstrip("0")


def _age_query_person(question: str) -> str:
    match = re.search(
        r"\bhow\s+old\s+was\s+(?P<name>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'.-]+(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'.-]+){1,4})\s+when\b",
        question or "",
        flags=re.IGNORECASE,
    )
    return match.group("name").strip() if match else ""


def _office_start_query_person(question: str) -> str:
    match = re.search(
        rf"\bwhen\s+did\s+(?P<name>{WEB_NAME_TOKEN}(?:\s+{WEB_NAME_TOKEN}){{1,4}})\s+(?:become|became|take\s+office|took\s+office|assume\s+office|assumed\s+office|come\s+(?:into|to)\s+power|came\s+(?:into|to)\s+power)\b",
        question or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    candidate = _clean_person_name(match.group("name"))
    return candidate if _looks_like_person_name(candidate) else ""


def _office_start_query_role_country(question: str) -> str:
    role_markers = "|".join(
        re.escape(marker)
        for marker in sorted(
            (marker for marker, _role in conversation_role_patterns()),
            key=len,
            reverse=True,
        )
    )
    match = re.search(
        rf"\bwhen\s+did\s+(?:the\s+)?(?P<role>{role_markers})(?:\s+of\s+(?P<country>.+?))?\s+(?:come|came)\s+(?:into|to)\s+power\b",
        question or "",
        flags=re.IGNORECASE,
    )
    if match:
        return _normalize_role_country(match.group("role"), match.group("country") or question)
    match = re.search(
        r"\b(?:become|became)\s+(?P<role>.+?)\??$",
        question or "",
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\b(?:take|took|assume|assumed)\s+office\s+as\s+(?P<role>.+?)\??$",
            question or "",
            flags=re.IGNORECASE,
        )
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group("role")).strip(" ?!.")


def _normalize_role_country(role_text: str, country_text: str) -> str:
    role = _government_role_from_text(normalize_intent_text(role_text)) or re.sub(
        r"\s+",
        " ",
        (role_text or "").strip(),
    ).title()
    country = _country_from_text(country_text or "")
    if country:
        return f"{role} of {_country_phrase(country)}"
    return role


def _age_query_role_country(question: str) -> str:
    match = re.search(r"\bwhen\s+he\s+became\s+(?P<role>.+?)\??$", question or "", flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bwhen\s+.+?\s+became\s+(?P<role>.+?)\??$", question or "", flags=re.IGNORECASE)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group("role")).strip(" ?!.")


def _role_country_matches(text: str, role_country: str) -> bool:
    terms = query_terms(role_country)
    haystack = (text or "").lower()
    if not terms:
        return True
    return sum(1 for term in terms if term in haystack) >= min(2, len(terms))


def _source_haystack(source: WebSource) -> str:
    return f"{source.title}. {source.content}"


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _news_answer_from_web(question: str, web_sources: Sequence[WebSource]) -> str:
    sources = [source for source in web_sources if not _is_noisy_web_source(source)]
    if not sources:
        return _fallback_answer_from_web_results(web_sources=web_sources, previous_answer="")
    confidence = "High" if any(_is_news_source(source) for source in sources[:3]) else "Medium"
    focus_terms = _news_focus_terms(question)
    lead = _best_news_sentence(sources[0], focus_terms)
    if lead:
        lines = [f"{lead} [{sources[0].label}]"]
    else:
        lines = [f"The strongest news source is {sources[0].title} [{sources[0].label}]."]

    supporting = []
    for source in sources[1:4]:
        sentence = _best_news_sentence(source, focus_terms)
        if sentence:
            supporting.append(f"- {sentence} [{source.label}]")
    if supporting:
        lines.extend(("", "Supporting sources:", *supporting))
    lines.extend(("", f"Confidence: {confidence}"))
    return "\n".join(lines)


def _news_focus_terms(question: str) -> set[str]:
    terms = set(query_terms(question))
    terms.update({"resign", "resigned", "resignation"} & set((question or "").lower().split()))
    if "why" in (question or "").lower():
        terms.update({"because", "after", "amid", "over", "following", "due"})
    return terms


def _best_news_sentence(source: WebSource, focus_terms: set[str]) -> str:
    text = f"{source.title}. {source.content}"
    scored = []
    for index, sentence in enumerate(_candidate_sentences(text)):
        cleaned = _clean_public_sentence(sentence)
        lower = cleaned.lower()
        if _looks_like_boilerplate_text(cleaned):
            continue
        overlap = sum(1 for term in focus_terms if term in lower)
        news_event_score = sum(
            1
            for marker in ("resigned", "resignation", "resigns", "stepped down", "amid", "over", "after", "because")
            if marker in lower
        )
        score = overlap + news_event_score * 2 - index * 0.03
        if score > 0:
            scored.append((score, cleaned))
    if scored:
        return _compact_source_text(max(scored, key=lambda item: item[0])[1], limit=360)
    return _compact_source_text(_clean_public_sentence(text), limit=320)


def _is_news_source(source: WebSource) -> bool:
    domain = _web_source_domain(source)
    return any(marker in domain for marker in NEWS_WEB_MARKERS)


def _public_knowledge_topics(question: str) -> list[str]:
    lower = (question or "").lower()
    topics = []
    if "toxicology" in lower:
        topics.append("toxicology")
    if "reach" in lower and any(
        marker in lower for marker in ("directive", "regulation", "eu", "european", "chemical")
    ):
        topics.append("reach")
    return topics


def _public_topics_covered(question: str, web_sources: Sequence[WebSource]) -> bool:
    topics = _public_knowledge_topics(question)
    if not topics:
        return bool(_public_topic_from_text(question) and web_sources)
    return all(_best_public_source_for_topic(topic, web_sources) for topic in topics)


def _best_public_source_for_topic(topic: str, web_sources: Sequence[WebSource]) -> WebSource | None:
    scored = []
    for source in web_sources:
        score = _public_topic_source_score(topic, source)
        if score > 0:
            scored.append((score, source))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _public_topic_source_score(topic: str, source: WebSource) -> float:
    haystack = f"{source.title} {source.url} {source.content}".lower()
    title = (source.title or "").lower()
    score = 0.0
    if topic == "toxicology":
        if "toxicology" not in haystack:
            return 0.0
        score += 3.0
        if "definition" in title or "dictionary" in title:
            score += 2.0
        if any(marker in haystack for marker in ("study of", "science", "field of science")):
            score += 1.5
    elif topic == "reach":
        if "reach" not in haystack:
            return 0.0
        if not any(marker in haystack for marker in ("chemical", "echa", "regulation", "authorisation")):
            return 0.0
        score += 3.0
        if "echa" in haystack or "europa.eu" in haystack:
            score += 2.5
        if "regulation" in haystack:
            score += 1.5
    if _is_authoritative_web_source(source):
        score += 2.0
    if isinstance(source.score, (int, float)):
        score += float(source.score or 0.0)
    return score


def _public_topic_answer(question: str, topic: str, source: WebSource) -> str:
    citation = f"[{source.label}]"
    if topic == "toxicology":
        sentence = _best_public_sentence(
            source,
            ("toxicology", "study", "science", "poison", "chemical", "substance"),
        )
        if sentence.lower().startswith("toxicology"):
            fact = sentence
        elif sentence.lower().startswith(("the study", "a field", "a branch")):
            fact = f"Toxicology is {_lower_first(sentence)}"
        else:
            fact = f"Toxicology is described as: {sentence}"
        return f"{fact} {citation}"

    sentence = _best_public_sentence(
        source,
        ("reach", "registration", "evaluation", "authorisation", "restriction", "chemical"),
    )
    expansion = _reach_expansion(sentence) or _reach_expansion(
        f"{source.title}. {source.content}"
    )
    if expansion:
        fact = f"EU REACH is the chemicals regulation known as {expansion}."
    else:
        fact = f"EU REACH is described as: {sentence}"
    if "directive" in (question or "").lower():
        fact += " It is a regulation, not a directive."
    return f"{fact} {citation}"


def _best_public_sentence(source: WebSource, markers: Sequence[str]) -> str:
    text = f"{source.title}. {source.content}"
    scored = []
    for index, sentence in enumerate(_candidate_sentences(text)):
        cleaned = _clean_public_sentence(sentence)
        lower = cleaned.lower()
        if _looks_like_boilerplate_text(cleaned):
            continue
        overlap = sum(1 for marker in markers if marker in lower)
        if overlap:
            scored.append((overlap * 2 - index * 0.03, cleaned))
    if scored:
        return _compact_source_text(max(scored, key=lambda item: item[0])[1], limit=340)
    return _compact_source_text(_clean_public_sentence(text), limit=340)


def _clean_public_sentence(sentence: str) -> str:
    cleaned = re.sub(r"^\s*#+\s*", "", sentence or "")
    cleaned = re.sub(r"\*\*", "", cleaned)
    cleaned = re.sub(r"^\s*definition\s+", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip(" -")


def _reach_expansion(text: str) -> str:
    match = re.search(
        r"Registration,\s*Evaluation,\s*Authori[sz]ation\s+and\s+Restriction\s+of\s+Chemicals",
        text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return _compact_source_text(match.group(0), limit=120)


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[:1].lower() + text[1:]


def _public_knowledge_confidence(topics: Sequence[str], answered: dict[str, WebSource]) -> str:
    if len(answered) < len(topics):
        return "Medium" if answered else "Low"
    if all(_is_authoritative_web_source(source) for source in answered.values()):
        return "High"
    return "Medium"


def _public_topic_label(topic: str) -> str:
    return "EU REACH" if topic == "reach" else topic.title()


def _scientific_subject(question: str) -> str:
    subject = _normalize_scientific_query(_strip_question_prefix(question))
    return _compact_source_text(subject.strip(" ?!."), limit=120) or "This method"


def _best_scientific_fact(evidence_items, terms, markers, exclude=frozenset()):
    excluded = {_normalized_fact(value) for value in exclude if value}
    best: tuple[float, str] | None = None
    for item in evidence_items[:8]:
        evidence_text = _scientific_evidence_text(item)
        item_matches_focus = _sentence_matches_scientific_focus(evidence_text.lower(), terms)
        for sentence in _candidate_sentences(evidence_text):
            sentence = _clean_scientific_sentence(sentence)
            if _noisy_scientific_sentence(sentence):
                continue
            if _normalized_fact(sentence) in excluded:
                continue
            lower = sentence.lower()
            if not item_matches_focus and not _sentence_matches_scientific_focus(lower, terms):
                continue
            marker_score = sum(1 for marker in markers if marker in lower)
            if marker_score <= 0:
                continue
            overlap = sum(1 for term in terms if term in lower)
            score = marker_score * 2 + overlap + float(getattr(item, "score", 0.0) or 0.0)
            fact = f"{_compact_source_text(sentence, limit=320)} {item.citation()}"
            if best is None or score > best[0]:
                best = (score, fact)
    return best[1] if best else ""


def _clean_scientific_sentence(sentence: str) -> str:
    return re.sub(
        r"^(?:algorithmic|abstract|summary|introduction)\s*[•:\-]\s*",
        "",
        sentence or "",
        flags=re.IGNORECASE,
    ).strip()


def _noisy_scientific_sentence(sentence: str) -> bool:
    lower = (sentence or "").lower()
    if lower.startswith(("dtu driven", "download ", "pdf ", "copyright ")):
        return True
    if re.search(r"\bdamian\s+n\.?\b|\bpage\s+\d+\b", lower):
        return True
    if re.search(r"\blog\s*p\b|[≈∑θβ]|\\theta|\\beta|\bn\s+x\b|\bs\s+x\b", lower):
        return True
    symbol_count = sum(1 for char in sentence if not char.isalnum() and not char.isspace())
    letter_count = sum(1 for char in sentence if char.isalpha())
    return bool(len(sentence) > 180 and symbol_count > max(12, letter_count * 0.18))


def _sentence_matches_scientific_focus(lower_sentence: str, terms: Sequence[str]) -> bool:
    term_set = set(terms)
    if "hamiltonian" in term_set and not any(
        marker in lower_sentence for marker in ("hamiltonian", "hmc", "rehmc")
    ):
        return False
    if {"replica", "exchange"}.issubset(term_set) and not re.search(
        r"\breplica[- ]exchange\b|\brehmc\b",
        lower_sentence,
    ):
        return False
    return True


def _normalized_fact(value: str) -> str:
    text = re.sub(r"\[[SW]\d+\]", "", value or "", flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _scientific_evidence_text(item):
    content = getattr(item, "content", "") or ""
    title = getattr(item, "title", "") or ""
    if _looks_like_boilerplate_text(content):
        return title
    return f"{title}. {content}" if title and title.lower() not in content.lower()[:240] else content or title


def _fact_from_item(item, terms):
    sentence = _best_evidence_sentence(_useful_evidence_text(item, terms), terms)
    if sentence:
        return f"{_compact_source_text(sentence, limit=320)} {item.citation()}"
    return f"{_compact_source_text(item.title, limit=220)} {item.citation()}"


def _extractive_answer_lines(question, evidence_items, terms):
    if _identity_tokens(question):
        subject = _compact_source_text(question.strip(" ?!."), limit=80)
        best = evidence_items[0]
        first_description = _compact_source_text(_useful_evidence_text(best, terms), limit=280)
        lines = [f"{subject}: {first_description} {best.citation()}"]
        if len(evidence_items) > 1:
            lines.extend(("", "Additional relevant sources:"))
            for item in evidence_items[1:4]:
                description = _compact_source_text(_useful_evidence_text(item, terms), limit=220)
                lines.append(f"- {description} {item.citation()}")
        return lines

    best = evidence_items[0]
    sentence = _best_evidence_sentence(_useful_evidence_text(best, terms), terms)
    if sentence:
        return [f"{sentence} {best.citation()}"]
    return [f"I found relevant evidence in {best.citation()}, but it only contains a brief source summary."]


def _useful_evidence_text(item, terms):
    content = getattr(item, "content", "") or ""
    title = getattr(item, "title", "") or ""
    if _looks_like_boilerplate_text(content):
        return title
    sentence = _best_evidence_sentence(content, terms)
    if sentence and sentence.lower() != title.lower():
        return f"{title}: {sentence}" if title else sentence
    return title or content


def _useful_source_text(source):
    if source is None:
        return ""
    content = getattr(source, "content", "") or ""
    title = getattr(source, "title", "") or ""
    return title if _looks_like_boilerplate_text(content) else content or title


def _looks_like_boilerplate_text(text):
    lower = (text or "").lower()
    return any(
        marker in lower
        for marker in (
            "cookies on ",
            "essential cookies",
            "analytics cookies",
            "you've accepted analytics cookies",
            "enable javascript",
            "please enable cookies",
        )
    )


def _extractive_confidence(evidence_items):
    if not evidence_items:
        return "Low"
    best_score = max(float(getattr(item, "score", 0.0) or 0.0) for item in evidence_items)
    labels = {getattr(item, "citation_label", "") for item in evidence_items if getattr(item, "citation_label", "")}
    authoritative = any(getattr(item, "authority", None) and str(item.authority).endswith("OFFICIAL") for item in evidence_items)
    if best_score >= 0.78 and (len(labels) >= 2 or authoritative):
        return "High"
    if best_score >= 0.5 or labels:
        return "Medium"
    return "Low"


def _best_evidence_sentence(text, terms):
    scored = []
    for index, cleaned in enumerate(_candidate_sentences(text)):
        lower = cleaned.lower()
        overlap = sum(1 for term in terms if term in lower)
        score = overlap * 2 - index * 0.02
        if overlap:
            scored.append((score, cleaned))
    if not scored:
        return _compact_source_text(text, limit=300)
    best = max(scored, key=lambda item: item[0])[1]
    return _compact_source_text(best, limit=320)


def _candidate_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    values = []
    for sentence in sentences:
        cleaned = " ".join(sentence.split())
        if len(cleaned) >= 24:
            values.append(cleaned)
    return values


def _verify_citations(answer, local_sources, web_sources):
    valid = {s.label for s in local_sources} | {w.label for w in web_sources}
    def replace(match):
        label = f"{match.group(1)}{match.group(2)}"
        return match.group(0) if label in valid else ""
    cleaned = re.sub(r"\[([SW])(\d+)\]", replace, answer or "").strip()
    if _labels_in_answer(cleaned, "S") or _labels_in_answer(cleaned, "W"):
        return cleaned
    if _looks_like_non_answer(cleaned):
        return cleaned
    citation = _best_missing_citation(cleaned, local_sources, web_sources)
    if not citation:
        return cleaned
    return _append_citation_to_answer(cleaned, citation)


def _best_missing_citation(answer, local_sources, web_sources):
    candidates = [*local_sources, *web_sources]
    best_label = ""
    best_score = 0.0
    answer_terms = set(query_terms(answer))
    if not answer_terms:
        return ""
    for source in candidates:
        text = getattr(source, "text", "") or getattr(source, "content", "")
        source_terms = set(query_terms(text))
        if not source_terms:
            continue
        overlap = len(answer_terms & source_terms) / max(1, min(len(answer_terms), len(source_terms)))
        source_score = float(getattr(source, "score", 0.0) or 0.0)
        score = overlap + min(0.2, source_score * 0.2)
        if score > best_score:
            best_score = score
            best_label = getattr(source, "label", "")
    return best_label if best_score >= 0.28 else ""


def _append_citation_to_answer(answer, citation_label):
    citation = f"[{citation_label}]"
    if not answer:
        return citation
    if answer.endswith(citation):
        return answer
    return f"{answer.rstrip()} {citation}".strip()


def _verify_answer_against_evidence(
    answer,
    local_sources,
    web_sources,
    question,
    settings: AppSettings,
):
    mode = str(getattr(settings, "answer_verification_mode", "heuristic") or "heuristic").lower()
    sources = [*local_sources, *web_sources]
    if mode in {"off", "none", "disabled"}:
        return {"status": "skipped", "score": 0.0, "note": "Verification disabled."}
    if not sources or _looks_like_non_answer(answer):
        return {"status": "skipped", "score": 0.0, "note": "No source-backed claims to verify."}
    cited_labels = _labels_in_answer(answer, "S") | _labels_in_answer(answer, "W")
    cited_sources = [source for source in sources if getattr(source, "label", "") in cited_labels]
    evidence_sources = cited_sources or sources[: min(4, len(sources))]
    answer_terms = set(query_terms(_remove_citation_text(answer)))
    question_terms = set(query_terms(question))
    claim_terms = answer_terms - question_terms
    if not claim_terms:
        claim_terms = answer_terms
    evidence_terms: set[str] = set()
    for source in evidence_sources:
        evidence_terms.update(query_terms(getattr(source, "text", "") or getattr(source, "content", "")))
        evidence_terms.update(query_terms(getattr(source, "document", "") or getattr(source, "title", "")))
    if not claim_terms or not evidence_terms:
        return {"status": "partial", "score": 0.0, "note": "Insufficient text for verification."}
    overlap = len(claim_terms & evidence_terms) / max(1, len(claim_terms))
    min_overlap = float(getattr(settings, "answer_verification_min_overlap", 0.18))
    has_citation = bool(cited_labels)
    if overlap >= min_overlap and has_citation:
        status = "verified"
    elif overlap >= min_overlap * 0.5:
        status = "partial"
    else:
        status = "unsupported"
    note = (
        f"{len(claim_terms & evidence_terms)} of {len(claim_terms)} answer terms overlap "
        f"with {'cited' if cited_sources else 'available'} evidence."
    )
    if not has_citation:
        note += " No valid citation was present in the final answer."
    return {"status": status, "score": round(overlap, 4), "note": note}


def _remove_citation_text(answer: str) -> str:
    return re.sub(r"\[[SW]\d+\]", " ", answer or "", flags=re.IGNORECASE)


def _looks_like_non_answer(answer):
    return any(marker in answer.lower() for marker in ("could not", "cannot", "can't", "i do not know", "i don't know", "no web sources", "not enough information"))


def _compact_source_text(text, limit):
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."


def _labels_in_answer(answer, prefix):
    pattern = re.compile(rf"\[{prefix}(\d+)\]", re.IGNORECASE)
    return {f"{prefix}{match.group(1)}" for match in pattern.finditer(answer or "")}


def _format_history(history, max_chars=3000):
    if not history:
        return "No previous conversation."
    lines = []
    for item in history[-8:]:
        content = getattr(item, "content", "").strip()
        if content:
            lines.append(f"{item.role}: {content}")
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


def _check_generation_stop(should_stop):
    if should_stop and should_stop():
        raise GenerationStopped("Generation stopped by user.")


def _emit_stage(on_stage, label):
    if on_stage:
        try:
            on_stage(label)
        except Exception:
            return


def _clean_error_message(exc):
    return " ".join(str(exc).split())[:500]


def _generation_error_confidence(message):
    lower = (message or "").lower()
    if is_model_selection_warning(message) or "select another model" in lower:
        return "model-selection-warning"
    if "token" in lower or "authentication" in lower or "unauthorized" in lower:
        return "needs-token"
    return "generation-error"


def _append_web_update_note(answer, provider_label, note):
    return f"{answer.rstrip()}\n\nWeb update: {provider_label} {note}"
