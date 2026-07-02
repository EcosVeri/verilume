"""Lightweight agentic routing before the expensive RAG pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verilume.core.conversation_state import ConversationState
from verilume.core.evidence import QueryUnderstanding, classify_question
from verilume.core.schemas import ChatMessage


REFERENCE_TERMS = {
    "again",
    "current",
    "former",
    "he",
    "her",
    "him",
    "his",
    "it",
    "its",
    "latter",
    "ones",
    "previous",
    "same",
    "she",
    "that",
    "there",
    "they",
    "this",
    "those",
}
NEWS_TERMS = {
    "ap",
    "bbc",
    "breaking",
    "financial times",
    "ft",
    "guardian",
    "news",
    "news channel",
    "news channels",
    "reuters",
    "sky",
    "sky news",
}
NEWS_OUTLET_ALIASES = {
    "reuters": "Reuters",
    "ap": "AP News",
    "ap news": "AP News",
    "associated press": "AP News",
    "bbc": "BBC News",
    "bbc news": "BBC News",
    "sky": "Sky News",
    "sky news": "Sky News",
    "financial times": "Financial Times",
    "ft": "Financial Times",
    "guardian": "The Guardian",
    "the guardian": "The Guardian",
}
COUNTRY_ALIASES = {
    "britain": "United Kingdom",
    "cameroon": "Cameroon",
    "cameroun": "Cameroon",
    "democratic republic of congo": "Democratic Republic of the Congo",
    "democratic republic of the congo": "Democratic Republic of the Congo",
    "dr congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "france": "France",
    "french": "France",
    "great britain": "United Kingdom",
    "luxembourg": "Luxembourg",
    "netherlands": "Netherlands",
    "norway": "Norway",
    "norwegian": "Norway",
    "rdc": "Democratic Republic of the Congo",
    "république démocratique du congo": "Democratic Republic of the Congo",
    "the congo": "Democratic Republic of the Congo",
    "the netherlands": "Netherlands",
    "the uk": "United Kingdom",
    "the united kingdom": "United Kingdom",
    "the united states": "United States",
    "u k": "United Kingdom",
    "u s": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "united states": "United States",
    "usa": "United States",
}
COUNTRY_ROLE_MARKERS = (
    "prime minister",
    "premier",
    "president",
    "secretary of state",
    "king",
    "queen",
    "monarch",
    "grand duke",
    "grand duchess",
    "minister of defence",
    "minister of defense",
    "foreign minister",
    "finance minister",
    "interior minister",
)
COUNTRY_PHRASE_BLOCKLIST = {
    "agency",
    "association",
    "bank",
    "board",
    "club",
    "committee",
    "company",
    "corporation",
    "council",
    "department",
    "foundation",
    "group",
    "institute",
    "ministry",
    "office",
    "party",
    "team",
    "university",
}
DEFAULT_ROLE_PATTERNS = (
    ("prime minister", "Prime Minister"),
    ("premier", "Prime Minister"),
    ("defence minister", "Minister of Defence"),
    ("defense minister", "Minister of Defence"),
    ("minister of defence", "Minister of Defence"),
    ("minister of defense", "Minister of Defence"),
    ("foreign minister", "Minister of Foreign Affairs"),
    ("minister of foreign affairs", "Minister of Foreign Affairs"),
    ("finance minister", "Minister of Finance"),
    ("minister of finance", "Minister of Finance"),
    ("interior minister", "Minister of the Interior"),
    ("president", "President"),
    ("secretary of state", "Secretary of State"),
    ("king", "King"),
    ("queen", "Queen"),
    ("monarch", "Monarch"),
    ("grand duke", "Grand Duke"),
    ("grand duchess", "Grand Duchess"),
    ("ceo", "CEO"),
    ("chief executive officer", "CEO"),
    ("founder", "Founder"),
    ("chair", "Chair"),
    ("chairman", "Chair"),
    ("chairwoman", "Chair"),
    ("director", "Director"),
    ("head of", "Head"),
)
ROLE_REGISTRY: dict[str, str] = dict(DEFAULT_ROLE_PATTERNS)
GOVERNMENT_ROLE_PATTERNS = tuple(ROLE_REGISTRY.items())
DEFAULT_NEWS_OUTLETS = (
    "Reuters",
    "AP News",
    "BBC News",
    "Sky News",
    "Financial Times",
    "The Guardian",
)
ROLE_STOPWORDS = {
    "AP",
    "BBC",
    "CNN",
    "EU",
    "FT",
    "Government",
    "News",
    "Reuters",
    "Sky",
    "Sources",
    "The Guardian",
    "United Kingdom",
    "United States",
    "UK",
    "US",
    "USA",
    "Web",
}
NAME_TOKEN = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'.-]+"
NAME_ACTION_BOUNDARIES = {
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
NAME_ORG_BOUNDARIES = {
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
NAME_SUFFIXES = {"ii", "iii", "iv", "jr", "sr"}


def register_conversation_role(alias: str, canonical: str | None = None) -> None:
    """Register a role alias used by conversation memory and follow-up resolution."""

    key = normalize_intent_text(alias)
    if not key:
        raise ValueError("Role alias cannot be empty.")
    ROLE_REGISTRY[key] = canonical or alias.strip().title()


def conversation_role_patterns() -> tuple[tuple[str, str], ...]:
    """Return registered role aliases sorted by specificity."""

    return tuple(
        sorted(ROLE_REGISTRY.items(), key=lambda item: len(item[0]), reverse=True)
    )


@dataclass(slots=True)
class ExtractedEntities:
    """Entities inferred from text without a heavyweight NLP dependency."""

    persons: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IntentRoute:
    """Routing decision for a user message."""

    route: str
    answer: str = ""
    uses_rag: bool = True
    diagnostics: dict[str, str | bool] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationResolution:
    """Resolved user message plus diagnostics for the RAG pipeline."""

    original_question: str
    resolved_question: str
    state: ConversationState
    is_followup: bool = False
    news_intent: bool = False
    requested_sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchPlan:
    """Explicit retrieval plan produced before touching local or web indexes."""

    intent: str = "general"
    entity: str = ""
    topic: str = ""
    country: str = ""
    role: str = ""
    preferred_sources: list[str] = field(default_factory=list)
    need_local: bool = True
    need_web: bool = False
    need_model: bool = True
    freshness_required: bool = False
    response_mode: str = "answer"

    def diagnostics(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "entity": self.entity,
            "topic": self.topic,
            "country": self.country,
            "role": self.role,
            "preferred_sources": self.preferred_sources,
            "need_local": self.need_local,
            "need_web": self.need_web,
            "need_model": self.need_model,
            "freshness_required": self.freshness_required,
            "response_mode": self.response_mode,
        }


class ConversationContextAgent:
    """Resolve follow-up references before intent routing and retrieval."""

    def resolve(
        self,
        question: str,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
        state: ConversationState | None = None,
    ) -> ConversationResolution:
        inferred_state = self.state_from_history(history)
        state = _merge_conversation_states(inferred_state, state) if state else inferred_state
        has_prior_context = bool(history) or _state_has_context(state)
        normalized = normalize_intent_text(question)
        topic_switch = _topic_switch_state(question, normalized)
        if (
            topic_switch
            and not _looks_like_topic_head_followup(normalized, state)
            # A pronoun still bound to the person in focus is a continuation, not
            # a topic switch — keep the focused person rather than wiping it.
            and not _pronoun_points_to_specific_person(state, normalized)
        ):
            state = topic_switch
        current_country = _country_from_text(question)
        if current_country:
            state.active_country = current_country
            state.active_topics = _unique_nonempty(
                [_country_topic(current_country), *state.active_topics]
            )
        current_role = _government_role_from_text(normalized)
        if current_role:
            state.active_role = current_role
        current_research_topic = _research_topic_from_text(question)
        if current_research_topic:
            state.active_research_topic = current_research_topic
            state.active_topics = _unique_nonempty([current_research_topic, *state.active_topics])
        requested_sources = requested_news_sources(question)
        news_intent = _has_news_intent(question) or bool(requested_sources)
        is_followup = has_prior_context and (
            _has_reference(normalized)
            or _looks_like_topic_head_followup(normalized, state)
            or _looks_like_government_role_followup(normalized)
            or _looks_like_short_followup(normalized)
            or _looks_like_source_followup(normalized)
            or news_intent
        )

        resolved = (question or "").strip()
        if is_followup:
            resolved = self._resolve_followup(question, state, requested_sources, news_intent)
        else:
            resolved = _expand_scientific_abbreviation(resolved)

        return ConversationResolution(
            original_question=question,
            resolved_question=resolved or question,
            state=state,
            is_followup=is_followup,
            news_intent=news_intent,
            requested_sources=requested_sources,
        )

    def state_from_history(
        self,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
    ) -> ConversationState:
        recent = list(history)[-20:]
        state = ConversationState()
        user_items = [item for item in recent if item.role == "user"]
        user_text = "\n".join(item.content for item in user_items)
        all_text = "\n".join(item.content for item in recent)
        assistant_text = "\n".join(item.content for item in recent if item.role == "assistant")

        state.active_country = _recent_country(user_items) or _recent_country(recent)
        state.active_topics = _unique_nonempty(
            _topic_candidates(user_text, state.active_country)
        )
        state.active_role = _recent_government_role(user_items) or _recent_government_role(recent)
        state.roles = _recent_role_holders(recent, state.active_country)
        office_holder = state.roles.get(_role_key(state.active_role), "")
        state.active_entities = _unique_nonempty(
            [office_holder]
            + list(state.roles.values())
            + _entity_candidates_for_topics(assistant_text, state.active_topics)
            + _entity_candidates(user_text)
        )
        state.active_person = office_holder or _best_person_for_topic(
            state.active_entities,
            state.active_topics[0] if state.active_topics else "",
        )
        state.active_research_topic = _recent_research_topic(user_items) or _research_topic_from_text(all_text)
        state.active_web_sources = _unique_nonempty(requested_news_sources(all_text))
        state.preferred_sources = state.active_web_sources
        state.active_dates = _unique_nonempty(re.findall(r"\b(?:20\d{2}|19\d{2}|today|yesterday)\b", all_text, flags=re.IGNORECASE))
        state.active_event = _active_event(all_text)
        state.active_news_story = _news_story_from_state(state)
        state.intent = _conversation_intent(all_text, state)
        state.last_answer_summary = _last_assistant_summary(recent)
        return state

    def _resolve_followup(
        self,
        question: str,
        state: ConversationState,
        requested_sources: list[str],
        news_intent: bool,
    ) -> str:
        normalized = normalize_intent_text(question)
        subject = _conversation_subject(state)
        if not subject:
            return question.strip()

        if _looks_like_source_followup(normalized):
            return _source_followup_query(
                question,
                subject=subject,
                event=state.active_event,
                requested_sources=requested_sources,
                news_intent=news_intent,
            )

        # When a pronoun refers to a specific person the user just brought into
        # focus (not merely the sticky role holder), resolve the pronoun directly
        # instead of letting a topic-head/role branch collapse the query into a
        # generic role question that would discard that person.
        if _pronoun_points_to_specific_person(state, normalized):
            resolved = _replace_references(
                question.strip(), _reference_subject(state, normalized) or subject
            )
            if news_intent:
                return _news_followup_query(resolved, subject, requested_sources)
            return resolved

        if _looks_like_scientific_invention_followup(normalized) and state.active_research_topic:
            return f"Who introduced {state.active_research_topic}?"

        if _looks_like_scientific_comparison_followup(normalized) and state.active_research_topic:
            comparison_topic = _research_topic_from_text(question)
            if comparison_topic and comparison_topic.lower() != state.active_research_topic.lower():
                return f"How is {state.active_research_topic} different from {comparison_topic}?"

        if _looks_like_topic_head_followup(normalized, state):
            return _expand_topic_head_followup(question, state)

        if _looks_like_age_at_role_followup(normalized):
            return _age_at_role_query(question, state)

        if _looks_like_became_role_followup(normalized):
            return _became_role_query(question, state)

        role = _government_role_from_text(normalized)
        if role and state.active_country and not _pronoun_points_to_specific_person(state, normalized):
            return f"Who is the {role} of {_country_phrase(state.active_country)}?"

        resolved = question.strip()
        if _has_reference(normalized):
            resolved = _replace_references(resolved, _reference_subject(state, normalized) or subject)

        if state.active_event and "resign" in state.active_event and "resign" not in resolved.lower():
            if "why" in normalized:
                resolved = f"Why did {subject} resign?"
            elif news_intent:
                resolved = f"{resolved} {subject} resignation"

        if news_intent:
            return _news_followup_query(resolved, subject, requested_sources)
        return resolved


class IntentRouterAgent:
    """Fast deterministic router for messages that do not need retrieval."""

    _GREETING_PATTERNS = {
        "afternoon",
        "evening",
        "good afternoon",
        "good evening",
        "good morning",
        "hello",
        "hello there",
        "hey",
        "hey there",
        "hi",
        "hi there",
        "hiya",
        "how are you",
        "how are you doing",
        "morning",
        "sup",
        "yo",
    }
    _CONVERSATION_PATTERNS = {
        "alright",
        "awesome",
        "bye",
        "cool",
        "good night",
        "goodbye",
        "great",
        "nice",
        "no",
        "ok",
        "okay",
        "see you",
        "thank you",
        "thanks",
        "thx",
        "yes",
    }
    _IDENTITY_PATTERNS = {
        "are you verilume",
        "what are you",
        "who are you",
    }
    _CAPABILITY_PATTERNS = {
        "capabilities",
        "help",
        "how can you help",
        "what can you do",
        "what do you do",
    }

    def route(self, message: str) -> IntentRoute:
        normalized = normalize_intent_text(message)
        if not normalized:
            return IntentRoute(
                route="empty",
                answer="Send me a question, document task, or research topic and I will help.",
                uses_rag=False,
                diagnostics={"agent": "intent_router"},
            )
        if normalized in self._GREETING_PATTERNS:
            return IntentRoute(
                route="greeting",
                answer=_greeting_answer(),
                uses_rag=False,
                diagnostics={"agent": "intent_router"},
            )
        if normalized in self._CONVERSATION_PATTERNS:
            return IntentRoute(
                route="conversation",
                answer=_conversation_answer(normalized),
                uses_rag=False,
                diagnostics={"agent": "intent_router"},
            )
        if normalized in self._IDENTITY_PATTERNS:
            return IntentRoute(
                route="identity",
                answer=_identity_answer(),
                uses_rag=False,
                diagnostics={"agent": "intent_router"},
            )
        if normalized in self._CAPABILITY_PATTERNS:
            return IntentRoute(
                route="capability",
                answer=_capability_answer(),
                uses_rag=False,
                diagnostics={"agent": "intent_router"},
            )
        return IntentRoute(route="rag", uses_rag=True, diagnostics={"agent": "intent_router"})


class QueryUnderstandingAgent:
    """Small wrapper that keeps query analysis explicit in the pipeline."""

    def understand(self, question: str) -> QueryUnderstanding:
        return classify_question(question)


class SearchPlanningAgent:
    """Create a retrieval plan before expensive search begins."""

    def plan(
        self,
        question: str,
        state: ConversationState,
        query_understanding: QueryUnderstanding,
        *,
        local_file_question: bool,
        news_intent: bool,
        requested_sources: list[str],
    ) -> SearchPlan:
        normalized = normalize_intent_text(question)
        country = _country_from_text(question) or state.active_country
        role = _government_role_from_text(normalized) or state.active_role
        topic = _research_topic_from_text(question) or state.active_research_topic
        entity = _bare_entity_name(question) or state.active_person

        if local_file_question:
            return SearchPlan(
                intent="local_document",
                entity=entity,
                topic=topic,
                country=country,
                role=role,
                preferred_sources=["Active document", "Metadata", "BM25", "Dense retrieval"],
                need_local=True,
                need_web=False,
                need_model=False,
                response_mode="document-grounded",
            )

        if news_intent:
            subject = _conversation_subject(state) if requested_sources else entity or _conversation_subject(state)
            return SearchPlan(
                intent="news",
                entity=subject,
                topic=state.active_news_story or topic,
                country=country,
                role=role,
                preferred_sources=requested_sources or list(DEFAULT_NEWS_OUTLETS),
                need_local=False,
                need_web=True,
                need_model=False,
                freshness_required=True,
                response_mode="current-summary",
            )

        if country and role:
            return SearchPlan(
                intent="government",
                entity=entity,
                country=country,
                role=role,
                preferred_sources=["Government", "Official biography", "Parliament", "Reuters", "Wikipedia"],
                need_local=False,
                need_web=True,
                need_model=False,
                freshness_required=True,
                response_mode="current-fact",
            )

        if _looks_like_person_lookup(question, query_understanding):
            person = _bare_entity_name(question) or entity
            return SearchPlan(
                intent="person",
                entity=person,
                preferred_sources=[
                    "University",
                    "ORCID",
                    "Google Scholar",
                    "GitHub",
                    "LinkedIn",
                    "ResearchGate",
                ],
                need_local=True,
                need_web=True,
                need_model=False,
                response_mode="profile",
            )

        public_topic = _public_topic_from_text(question)
        if public_topic:
            return SearchPlan(
                intent="public_knowledge",
                topic=public_topic,
                preferred_sources=["Web evidence", "Official sources", "Reference sources"],
                need_local=False,
                need_web=True,
                need_model=False,
                response_mode="public-answer",
            )

        if topic or _looks_like_scientific_query(normalized):
            return SearchPlan(
                intent="scientific_definition",
                topic=topic or _scientific_topic_from_question(question),
                preferred_sources=["Local papers", "University", "arXiv", "DOI", "Model knowledge"],
                need_local=True,
                need_web=True,
                need_model=True,
                response_mode="explanation",
            )

        return SearchPlan(
            intent="general",
            entity=entity,
            topic=topic,
            country=country,
            role=role,
            preferred_sources=["Local evidence", "Web", "Model knowledge"],
            need_local=True,
            need_web=bool(query_understanding.requires_web_validation),
            need_model=True,
        )


def normalize_intent_text(message: str) -> str:
    text = re.sub(r"[^a-z0-9' ]+", " ", (message or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def requested_news_sources(text: str) -> list[str]:
    normalized = normalize_intent_text(text)
    sources = []
    for alias, label in NEWS_OUTLET_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            sources.append(label)
    if "news channel" in normalized or "news channels" in normalized:
        sources.extend(DEFAULT_NEWS_OUTLETS)
    return _unique_nonempty(sources)


def _has_news_intent(text: str) -> bool:
    normalized = normalize_intent_text(text)
    if any(term in normalized for term in NEWS_TERMS):
        return True
    return any(term in normalized for term in ("resign", "resigned", "resignation", "breaking"))


def _has_reference(normalized: str) -> bool:
    tokens = set(normalized.split())
    return bool(tokens & REFERENCE_TERMS)


def _looks_like_short_followup(normalized: str) -> bool:
    tokens = normalized.split()
    if not tokens:
        return False
    return len(tokens) <= 6 and normalized.startswith(
        ("and ", "also ", "compare ", "how ", "tell me why", "what about", "what happened", "when ", "which ", "why ")
    )


def _looks_like_government_role_followup(normalized: str) -> bool:
    if not _government_role_from_text(normalized):
        return False
    return not _country_from_text(normalized)


def _looks_like_age_at_role_followup(normalized: str) -> bool:
    return normalized.startswith("how old") and any(
        marker in normalized for marker in ("became", "become", "took office", "assumed office")
    )


def _looks_like_became_role_followup(normalized: str) -> bool:
    if not normalized.startswith("when "):
        return False
    if not _has_office_start_marker(normalized):
        return False
    return bool(_government_role_from_text(normalized) or _has_reference(normalized))


def _has_office_start_marker(normalized: str) -> bool:
    return any(
        marker in normalized
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


def _looks_like_scientific_invention_followup(normalized: str) -> bool:
    return normalized.startswith(("who invented ", "who introduced ", "who developed ")) and _has_reference(normalized)


def _looks_like_scientific_comparison_followup(normalized: str) -> bool:
    return normalized.startswith(("how is ", "how does ", "how are ")) and "different from" in normalized and _has_reference(normalized)


def _looks_like_source_followup(normalized: str) -> bool:
    return bool(
        re.fullmatch(r"(?:search|check|look up)?\s*(?:reuters|ap|ap news|bbc|bbc news|sky|sky news|financial times|ft|guardian|the guardian)(?:\s+(?:and\s+)?(?:tell me )?(?:why|more))?", normalized)
        or normalized.startswith(("search reuters", "search news", "check reuters", "look up reuters"))
    )


def _looks_like_topic_head_followup(normalized: str, state: ConversationState) -> bool:
    topic = state.active_research_topic or (state.active_topics[0] if state.active_topics else "")
    head = _topic_head(topic)
    if not head:
        return False
    return normalized.startswith((f"which {head} ", f"what {head} ", f"which {head}s ", f"what {head}s "))


def _expand_topic_head_followup(question: str, state: ConversationState) -> str:
    topic = state.active_research_topic or (state.active_topics[0] if state.active_topics else "")
    head = _topic_head(topic)
    if not topic or not head:
        return question.strip()
    return re.sub(rf"\b{re.escape(head)}s?\b", topic, question.strip(), count=1, flags=re.IGNORECASE)


def _topic_head(topic: str) -> str:
    words = normalize_intent_text(topic).split()
    if len(words) < 2:
        return ""
    return words[-1]


def _topic_candidates(text: str, active_country: str = "") -> list[str]:
    lower = (text or "").lower()
    topics = []
    if "prime minister" in lower and re.search(r"\b(?:uk|u\.k\.?|united kingdom|britain)\b", lower):
        topics.append("UK Prime Minister")
    if "prime minister" in lower and "luxembourg" in lower:
        topics.append("Luxembourg Prime Minister")
    if _country_from_text(text) and any(
        role in lower
        for role, _label in conversation_role_patterns()
    ):
        topics.append(_country_topic(_country_from_text(text)))
    if "secretary of state" in lower and re.search(r"\b(?:u\.s\.?|us|usa|united states)\b", lower):
        topics.append("U.S. Secretary of State")
    if "reach" in lower:
        topics.append("EU REACH regulation")
    research_topic = _research_topic_from_text(text)
    if research_topic:
        topics.append(research_topic)
    if "hamiltonian monte carlo" in lower or " hmc" in f" {lower}":
        topics.append("Hamiltonian Monte Carlo")
    if "replica exchange hamiltonian" in lower:
        topics.append("Replica Exchange Hamiltonian Monte Carlo")
    if active_country:
        topics.append(_country_topic(active_country))
    return topics


def _recent_country(history: list[ChatMessage]) -> str:
    for item in reversed(history):
        country = _country_from_text(item.content)
        if country:
            return country
    return ""


def _recent_government_role(history: list[ChatMessage]) -> str:
    for item in reversed(history):
        role = _government_role_from_text(normalize_intent_text(item.content))
        if role:
            return role
    return ""


def _topic_switch_state(question: str, normalized: str) -> ConversationState | None:
    if _looks_like_source_followup(normalized) or _has_news_intent(question):
        return None
    country = _country_from_text(question)
    role = _government_role_from_text(normalized)
    topic = _research_topic_from_text(question)
    public_topic = _public_topic_from_text(question)
    bare_entity = _bare_entity_name(question)
    if country and role:
        return ConversationState(
            active_topics=[_country_topic(country)],
            active_country=country,
            active_role=role,
            intent="government",
        )
    if country and _looks_like_country_topic_switch(question, normalized):
        return ConversationState(
            active_topics=[_country_topic(country)],
            active_country=country,
            intent="politics",
        )
    if topic and not _has_reference(normalized):
        return ConversationState(
            active_entities=[topic],
            active_topics=[topic],
            active_research_topic=topic,
            intent="research",
        )
    if public_topic and not _has_reference(normalized):
        return ConversationState(
            active_entities=[public_topic],
            active_topics=[public_topic],
            active_research_topic=public_topic,
            intent="public_knowledge",
        )
    if bare_entity and not country and not role:
        return ConversationState(
            active_entities=[bare_entity],
            active_person=bare_entity if _looks_like_person_name_text(bare_entity) else "",
            active_research_topic=topic,
            active_topics=[topic] if topic else [],
            intent="person" if _looks_like_person_name_text(bare_entity) else "entity",
        )
    return None


def _looks_like_country_topic_switch(question: str, normalized: str) -> bool:
    country = _country_from_text(question)
    if not country:
        return False
    aliases = [alias for alias, value in COUNTRY_ALIASES.items() if value == country]
    country_only_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    if country_only_pattern and re.fullmatch(
        rf"(?:now\s+|what\s+about\s+|and\s+|next\s+)?(?:{country_only_pattern})(?:\s+now)?",
        normalized,
    ):
        return True
    return normalized.startswith(("now ", "what about ", "and ")) and country.lower() in normalized


def _bare_entity_name(question: str) -> str:
    text = re.sub(r"\s+", " ", (question or "").strip().strip(".,;:!?"))
    if not text or re.search(r"[?]", question or ""):
        return ""
    if re.search(r"\b(?:is|are|was|were|has|have|had|do|does|did|should|could|would|will|when|why|how|what|who|search|check|find|look|show|tell)\b", text, re.IGNORECASE):
        return ""
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", text)
    if not 2 <= len(words) <= 5:
        return ""
    normalized = [word.strip("'’").lower() for word in words]
    if any(word in {"minister", "president", "prime", "finance", "foreign", "defence", "defense"} for word in normalized):
        return ""
    if any(len(word) <= 2 for word in normalized):
        return ""
    return text


def _looks_like_person_lookup(question: str, query_understanding: QueryUnderstanding) -> bool:
    if _bare_entity_name(question) and _looks_like_person_name_text(_bare_entity_name(question)):
        return True
    normalized = normalize_intent_text(question)
    return bool(
        getattr(query_understanding, "personal_company_entity_lookup", False)
        and not _country_from_text(question)
        and not _government_role_from_text(normalized)
    )


def _looks_like_person_name_text(value: str) -> bool:
    words = [word for word in re.split(r"\s+", (value or "").strip()) if word]
    if not 2 <= len(words) <= 5:
        return False
    return all(word[:1].isupper() for word in words[:2])


def _recent_office_holder(history: list[ChatMessage], role: str, country: str) -> str:
    if not role:
        return ""
    for item in reversed(history):
        if item.role != "assistant":
            continue
        holder = _office_holder_from_text(item.content, role, country)
        if holder:
            return holder
    return ""


def _recent_role_holders(history: list[ChatMessage], country: str) -> dict[str, str]:
    holders: dict[str, str] = {}
    for role in _government_roles():
        holder = _recent_office_holder(history, role, country)
        if holder:
            holders[_role_key(role)] = holder
    return holders


def _government_roles() -> list[str]:
    seen = set()
    roles = []
    for _marker, role in conversation_role_patterns():
        key = _role_key(role)
        if key in seen:
            continue
        seen.add(key)
        roles.append(role)
    return roles


def _role_key(role: str) -> str:
    return normalize_intent_text(role)


def _office_holder_from_text(text: str, role: str, country: str = "") -> str:
    role_pattern = _role_pattern(role)
    country_pattern = _country_pattern(country)
    prefix_pattern = (
        rf"\b(?:the\s+)?(?:current\s+)?{role_pattern}\s+"
        rf"(?!of\b)(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,4}})\b"
    )
    for match in re.finditer(prefix_pattern, text or "", flags=re.IGNORECASE):
        name = _clean_name(match.group("name"))
        if _looks_like_name(name):
            return _role_prefixed_name(role, name)
    patterns = (
        rf"\b(?:the\s+)?(?:current\s+)?{role_pattern}\s+of\s+{country_pattern}\s+(?:is|:)\s+(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,4}})\b",
        rf"\b(?:the\s+)?(?:current\s+)?{role_pattern}\s+(?:is|:)\s+(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,4}})\b",
        rf"\b(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,4}})\s+(?:is|has\s+been|serves\s+as|served\s+as)\s+(?:the\s+)?(?:current\s+)?{role_pattern}(?:\s+of\s+{country_pattern})?\b",
        rf"\b(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,4}})\s*,\s+(?:the\s+)?(?:current\s+)?{role_pattern}\s+of\s+{country_pattern}\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            name = _clean_name(match.group("name"))
            if _looks_like_name(name):
                return name
    return ""


def _role_prefixed_name(role: str, name: str) -> str:
    if len(name.split()) <= 1:
        return f"{role} {name}".strip()
    return name


def _role_pattern(role: str) -> str:
    return re.escape(role).replace(r"\ ", r"\s+")


def _country_pattern(country: str) -> str:
    if not country:
        return rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,6}}"
    return re.escape(_country_phrase(country)).replace(r"\ ", r"\s+")


def _country_from_text(text: str) -> str:
    normalized = normalize_intent_text(text)
    for alias in sorted(COUNTRY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return COUNTRY_ALIASES[alias]
    return _country_from_role_phrase(text)


def _country_from_role_phrase(text: str) -> str:
    role_pattern = "|".join(
        re.escape(role).replace(r"\ ", r"\s+")
        for role in sorted(COUNTRY_ROLE_MARKERS, key=len, reverse=True)
    )
    pattern = rf"\b(?:{role_pattern})\s+(?:of|in|for)\s+(?P<country>[^?.,;:\n]+)"
    for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
        country = _clean_country_phrase(match.group("country"))
        if country:
            return country
    return ""


def _clean_country_phrase(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip(" .,;:-")
    text = re.split(
        r"\b(?:currently|current|latest|now|today|tomorrow|yesterday|official|incumbent|biography)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,;:-")
    text = re.sub(r"^(?:the)\s+", "", text, flags=re.IGNORECASE).strip()
    words = [word.strip(" '’\"()[]{}") for word in text.split() if word.strip(" '’\"()[]{}")]
    if not 1 <= len(words) <= 6:
        return ""
    normalized = normalize_intent_text(" ".join(words))
    if not normalized or normalized in COUNTRY_PHRASE_BLOCKLIST:
        return ""
    if any(word in COUNTRY_PHRASE_BLOCKLIST for word in normalized.split()):
        return ""
    return _title_country_phrase(" ".join(words))


def _title_country_phrase(value: str) -> str:
    small_words = {"and", "of", "the"}
    words = []
    for word in re.split(r"\s+", value.strip()):
        lower = word.lower()
        if lower in small_words:
            words.append(lower)
        elif word.isupper() and len(word) <= 3:
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)


def _country_topic(country: str) -> str:
    if not country:
        return ""
    if country == "France":
        return "French politics"
    if country == "United Kingdom":
        return "UK politics"
    if country == "United States":
        return "U.S. politics"
    if country == "Democratic Republic of the Congo":
        return "DR Congo politics"
    return f"{country} politics"


def _country_phrase(country: str) -> str:
    if country in {
        "Democratic Republic of the Congo",
        "Netherlands",
        "United Kingdom",
        "United States",
    }:
        return f"the {country}"
    return country


def _government_role_from_text(normalized: str) -> str:
    for marker, role in conversation_role_patterns():
        if re.search(rf"\b{re.escape(marker)}\b", normalized):
            return role
    return ""


def _entity_candidates_for_topics(text: str, topics: list[str]) -> list[str]:
    candidates = []
    for topic in topics:
        if topic == "UK Prime Minister":
            candidates.extend(_names_near_role(text, "prime minister"))
        elif topic == "Luxembourg Prime Minister":
            candidates.extend(_names_near_role(text, "prime minister"))
        elif topic == "U.S. Secretary of State":
            candidates.extend(_names_near_role(text, "secretary of state"))
    return candidates


def _names_near_role(text: str, role: str) -> list[str]:
    values = []
    patterns = (
        rf"\b(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,4}})\s+(?:is|was|became|has been)\s+(?:the\s+)?(?:current\s+)?{re.escape(role)}\b",
        rf"\b(?:current\s+)?{re.escape(role)}(?:\s+of\s+{NAME_TOKEN}(?:\s+{NAME_TOKEN})*)?\s+(?:is|was|became|has been)\s+(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,4}})\b",
        rf"\b(?P<name>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,4}})\b[^.\n]{{0,120}}\b{re.escape(role)}\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            name = _clean_name(match.group("name"))
            if _looks_like_name(name):
                values.append(name)
    return values


def _entity_candidates(text: str) -> list[str]:
    values = []
    for match in re.finditer(rf"\b{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}}\b", text or ""):
        name = _clean_name(match.group(0))
        if _looks_like_name(name):
            values.append(name)
    return values


def extract_entities_from_text(text: str) -> ExtractedEntities:
    """Extract people, countries, organizations, topics, and role holders from text."""

    value = text or ""
    entities = ExtractedEntities()
    entities.countries = _unique_nonempty(
        [country for country in (_country_from_text(value),) if country]
    )
    entities.topics = _unique_nonempty(
        [
            _research_topic_from_text(value),
            _public_topic_from_text(value),
            *(_topic_candidates(value, entities.countries[0]) if entities.countries else _topic_candidates(value)),
        ]
    )
    role_holders: dict[str, str] = {}
    for _alias, role in conversation_role_patterns():
        holder = _office_holder_from_text(value, role, entities.countries[0] if entities.countries else "")
        if holder:
            role_holders[_role_key(role)] = holder
    entities.roles = role_holders
    entities.persons = _unique_nonempty(
        [
            *role_holders.values(),
            *_entity_candidates(value),
        ]
    )
    entities.organizations = _organization_candidates(value)
    return entities


def _organization_candidates(text: str) -> list[str]:
    values = []
    pattern = (
        r"\b(?P<org>[A-Z][A-Za-z0-9&'.-]+"
        r"(?:\s+(?:University|Institute|Agency|Commission|Council|Ministry|Department|"
        r"Corporation|Company|Inc|Ltd|LLC|Foundation|Parliament|Government|Bank|Group|Lab|Laboratory))"
        r"(?:\s+[A-Z][A-Za-z0-9&'.-]+){0,3})\b"
    )
    for match in re.finditer(pattern, text or ""):
        org = re.sub(r"\s+", " ", match.group("org")).strip(" .,;:-")
        if org and org not in ROLE_STOPWORDS:
            values.append(org)
    return _unique_nonempty(values)


def _clean_name(value: str) -> str:
    text = re.sub(r"^(?:the\s+)?(?:(?:rt\s+hon|hon|sir|dr|mr|mrs|ms)\s+)+", "", value or "", flags=re.IGNORECASE)
    text = re.sub(r"^(?:of|in|from|for|to)\s+", "", text, flags=re.IGNORECASE)
    text = _trim_name_at_boundary(text)
    text = re.sub(r"\s+", " ", text).strip(" .,;:-")
    text = _normalize_name_case(text)
    return text


def _looks_like_name(value: str) -> bool:
    if not value or value in ROLE_STOPWORDS:
        return False
    normalized = normalize_intent_text(value)
    if _looks_like_country_fragment(value):
        return False
    if _government_role_from_text(normalized) and normalized == _role_key(_government_role_from_text(normalized)):
        return False
    words = [word for word in re.split(r"\s+", value.strip()) if word]
    if not 1 <= len(words) <= 5:
        return False
    lower = value.lower()
    if any(term.lower() == lower for term in ROLE_STOPWORDS):
        return False
    if _name_has_boundary_term(value):
        return False
    return not re.search(
        r"\b(?:answer|confidence|current|evidence|government|minister|news|prime|source|sources|state|united)\b",
        lower,
    )


def _looks_like_country_fragment(value: str) -> bool:
    normalized = normalize_intent_text(value)
    if not normalized:
        return False
    if normalized.endswith(" of"):
        return True
    country_names = {
        normalize_intent_text(country)
        for country in [*COUNTRY_ALIASES.keys(), *COUNTRY_ALIASES.values()]
        if country
    }
    if normalized in country_names:
        return True
    if len(normalized.split()) >= 2:
        return any(country.startswith(f"{normalized} ") for country in country_names)
    return False


def _trim_name_at_boundary(value: str) -> str:
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
        if index > 0 and folded in NAME_ACTION_BOUNDARIES:
            break
        if index >= 2 and folded in NAME_ORG_BOUNDARIES:
            break
        if index >= 2 and _looks_like_organization_acronym(token):
            break
        kept.append(raw_token)
    return " ".join(kept)


def _looks_like_organization_acronym(token: str) -> bool:
    cleaned = token.strip(" ,.;:()[]{}")
    return bool(re.fullmatch(r"[A-Z][A-Z0-9&./-]{1,}", cleaned)) and cleaned.lower() not in NAME_SUFFIXES


def _normalize_name_case(value: str) -> str:
    words = []
    for word in re.split(r"\s+", value or ""):
        bare = word.strip(" ,.;:()[]{}")
        if len(bare) > 1 and bare.isupper() and bare.lower() not in NAME_SUFFIXES:
            words.append(word.replace(bare, bare.title()))
        else:
            words.append(word)
    return " ".join(words)


def _name_has_boundary_term(value: str) -> bool:
    words = [word.strip(" ,.;:()[]{}").lower().strip("'’") for word in re.split(r"\s+", value or "")]
    return any(word in NAME_ACTION_BOUNDARIES or word in NAME_ORG_BOUNDARIES for word in words)


def _active_event(text: str) -> str:
    lower = (text or "").lower()
    if "resign" in lower or "resignation" in lower:
        return "resignation"
    if "introduced" in lower:
        return "introduced"
    return ""


def _last_assistant_summary(history: list[ChatMessage]) -> str:
    for item in reversed(history):
        if item.role != "assistant":
            continue
        text = " ".join((item.content or "").split())
        return text[:300]
    return ""


def _conversation_subject(state: ConversationState) -> str:
    topic = state.active_topics[0] if state.active_topics else ""
    person = _best_person_for_topic(state.active_entities, topic)
    if topic == "UK Prime Minister":
        return f"UK Prime Minister {person}" if person else "the UK Prime Minister"
    if topic == "Luxembourg Prime Minister":
        return f"Luxembourg Prime Minister {person}" if person else "the Luxembourg Prime Minister"
    if topic == "U.S. Secretary of State":
        return f"U.S. Secretary of State {person}" if person else "the U.S. Secretary of State"
    if state.active_news_story:
        return state.active_news_story
    if state.active_country and "politics" in topic.lower():
        return f"{state.active_country} government"
    if topic:
        return topic
    if state.active_person:
        return state.active_person
    if state.active_country:
        return f"{state.active_country} government"
    return person


def _pronoun_points_to_specific_person(state: ConversationState, normalized: str) -> str:
    """True when a person pronoun in the query resolves to a specific person who
    is not merely the sticky role holder.

    Guards the "collapse to generic role question" shortcut: "He is the current
    prime minister of Luxembourg" must resolve "he" to the person in focus
    (e.g. Luc Frieden), not be rewritten into "Who is the prime minister of
    Luxembourg?" — which would silently discard the entity the user named.
    """
    tokens = set(normalized.split())
    if not ({"he", "him", "his", "she", "her"} & tokens):
        return False
    person = state.active_person
    if not person:
        return False
    role_holder = state.roles.get(_role_key(state.active_role), "") if state.active_role else ""
    return person != role_holder


def _reference_subject(state: ConversationState, normalized: str) -> str:
    tokens = set(normalized.split())
    person_pronouns = {"he", "him", "his", "she", "her"}
    object_pronouns = {"it", "its", "ones", "this", "that", "there", "those", "same", "previous"}
    if person_pronouns & tokens and state.active_person:
        return state.active_person
    if object_pronouns & tokens:
        return (
            state.active_research_topic
            or state.active_law
            or state.active_document
            or state.active_dataset
            or _conversation_subject(state)
        )
    return _conversation_subject(state)


def _best_person_for_topic(entities: list[str], topic: str) -> str:
    if not entities:
        return ""
    for entity in entities:
        if topic and any(part in entity for part in ("United Kingdom", "United States")):
            continue
        if len(entity.split()) >= 2:
            return entity
    return entities[0]


def _replace_references(question: str, subject: str) -> str:
    resolved = re.sub(r"\b(?:he|him|his|she|her|it|its|they|them|their)\b", subject, question, flags=re.IGNORECASE)
    resolved = re.sub(r"\b(?:ones|this|that|there|those|same|previous)\b", subject, resolved, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", resolved).strip()


def _source_followup_query(
    question: str,
    *,
    subject: str,
    event: str,
    requested_sources: list[str],
    news_intent: bool,
) -> str:
    source_text = " ".join(requested_sources) if requested_sources else "news"
    normalized = normalize_intent_text(question)
    if "why" in normalized and event == "resignation":
        return f"Search {source_text} for why {subject} resigned."
    if "resign" in normalized:
        return f"Search {source_text} for whether {subject} has resigned."
    if event == "resignation":
        return f"Search {source_text} for {subject} resignation."
    if "why" in normalized:
        return f"Search {source_text} for why {subject}."
    if news_intent:
        return f"Search {source_text} for {subject}."
    return f"Search {source_text} for {subject}."


def _age_at_role_query(question: str, state: ConversationState) -> str:
    normalized = normalize_intent_text(question)
    explicit_role = _government_role_from_text(normalized) or _role_reference_from_text(normalized)
    role_text = _strip_age_reference(question)
    role = explicit_role or (state.active_role if not role_text else "")
    subject = _subject_for_role(state, role) or state.active_person
    country = state.active_country
    if role and country and not subject:
        return f"How old was the {role} of {_country_phrase(country)} when they took office?"
    subject = subject or _conversation_subject(state)
    if not subject:
        return question.strip()
    if role and country:
        return f"How old was {subject} when he became {role} of {_country_phrase(country)}?"
    if role_text:
        return f"How old was {subject} when he became {role_text}?"
    if state.active_role and country:
        return f"How old was {subject} when he became {state.active_role} of {_country_phrase(country)}?"
    return f"How old was {subject} when he took office?"


def _became_role_query(question: str, state: ConversationState) -> str:
    normalized = normalize_intent_text(question)
    role = _government_role_from_text(normalized) or _role_reference_from_text(normalized) or state.active_role
    subject = _subject_for_role(state, role) or state.active_person
    country = state.active_country
    if role and country and not subject:
        return f"When did the {role} of {_country_phrase(country)} take office?"
    subject = subject or _reference_subject(state, normalized)
    if not subject:
        return question.strip()
    if role and country:
        return f"When did {subject} become {role} of {_country_phrase(country)}?"
    if role:
        return f"When did {subject} become {role}?"
    return f"When did {subject} take office?"


def _subject_for_role(state: ConversationState, role: str) -> str:
    return state.get_person_by_role(role) if role else ""


def _role_reference_from_text(normalized: str) -> str:
    for marker, role in conversation_role_patterns():
        if re.search(rf"\b(?:the\s+)?{re.escape(marker)}\b", normalized):
            return role
    return ""


def _strip_age_reference(question: str) -> str:
    cleaned = re.sub(
        r"^\s*how\s+old\s+(?:was|will|did|does)?\s*(?:he|she|they|it)?\s*(?:become|became|when\s+became)?\s*",
        "",
        question or "",
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ?!.")


def _expand_scientific_abbreviation(question: str) -> str:
    text = question or ""
    text = re.sub(r"\bHMC\b", "Hamiltonian Monte Carlo", text)
    text = re.sub(r"\bMCMC\b", "Markov Chain Monte Carlo", text)
    return re.sub(r"\s+", " ", text).strip()


def _recent_research_topic(history: list[ChatMessage]) -> str:
    for item in reversed(history):
        topic = _research_topic_from_text(item.content)
        if topic:
            return topic
    return ""


def _research_topic_from_text(text: str) -> str:
    lower = (text or "").lower()
    if "replica exchange hamiltonian monte carlo" in lower:
        return "Replica Exchange Hamiltonian Monte Carlo"
    if "hamiltonian monte carlo" in lower or re.search(r"\bhmc\b", lower):
        return "Hamiltonian Monte Carlo"
    if "markov chain monte carlo" in lower or re.search(r"\bmcmc\b", lower):
        return "Markov Chain Monte Carlo"
    if "bayesian" in lower and "inference" in lower:
        return "Bayesian inference"
    return ""


def _public_topic_from_text(text: str) -> str:
    normalized = normalize_intent_text(text)
    if not normalized:
        return ""
    patterns = (
        r"^(?:name|list|identify|show)(?: me)?(?: the)? (?P<topic>.+?)(?: in the world| around the world| worldwide| globally| in history| over time| in the last \d+ years| during the last \d+ years|$)",
        r"^(?:which|what) (?P<topic>.+?) (?:have|has|are|were|was|erupted|occurred|exist|exist in|are found|can be found)\b",
        r"^(?:what are|what were)(?: the)? (?P<topic>.+?)(?: in the world| around the world| worldwide| globally| in history|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        topic = _clean_public_topic(match.group("topic"))
        if topic:
            return topic
    return ""


def _clean_public_topic(value: str) -> str:
    topic = re.sub(r"\b(?:that|which|who|where|when|with|from|not only|only)\b.*$", "", value or "")
    topic = re.sub(r"^(?:the|a|an)\s+", "", topic)
    topic = re.sub(r"\b(?:all|some|major|main|known|famous)\b", "", topic)
    topic = re.sub(r"\s+", " ", topic).strip(" ,.;:-")
    if not topic or len(topic) < 4:
        return ""
    if any(marker in topic for marker in ("local file", "document", "uploaded")):
        return ""
    return topic


def _scientific_topic_from_question(question: str) -> str:
    return _research_topic_from_text(question)


def _looks_like_scientific_query(normalized: str) -> bool:
    return bool(
        _research_topic_from_text(normalized)
        or normalized.startswith(("explain ", "define ", "what is ", "what are "))
        and any(term in normalized for term in ("algorithm", "bayesian", "monte carlo", "mcmc", "hmc", "statistic", "model"))
    )


def _news_story_from_state(state: ConversationState) -> str:
    if state.active_event == "resignation" and state.active_country:
        return f"{state.active_country} government resignation"
    return ""


def _conversation_intent(text: str, state: ConversationState) -> str:
    if _has_news_intent(text):
        return "news"
    if state.active_country:
        return "politics"
    if state.active_research_topic:
        return "research"
    return ""


def _news_followup_query(question: str, subject: str, requested_sources: list[str]) -> str:
    normalized = normalize_intent_text(question)
    if any(source.lower() in normalized for source in requested_sources):
        return question
    source_text = " ".join(requested_sources or DEFAULT_NEWS_OUTLETS[:3])
    if subject.lower() not in question.lower():
        return f"{question.rstrip(' ?!.')} about {subject} from {source_text}."
    return question


def _unique_nonempty(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = set()
    unique = []
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


def _state_has_context(state: ConversationState) -> bool:
    return bool(
        state.active_country
        or state.active_person
        or state.active_role
        or state.active_topics
        or state.roles
        or state.active_news_story
        or state.active_research_topic
    )


def _merge_conversation_states(base: ConversationState, override: ConversationState | None) -> ConversationState:
    if override is None:
        return base
    merged = ConversationState(
        active_topic=override.active_topic or base.active_topic,
        active_entities=_unique_nonempty([*override.active_entities, *base.active_entities]),
        active_topics=_unique_nonempty([*override.active_topics, *base.active_topics]),
        active_documents=_unique_nonempty([*override.active_documents, *base.active_documents]),
        active_web_sources=_unique_nonempty([*override.active_web_sources, *base.active_web_sources]),
        active_dates=_unique_nonempty([*override.active_dates, *base.active_dates]),
        active_country=override.active_country or base.active_country,
        active_person=override.active_person or base.active_person,
        active_role=override.active_role or base.active_role,
        active_company=override.active_company or base.active_company,
        active_organization=override.active_organization or base.active_organization,
        active_law=override.active_law or base.active_law,
        active_document=override.active_document or base.active_document,
        active_research_topic=override.active_research_topic or base.active_research_topic,
        active_dataset=override.active_dataset or base.active_dataset,
        active_news_story=override.active_news_story or base.active_news_story,
        entities=[*override.entities, *base.entities],
        intent=override.intent or base.intent,
        preferred_sources=_unique_nonempty([*override.preferred_sources, *base.preferred_sources]),
        roles={**base.roles, **override.roles},
        expires_after=override.expires_after or base.expires_after,
        active_event=override.active_event or base.active_event,
        last_answer_summary=override.last_answer_summary or base.last_answer_summary,
        last_resolved_question=override.last_resolved_question or base.last_resolved_question,
    )
    if merged.active_role and not merged.active_person:
        merged.active_person = merged.roles.get(_role_key(merged.active_role), "")
    return merged


def _question_focus_person(question: str) -> str:
    """The named person a question is directly asking about, if any.

    Returns "" when the question names no person, or when it is a role query
    ("who is the prime minister of X") — those are handled by the roles map, not
    by the pronoun antecedent.
    """
    text = (question or "").strip()
    if not text:
        return ""
    normalized = normalize_intent_text(text)
    if _government_role_from_text(normalized):
        return ""
    persons = _entity_candidates(text)
    if not persons:
        return ""
    return next((person for person in persons if len(person.split()) >= 2), persons[0])


def update_state_from_answer(
    state: ConversationState,
    *,
    question: str,
    resolved_query: str,
    answer: str,
) -> ConversationState:
    updated = _merge_conversation_states(ConversationState(), state)
    text = "\n".join([question or "", resolved_query or "", answer or ""])
    extracted = extract_entities_from_text(text)
    country = _country_from_text(text) or updated.active_country
    role = _government_role_from_text(normalize_intent_text(resolved_query or question)) or updated.active_role
    public_topic = _public_topic_from_text(resolved_query or question)
    if not country and extracted.countries:
        country = extracted.countries[0]
    if country:
        updated.active_country = country
        updated.active_topic = updated.active_topic or _country_topic(country)
        updated.active_topics = _unique_nonempty([_country_topic(country), *updated.active_topics])
        updated.remember_entity(country, "country")
    for role_key, holder in extracted.roles.items():
        updated.remember_role(role_key, holder)
    if role:
        updated.active_role = role
        holder = _office_holder_from_text(answer, role, country) or updated.get_person_by_role(role)
        if holder:
            updated.remember_role(role, holder)
            updated.active_person = holder
            updated.active_entities = _unique_nonempty([holder, *updated.active_entities])
            updated.remember_entity(holder, "person", role)
    if not updated.active_person and extracted.persons:
        updated.active_person = extracted.persons[0]
        updated.remember_entity(updated.active_person, "person")
    # When the user's question is *about* a specific named person (e.g. "Who is
    # Luc Frieden"), shift the pronoun antecedent to that person even if a role
    # holder from an earlier turn is still remembered. This is what lets a later
    # "he ..." resolve to the person the user actually just asked about, instead
    # of stubbornly pointing back at the sticky role holder. Explicit role
    # follow-ups still resolve via the roles map, so they are unaffected.
    focus_person = _question_focus_person(question)
    if focus_person:
        updated.active_person = focus_person
        updated.remember_entity(focus_person, "person")
    updated.active_entities = _unique_nonempty(
        [updated.active_person, *extracted.persons, *updated.active_entities]
    )
    if extracted.organizations:
        updated.active_organization = updated.active_organization or extracted.organizations[0]
        updated.active_company = updated.active_company or extracted.organizations[0]
    if public_topic:
        updated.active_topic = public_topic
        updated.active_research_topic = public_topic
        updated.active_topics = _unique_nonempty([public_topic, *updated.active_topics])
        updated.active_entities = _unique_nonempty([public_topic, *updated.active_entities])
        updated.remember_entity(public_topic, "topic")
    if extracted.topics:
        updated.active_topics = _unique_nonempty([*extracted.topics, *updated.active_topics])
    updated.last_answer_summary = _last_assistant_summary([ChatMessage(role="assistant", content=answer)])
    updated.last_resolved_question = resolved_query or updated.last_resolved_question
    updated.preferred_sources = _unique_nonempty([*requested_news_sources(text), *updated.preferred_sources])
    updated.intent = _conversation_intent(text, updated) or updated.intent
    updated.active_news_story = _news_story_from_state(updated) or updated.active_news_story
    return updated


def _greeting_answer() -> str:
    return (
        "Hello. I'm Verilume.\n\n"
        "I can help with uploaded documents, scientific literature, current web information, "
        "programming, data analysis, research papers, PDFs, tables, images, transparent citations, "
        "and conflicting evidence.\n\n"
        "How can I help you today?"
    )


def _conversation_answer(normalized: str) -> str:
    if normalized in {"bye", "goodbye", "good night", "see you"}:
        return "Goodbye. I will be here when you want to continue."
    if normalized in {"thanks", "thank you", "thx"}:
        return "You're welcome. Send me a question, document task, or research topic whenever you're ready."
    if normalized in {"yes", "no", "ok", "okay", "alright"}:
        return "Got it. Tell me what you would like to do next."
    return "Glad to hear it. What would you like to explore next?"


def _identity_answer() -> str:
    return (
        "I'm Verilume, a local-first AI research assistant designed to search your documents, "
        "scientific literature, and trusted web sources while providing transparent citations."
    )


def _capability_answer() -> str:
    return (
        "I can search local PDFs, search the web, summarize and compare documents, read tables, "
        "analyze images, explain code, analyze data, answer scientific questions, provide citations, "
        "and reconcile conflicting evidence."
    )
