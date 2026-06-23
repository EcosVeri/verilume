"""LLM-first query interpretation for semantic conversation handling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from verilume.core.agents import (
    _country_from_text,
    _country_phrase,
    _government_role_from_text,
    _public_topic_from_text,
    normalize_intent_text,
    requested_news_sources,
)
from verilume.core.conversation_state import ConversationState
from verilume.core.query_preprocessing import normalize_query, query_variants
from verilume.core.schemas import ChatMessage


INTERPRETER_SYSTEM_PROMPT = """You are Verilume's Query Interpretation Agent.

Your job is not to answer the question.

Your job is to:
- understand the user's current question
- detect whether it is a follow-up
- resolve pronouns and implicit references
- use conversation state when helpful
- detect ambiguity
- rewrite the question into a standalone search query
- decide which sources are needed
- generate search queries

Return valid JSON only.

Rules:
- If the question can be answered from conversation state, still produce a resolved question.
- If the question is ambiguous and cannot be resolved, set needs_clarification=true.
- Do not invent facts.
- Do not answer the question.
- Keep local search enabled for most answerable questions so local evidence is checked first.
- Use model knowledge for stable general knowledge when local files do not answer.
- Use web search for current or changing facts, news, or when the user explicitly asks to search the web.
- For uploaded/local document questions, prefer local search.
- For news requests, prefer news sources.
- For scientific explanations, use local files, model knowledge, and scientific web sources.
- If the user asks "Reuters", "BBC", or "news", combine that source with the active topic.
"""


@dataclass(slots=True)
class InterpretedQuery:
    original_question: str
    resolved_question: str
    intent: str = "general"
    is_follow_up: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None
    entities: list[dict[str, Any]] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    use_local: bool = True
    use_web: bool = False
    use_model_knowledge: bool = True
    search_queries: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def normalized_search_queries(self) -> list[str]:
        return _unique_nonempty([*self.search_queries, self.resolved_question])


class QueryInterpretationAgent:
    """Resolve user intent into a standalone plan before retrieval starts."""

    def __init__(self, generator: Any | None = None) -> None:
        self.generator = generator

    def interpret(
        self,
        question: str,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
        state: ConversationState,
    ) -> InterpretedQuery:
        llm_result = self._interpret_with_llm(question, history, state)
        if llm_result is not None:
            return _enforce_source_policy(llm_result)
        return _enforce_source_policy(_fallback_interpretation(question, history, state))

    def _interpret_with_llm(
        self,
        question: str,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
        state: ConversationState,
    ) -> InterpretedQuery | None:
        from verilume.core.generation import BaseGenerator

        if not isinstance(self.generator, BaseGenerator):
            return None
        chat = getattr(self.generator, "chat", None)
        if not callable(chat):
            return None
        messages = [
            {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{_format_history(history)}\n\n"
                    f"Conversation state:\n{_state_json(state)}\n\n"
                    f"Current user question:\n{question}\n\n"
                    "Return JSON with:\n"
                    "original_question\nresolved_question\nintent\nis_follow_up\n"
                    "needs_clarification\nclarification_question\nentities\npreferred_sources\n"
                    "use_local\nuse_web\nuse_model_knowledge\nsearch_queries"
                ),
            },
        ]
        try:
            content = chat(messages).strip()
        except Exception:
            return None
        data = _json_object(content)
        if not isinstance(data, dict):
            return None
        try:
            interpreted = InterpretedQuery(
                original_question=str(data.get("original_question") or question),
                resolved_question=str(data.get("resolved_question") or question).strip() or question,
                intent=str(data.get("intent") or "general"),
                is_follow_up=bool(data.get("is_follow_up")),
                needs_clarification=bool(data.get("needs_clarification")),
                clarification_question=data.get("clarification_question") or None,
                entities=_coerce_entities(data.get("entities")),
                preferred_sources=_string_list(data.get("preferred_sources")),
                use_local=bool(data.get("use_local", True)),
                use_web=bool(data.get("use_web", False)),
                use_model_knowledge=bool(data.get("use_model_knowledge", True)),
                search_queries=_string_list(data.get("search_queries")),
                diagnostics={"query_interpreter": "llm"},
            )
        except Exception:
            return None
        if interpreted.needs_clarification and not interpreted.clarification_question:
            interpreted.clarification_question = "Can you clarify what you mean?"
        return interpreted


def apply_interpretation_to_state(
    state: ConversationState,
    interpretation: InterpretedQuery,
) -> ConversationState:
    updated = ConversationState(
        active_topic=state.active_topic,
        active_country=state.active_country,
        active_person=state.active_person,
        active_document=state.active_document,
        active_news_story=state.active_news_story,
        entities=list(state.entities),
        roles=dict(state.roles),
        preferred_sources=list(state.preferred_sources),
        last_answer_summary=state.last_answer_summary,
        last_resolved_question=interpretation.resolved_question or state.last_resolved_question,
        active_entities=list(state.active_entities),
        active_topics=list(state.active_topics),
        active_documents=list(state.active_documents),
        active_web_sources=list(state.active_web_sources),
        active_dates=list(state.active_dates),
        active_role=state.active_role,
        active_company=state.active_company,
        active_organization=state.active_organization,
        active_law=state.active_law,
        active_research_topic=state.active_research_topic,
        active_dataset=state.active_dataset,
        intent=interpretation.intent or state.intent,
        expires_after=state.expires_after,
        active_event=state.active_event,
    )
    if interpretation.intent == "person":
        updated.active_country = ""
        updated.active_role = ""
        updated.roles = {}
        updated.active_topic = ""
        updated.active_topics = []
    for entity in interpretation.entities:
        name = str(entity.get("name") or "").strip()
        entity_type = str(entity.get("type") or "").strip().lower()
        role = str(entity.get("role") or "").strip()
        if not name:
            continue
        updated.remember_entity(name, entity_type or "entity", role or None)
        if entity_type == "country":
            updated.active_country = name
            updated.active_topic = updated.active_topic or f"{name} government"
            updated.active_topics = _unique_nonempty([updated.active_topic, *updated.active_topics])
        elif entity_type == "person":
            updated.active_person = name
            if role:
                updated.remember_role(role, name)
        elif entity_type in {"document", "file"}:
            updated.active_document = name
        elif entity_type in {"topic", "research_topic"}:
            updated.active_topic = name
            updated.active_research_topic = name
            updated.active_topics = _unique_nonempty([name, *updated.active_topics])
    updated.preferred_sources = _unique_nonempty(
        [*interpretation.preferred_sources, *updated.preferred_sources]
    )
    return updated


def _fallback_interpretation(
    question: str,
    history: list[ChatMessage] | tuple[ChatMessage, ...],
    state: ConversationState,
) -> InterpretedQuery:
    original = (question or "").strip()
    normalized = normalize_intent_text(original)
    explicit_country = _country_from_text(original)
    if explicit_country.lower() == "country":
        explicit_country = ""
    country = explicit_country or state.active_country
    role = _government_role_from_text(normalized)
    requested_sources = requested_news_sources(original)
    active_subject = _active_subject(state)
    public_topic = _public_topic_from_text(original)
    normalized_query = normalize_query(original)
    bare_person = _bare_person_query(original)

    if _asks_for_unspecified_country_president(normalized) and not country:
        return InterpretedQuery(
            original_question=original,
            resolved_question=original,
            intent="clarification",
            is_follow_up=bool(history) or _state_has_context(state),
            needs_clarification=True,
            clarification_question="Which country do you mean?",
            use_local=False,
            use_web=False,
            use_model_knowledge=False,
            diagnostics={"query_interpreter": "fallback"},
        )

    if requested_sources:
        source_only = _source_only_request(normalized)
        topic = (
            f"{state.active_country} government"
            if source_only and state.active_country
            else _news_topic_from_state(state) or active_subject
        )
        resolved = original
        if topic and source_only:
            resolved = f"What does {', '.join(requested_sources)} report about {topic}?"
        elif topic and (
            _has_pronoun_reference(normalized)
            or re.search(r"\b(?:why|more|whether)\b", normalized)
        ):
            resolved = f"What does {', '.join(requested_sources)} report about {topic}?"
        queries = _source_queries(requested_sources, topic or original)
        return InterpretedQuery(
            original_question=original,
            resolved_question=resolved,
            intent="news",
            is_follow_up=bool(topic),
            entities=_entities_for(country=country, person=state.active_person, topic=topic),
            preferred_sources=requested_sources,
            use_local=False,
            use_web=True,
            use_model_knowledge=False,
            search_queries=queries,
            diagnostics={"query_interpreter": "fallback"},
        )

    if bare_person:
        return InterpretedQuery(
            original_question=original,
            resolved_question=bare_person,
            intent="person",
            is_follow_up=False,
            entities=_entities_for(person=bare_person),
            preferred_sources=[
                "University",
                "ORCID",
                "Google Scholar",
                "GitHub",
                "ResearchGate",
            ],
            use_local=True,
            use_web=True,
            use_model_knowledge=False,
            search_queries=_query_forms(bare_person),
            diagnostics={"query_interpreter": "fallback"},
        )

    if normalized.startswith("how old") and active_subject:
        resolved = _age_followup_question(original, state, active_subject)
        return InterpretedQuery(
            original_question=original,
            resolved_question=resolved,
            intent="current_or_public_fact",
            is_follow_up=True,
            entities=_entities_for(
                country=state.active_country,
                person=state.active_person or active_subject,
                role=state.active_role,
            ),
            preferred_sources=["official biography", "reliable web"],
            use_local=False,
            use_web=True,
            use_model_knowledge=False,
            search_queries=_query_forms(resolved),
            diagnostics={"query_interpreter": "fallback"},
        )

    superlative_country = _superlative_country_query(original)
    if superlative_country:
        return InterpretedQuery(
            original_question=original,
            resolved_question=original,
            intent="public_knowledge",
            is_follow_up=False,
            entities=_entities_for(topic=superlative_country),
            preferred_sources=["official sources", "reliable web", "reference sources"],
            use_local=False,
            use_web=True,
            use_model_knowledge=True,
            search_queries=_superlative_country_queries(original),
            diagnostics={"query_interpreter": "fallback"},
        )

    if country and role:
        resolved = _government_role_question(original, country, role, state)
        return InterpretedQuery(
            original_question=original,
            resolved_question=resolved,
            intent="government",
            is_follow_up=country == state.active_country and not explicit_country,
            entities=_entities_for(country=country, person=state.get_person_by_role(role), role=role),
            preferred_sources=["Government", "Official biography", "Reuters", "Wikipedia"],
            use_local=False,
            use_web=True,
            use_model_knowledge=False,
            search_queries=_government_queries(country, role, resolved, state.get_person_by_role(role)),
            diagnostics={"query_interpreter": "fallback"},
        )

    if role and state.active_country:
        country = state.active_country
        resolved = _government_role_question(original, country, role, state)
        return InterpretedQuery(
            original_question=original,
            resolved_question=resolved,
            intent="government",
            is_follow_up=True,
            entities=_entities_for(country=country, person=state.get_person_by_role(role), role=role),
            preferred_sources=["Government", "Official biography", "Reuters", "Wikipedia"],
            use_local=False,
            use_web=True,
            use_model_knowledge=False,
            search_queries=_government_queries(country, role, resolved, state.get_person_by_role(role)),
            diagnostics={"query_interpreter": "fallback"},
        )

    topic_followup = _expand_topic_reference_followup(original, state)
    if topic_followup:
        return InterpretedQuery(
            original_question=original,
            resolved_question=topic_followup,
            intent="public_knowledge",
            is_follow_up=True,
            entities=_entities_for(topic=state.active_topic or state.active_research_topic),
            preferred_sources=state.preferred_sources or ["reliable web"],
            use_local=False,
            use_web=True,
            use_model_knowledge=True,
            search_queries=_query_forms(topic_followup),
            diagnostics={"query_interpreter": "fallback"},
        )

    if _has_pronoun_reference(normalized) and active_subject:
        is_scientific_origin_followup = normalized.startswith(
            ("who invented ", "who introduced ", "who developed ")
        ) and (
            state.active_research_topic or state.active_topic
        )
        if is_scientific_origin_followup:
            topic = state.active_research_topic or state.active_topic
            resolved = f"Who introduced {topic}?"
        else:
            resolved = _replace_reference(original, active_subject)
        if _has_office_start_language(normalized) and state.active_role and state.active_country:
            holder = state.get_person_by_role(state.active_role) or state.active_person or active_subject
            resolved = f"When did {holder} become {state.active_role} of {_country_phrase(state.active_country)}?"
        return InterpretedQuery(
            original_question=original,
            resolved_question=resolved,
            intent="scientific_definition" if is_scientific_origin_followup else state.intent or "follow_up",
            is_follow_up=True,
            entities=_entities_for(
                country=state.active_country,
                person=state.active_person or active_subject,
                role=state.active_role,
                topic=state.active_topic,
            ),
            preferred_sources=(
                ["Local papers", "University", "arXiv", "DOI", "Model knowledge"]
                if is_scientific_origin_followup
                else state.preferred_sources or ["reliable web"]
            ),
            use_local=True if is_scientific_origin_followup else not bool(state.active_country),
            use_web=True if is_scientific_origin_followup else bool(
                state.active_country or requested_sources or state.active_news_story
            ),
            use_model_knowledge=True if is_scientific_origin_followup else not bool(
                state.active_country or requested_sources
            ),
            search_queries=_query_forms(resolved),
            diagnostics={"query_interpreter": "fallback"},
        )

    if public_topic or normalized_query.intent in {"area", "population", "capital"}:
        topic = public_topic or normalized_query.canonical or original
        return InterpretedQuery(
            original_question=original,
            resolved_question=original,
            intent="public_knowledge",
            is_follow_up=False,
            entities=_entities_for(topic=topic),
            preferred_sources=["official sources", "reliable web"],
            use_local=False,
            use_web=True,
            use_model_knowledge=True,
            search_queries=_query_forms(original),
            diagnostics={"query_interpreter": "fallback"},
        )

    if _looks_scientific(normalized):
        topic = normalized_query.canonical or original
        return InterpretedQuery(
            original_question=original,
            resolved_question=original,
            intent="scientific_explanation",
            is_follow_up=False,
            entities=_entities_for(topic=topic),
            preferred_sources=["local files", "scientific web", "model knowledge"],
            use_local=True,
            use_web=True,
            use_model_knowledge=True,
            search_queries=_query_forms(original),
            diagnostics={"query_interpreter": "fallback"},
        )

    local_requested = _local_requested(normalized)
    web_requested = _web_requested(normalized)
    search_queries = _web_request_queries(original) if web_requested else _query_forms(original)
    return InterpretedQuery(
        original_question=original,
        resolved_question=original,
        intent="local_document" if local_requested else "general",
        is_follow_up=False,
        entities=_entities_for(country=country, topic=public_topic),
        preferred_sources=["local files"] if local_requested else ["local evidence", "web", "model knowledge"],
        use_local=True,
        use_web=web_requested,
        use_model_knowledge=not local_requested,
        search_queries=search_queries,
        diagnostics={"query_interpreter": "fallback"},
    )


def _government_role_question(
    question: str,
    country: str,
    role: str,
    state: ConversationState,
) -> str:
    normalized = normalize_intent_text(question)
    holder = state.get_person_by_role(role)
    if _has_office_start_language(normalized):
        subject = holder or f"the {role}"
        return f"When did {subject} become {role} of {_country_phrase(country)}?"
    if holder and _has_pronoun_reference(normalized):
        return _replace_reference(question, holder)
    return f"Who is the {role} of {_country_phrase(country)}?"


def _bare_person_query(question: str) -> str:
    text = re.sub(r"\s+", " ", (question or "").strip().strip(".,;:!?"))
    if not text or "?" in (question or ""):
        return ""
    if re.search(
        r"\b(?:search|check|find|look|show|tell|what|who|when|where|why|how|is|are|does|do|did|minister|president)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return ""
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", text)
    if not 2 <= len(words) <= 5:
        return ""
    if any(len(word) <= 2 for word in words):
        return ""
    return text


def _age_followup_question(question: str, state: ConversationState, subject: str) -> str:
    normalized = normalize_intent_text(question)
    if state.active_role and state.active_country and "president" in normalized:
        return (
            f"How old was {subject} when he became {state.active_role} "
            f"of {_country_phrase(state.active_country)}?"
        )
    match = re.search(
        r"\b(?:he|she|they|him|her)?\s*(?:will\s+)?become\s+(?P<role>.+?)\??$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        role_phrase = re.sub(r"\s+", " ", match.group("role").strip())
        return f"How old was {subject} when he became {role_phrase}?"
    return _replace_reference(question, subject)


def _news_topic_from_state(state: ConversationState) -> str:
    if state.active_country and state.active_role and state.active_person:
        country_label = "UK" if state.active_country == "United Kingdom" else state.active_country
        topic = f"{country_label} {state.active_role} {state.active_person}"
        if state.active_event:
            topic = f"{topic} {state.active_event}"
        return topic
    if state.active_country:
        return f"{state.active_country} government"
    return state.active_news_story or state.active_topic or _topic_from_state(state)


def _expand_topic_reference_followup(question: str, state: ConversationState) -> str:
    topic = state.active_research_topic or state.active_topic or (
        state.active_topics[0] if state.active_topics else ""
    )
    if not topic:
        return ""
    normalized = normalize_intent_text(question)
    topic_head = normalize_intent_text(topic).split()[-1] if normalize_intent_text(topic) else ""
    if normalized.startswith("which ones"):
        return re.sub(r"\bwhich ones\b", f"Which {topic}", question, count=1, flags=re.IGNORECASE)
    if topic_head and normalized.startswith((f"which {topic_head}", f"what {topic_head}")):
        return re.sub(
            rf"\b{re.escape(topic_head)}s?\b",
            topic,
            question,
            count=1,
            flags=re.IGNORECASE,
        )
    return ""


def _government_queries(country: str, role: str, resolved: str, holder: str = "") -> list[str]:
    queries = [
        f"{country} {role} official government",
        f"{country} {role} official",
        f"{role} of {country} official",
        resolved,
    ]
    if holder:
        queries.insert(0, f"{holder} {role} {country} official")
    return _unique_nonempty(queries)


def _source_queries(sources: list[str], topic: str) -> list[str]:
    queries = []
    for source in sources:
        queries.append(f"{source} {topic}".strip())
        domain = _source_domain(source)
        if domain:
            queries.append(f"site:{domain} {topic}".strip())
    return _unique_nonempty(queries)


def _query_forms(question: str) -> list[str]:
    return _unique_nonempty([question, *query_variants(question)])


def _web_request_queries(question: str) -> list[str]:
    cleaned = re.sub(
        r"\b(?:search|check|look up|find|use|the|web|internet|online|about|for)\b",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?!.,:")
    return _query_forms(cleaned or question)


def _superlative_country_query(question: str) -> str:
    normalized = normalize_intent_text(question)
    if re.search(r"\b(?:smallest|largest|biggest|least populated|most populated)\s+country\b", normalized):
        return question.strip()
    if re.search(r"\bpresident of (?:the )?(?:smallest|largest|biggest)\s+country\b", normalized):
        return question.strip()
    return ""


def _superlative_country_queries(question: str) -> list[str]:
    base = question.strip()
    queries = [base]
    normalized = normalize_intent_text(question)
    if "president" in normalized:
        queries.extend(
            [
                f"{base} head of state",
                f"{base} official government",
                "smallest country in Europe head of state",
            ]
        )
    else:
        queries.extend([f"{base} official", f"{base} reference"])
    return _unique_nonempty([*queries, *query_variants(base)])


def _entities_for(
    *,
    country: str = "",
    person: str = "",
    role: str = "",
    topic: str = "",
) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    if person:
        entity = {"name": person, "type": "person"}
        if role:
            entity["role"] = role
        entities.append(entity)
    if country:
        entities.append({"name": country, "type": "country"})
    if topic:
        entities.append({"name": topic, "type": "topic"})
    return entities


def _active_subject(state: ConversationState) -> str:
    if state.active_role:
        holder = state.get_person_by_role(state.active_role)
        if holder:
            return holder
    return state.active_person or state.active_topic or state.active_research_topic


def _topic_from_state(state: ConversationState) -> str:
    return (
        state.active_topic
        or state.active_news_story
        or state.active_research_topic
        or (state.active_topics[0] if state.active_topics else "")
    )


def _asks_for_unspecified_country_president(normalized: str) -> bool:
    return bool(re.search(r"\bpresident of (?:the )?country\b", normalized))


def _source_only_request(normalized: str) -> bool:
    if re.search(r"\b(?:why|whether|resigned|resignation|happened|more)\b", normalized):
        return False
    stripped = re.sub(r"\b(?:search|check|look up|tell me|about|on|and|more|why)\b", " ", normalized)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped in {"reuters", "bbc", "bbc news", "ap", "ap news", "news", "news channels"}


def _has_pronoun_reference(normalized: str) -> bool:
    return bool(re.search(r"\b(?:he|him|his|she|her|they|them|their|it|its|this|that|same)\b", normalized))


def _replace_reference(question: str, subject: str) -> str:
    return re.sub(
        r"\b(?:he|him|his|she|her|they|them|their|it|its|this|that|same)\b",
        subject,
        question,
        count=1,
        flags=re.IGNORECASE,
    )


def _has_office_start_language(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:became|become|came into power|come into power|came to power|come to power|took office|assumed office|in power)\b",
            normalized,
        )
    )


def _looks_scientific(normalized: str) -> bool:
    return bool(
        normalized.startswith(("what is ", "what are ", "explain ", "define "))
        and not re.search(r"\b(?:president|prime minister|king|queen|country|capital|population)\b", normalized)
    )


def _local_requested(normalized: str) -> bool:
    return bool(re.search(r"\b(?:local files?|uploaded documents?|my files?|the files?|which document|which file)\b", normalized))


def _web_requested(normalized: str) -> bool:
    return bool(re.search(r"\b(?:web|online|internet|latest|current|recent|today|news|search)\b", normalized))


def _state_has_context(state: ConversationState) -> bool:
    return bool(
        state.active_country
        or state.active_person
        or state.active_topic
        or state.active_topics
        or state.roles
        or state.last_answer_summary
    )


def _source_domain(source: str) -> str:
    normalized = source.lower()
    if "reuters" in normalized:
        return "reuters.com"
    if "bbc" in normalized:
        return "bbc.com"
    if "ap" in normalized:
        return "apnews.com"
    if "sky" in normalized:
        return "news.sky.com"
    if "financial times" in normalized or normalized == "ft":
        return "ft.com"
    if "guardian" in normalized:
        return "theguardian.com"
    return ""


def _format_history(history: list[ChatMessage] | tuple[ChatMessage, ...]) -> str:
    if not history:
        return "(none)"
    lines = []
    for item in list(history)[-10:]:
        lines.append(f"{item.role}: {item.content}")
    return "\n".join(lines)


def _state_json(state: ConversationState) -> str:
    data = {
        "active_topic": state.active_topic or (state.active_topics[0] if state.active_topics else ""),
        "active_country": state.active_country,
        "active_person": state.active_person,
        "active_document": state.active_document,
        "active_news_story": state.active_news_story,
        "entities": state.entities,
        "roles": state.roles,
        "preferred_sources": state.preferred_sources,
        "last_answer_summary": state.last_answer_summary,
        "last_resolved_question": state.last_resolved_question,
    }
    return json.dumps(data, ensure_ascii=False)


def _json_object(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _coerce_entities(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entities: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict) and item.get("name"):
            entities.append(dict(item))
    return entities


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _enforce_source_policy(result: InterpretedQuery) -> InterpretedQuery:
    question = result.resolved_question or result.original_question
    normalized = normalize_intent_text(question)
    explicit_web_request = _web_requested(normalized)
    current_web_required = _requires_current_web_sources(question, result.intent)

    if result.needs_clarification or result.intent == "clarification":
        result.use_local = False
        result.use_web = False
        result.use_model_knowledge = False
    elif result.intent == "local_document":
        result.use_local = True
        result.use_web = explicit_web_request
        result.use_model_knowledge = False
    else:
        result.use_local = True
        result.use_web = explicit_web_request or current_web_required
        result.use_model_knowledge = True

    result.diagnostics = {
        **dict(result.diagnostics or {}),
        "source_policy": "local_first_model_then_web",
        "source_policy_explicit_web": explicit_web_request,
        "source_policy_current_web": current_web_required,
    }
    return result


def _requires_current_web_sources(question: str, intent: str) -> bool:
    normalized = normalize_intent_text(question)
    if intent in {"news", "government", "current_or_public_fact"}:
        return True
    if not normalized:
        return False
    if re.search(r"\b(?:current|latest|recent|today|now|breaking|updated|newest|most recent)\b", normalized):
        return True
    role = _government_role_from_text(normalized)
    if role and not _has_office_start_language(normalized):
        return True
    return False


def _unique_nonempty(items: list[str]) -> list[str]:
    seen = set()
    values = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", (item or "").strip())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            values.append(cleaned)
    return values
