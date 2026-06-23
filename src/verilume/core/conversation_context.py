"""Conversation memory helpers for follow-up query resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

from verilume.core.agents import (
    ConversationContextAgent,
    ConversationResolution,
    ConversationState,
    ExtractedEntities,
    IntentRoute,
    IntentRouterAgent,
    SearchPlan,
    SearchPlanningAgent,
    extract_entities_from_text,
    register_conversation_role,
    update_state_from_answer,
)
from verilume.core.evidence import classify_question
from verilume.core.schemas import ChatMessage


@dataclass(slots=True)
class ConversationEntity:
    """Structured entity kept in working memory."""

    name: str
    entity_type: str
    role: str | None = None
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.8


def resolve_conversation_context(
    question: str,
    history: list[ChatMessage] | tuple[ChatMessage, ...],
    state: ConversationState | None = None,
) -> ConversationResolution:
    """Resolve pronouns, role references, and source follow-ups before retrieval."""

    return ConversationContextAgent().resolve(question, history, state)


def route_simple_input(message: str) -> IntentRoute:
    """Fast route for greetings and other messages that should not invoke RAG."""

    return IntentRouterAgent().route(message)


def search_plan_for_question(
    question: str,
    state: ConversationState | None = None,
) -> SearchPlan:
    """Create the lightweight retrieval plan used before local or web search."""

    return SearchPlanningAgent().plan(
        question,
        state or ConversationState(),
        classify_question(question),
        local_file_question=False,
        news_intent=False,
        requested_sources=[],
    )


__all__ = [
    "ConversationEntity",
    "ConversationResolution",
    "ConversationState",
    "ExtractedEntities",
    "SearchPlan",
    "extract_entities_from_text",
    "register_conversation_role",
    "resolve_conversation_context",
    "route_simple_input",
    "search_plan_for_question",
    "update_state_from_answer",
]
