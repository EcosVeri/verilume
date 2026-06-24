from __future__ import annotations

from verilume.core.conversation_state import ConversationState
from verilume.core.query_interpreter import QueryInterpretationAgent


def test_unspecified_country_president_asks_clarification_without_state() -> None:
    result = QueryInterpretationAgent().interpret(
        "The president of the country",
        [],
        ConversationState(),
    )

    assert result.needs_clarification
    assert result.clarification_question == "Which country do you mean?"


def test_unspecified_country_president_uses_active_country() -> None:
    state = ConversationState(active_country="Democratic Republic of the Congo")
    result = QueryInterpretationAgent().interpret(
        "The president of the country",
        [],
        state,
    )

    assert not result.needs_clarification
    assert result.resolved_question == (
        "Who is the President of the Democratic Republic of the Congo?"
    )
    assert result.use_web
    assert result.use_local
    assert result.use_model_knowledge


def test_smallest_country_query_uses_local_model_and_web() -> None:
    result = QueryInterpretationAgent().interpret(
        "The smallest country in Europe",
        [],
        ConversationState(),
    )

    assert result.intent == "public_knowledge"
    assert result.use_local
    assert result.use_model_knowledge
    assert not result.use_web
    assert any("smallest country in Europe" in query for query in result.search_queries)


def test_president_of_smallest_country_uses_head_of_state_queries() -> None:
    result = QueryInterpretationAgent().interpret(
        "The president of the smallest country in Europe",
        [],
        ConversationState(),
    )

    assert result.intent == "public_knowledge"
    assert result.use_web
    assert result.use_local
    assert result.use_model_knowledge
    assert any("head of state" in query for query in result.search_queries)


def test_general_people_question_stays_local_first_by_default() -> None:
    result = QueryInterpretationAgent().interpret(
        "Where is Damian from",
        [],
        ConversationState(),
    )

    assert result.intent == "general"
    assert result.use_local
    assert result.use_model_knowledge
    assert not result.use_web
