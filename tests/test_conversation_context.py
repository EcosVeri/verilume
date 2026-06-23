from __future__ import annotations

from verilume.core.conversation_context import (
    ConversationState,
    extract_entities_from_text,
    resolve_conversation_context,
    route_simple_input,
    search_plan_for_question,
    update_state_from_answer,
)
from verilume.core.schemas import ChatMessage


def test_rdc_president_followup_resolves_role_holder_from_state() -> None:
    state = ConversationState(
        active_country="Democratic Republic of the Congo",
        active_person="Félix Tshisekedi",
        active_role="President",
        roles={"president": "Félix Tshisekedi"},
    )

    result = resolve_conversation_context(
        "When did the president come into power?",
        history=[],
        state=state,
    )

    assert result.resolved_question == (
        "When did Félix Tshisekedi become President of the Democratic Republic of the Congo?"
    )


def test_explicit_role_followup_prefers_role_over_latest_person() -> None:
    history = [
        ChatMessage(role="user", content="Who is president of RDC?"),
        ChatMessage(
            role="assistant",
            content="The current president of the Democratic Republic of the Congo is Félix Tshisekedi [W1].",
        ),
        ChatMessage(role="user", content="Who is the prime minister?"),
        ChatMessage(
            role="assistant",
            content="The current prime minister of the Democratic Republic of the Congo is Judith Suminwa [W2].",
        ),
    ]

    result = resolve_conversation_context("When did the president come into power?", history)

    assert "Félix Tshisekedi" in result.resolved_question
    assert "Judith Suminwa" not in result.resolved_question
    assert "President of the Democratic Republic of the Congo" in result.resolved_question


def test_role_country_phrase_supports_generic_countries() -> None:
    plan = search_plan_for_question("Who is the president of Nigeria?")

    assert plan.intent == "government"
    assert plan.country == "Nigeria"
    assert plan.role == "President"
    assert plan.need_web
    assert not plan.need_local


def test_lightweight_inputs_do_not_use_rag() -> None:
    for message in ("Hi", "Thanks", "Who are you?", "What can you do?"):
        route = route_simple_input(message)
        assert not route.uses_rag


def test_country_only_topic_switch_clears_previous_role_holder() -> None:
    state = ConversationState(
        active_country="Democratic Republic of the Congo",
        active_person="Félix Tshisekedi",
        active_role="President",
        roles={"president": "Félix Tshisekedi"},
    )

    switched = resolve_conversation_context("Now France", history=[], state=state)
    followup = resolve_conversation_context(
        "When did the president come into power?",
        history=[],
        state=switched.state,
    )

    assert switched.state.active_country == "France"
    assert switched.state.roles == {}
    assert "Félix Tshisekedi" not in followup.resolved_question
    assert followup.resolved_question == "When did the President of France take office?"


def test_public_topic_followup_resolves_which_ones() -> None:
    state = ConversationState(
        active_entities=["volcanic lakes"],
        active_topics=["volcanic lakes"],
        active_research_topic="volcanic lakes",
        intent="public_knowledge",
    )

    result = resolve_conversation_context(
        "Which ones have erupted in history?",
        history=[],
        state=state,
    )

    assert result.is_followup
    assert result.resolved_question == "Which volcanic lakes have erupted in history?"


def test_public_topic_followup_restores_topic_modifier() -> None:
    state = ConversationState(
        active_entities=["volcanic lakes"],
        active_topics=["volcanic lakes"],
        active_research_topic="volcanic lakes",
        intent="public_knowledge",
    )

    result = resolve_conversation_context(
        "which lakes have erupted around the world not only in the usa in the last 50 years?",
        history=[],
        state=state,
    )

    assert result.is_followup
    assert result.resolved_question == (
        "which volcanic lakes have erupted around the world not only in the usa in the last 50 years?"
    )


def test_extract_entities_finds_generic_role_holder() -> None:
    entities = extract_entities_from_text(
        "Grand Duke Guillaume has served as Grand Duke of Luxembourg since 3 October 2025."
    )

    assert entities.roles["grand duke"] == "Grand Duke Guillaume"
    assert entities.persons[0] == "Grand Duke Guillaume"
    assert entities.countries == ["Luxembourg"]


def test_state_updates_from_answer_persist_generic_role_memory() -> None:
    state = update_state_from_answer(
        ConversationState(),
        question="Who is the Grand Duke of Luxembourg?",
        resolved_query="Who is the Grand Duke of Luxembourg?",
        answer="Grand Duke Guillaume has served as Grand Duke of Luxembourg since 3 October 2025 [W1].",
    )

    assert state.active_person == "Grand Duke Guillaume"
    assert state.roles["grand duke"] == "Grand Duke Guillaume"
    followup = resolve_conversation_context("When did he become grand duke?", [], state)
    assert followup.resolved_question == "When did Grand Duke Guillaume become Grand Duke of Luxembourg?"
