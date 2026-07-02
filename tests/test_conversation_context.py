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


def test_focus_shifts_to_newly_named_person_for_pronoun_resolution() -> None:
    # Reproduces the Luxembourg conversation from the screenshots.
    # Turn 1: a role answer anchors the role holder as the pronoun antecedent.
    state = update_state_from_answer(
        ConversationState(),
        question="Who is the prime minister of Luxembourg?",
        resolved_query="Who is the prime minister of Luxembourg?",
        answer="The current prime minister of Luxembourg is Xavier Bettel [W1].",
    )
    assert state.active_person == "Xavier Bettel"

    # Turn 2: with no new person named, "he" still resolves to the role holder.
    turn2 = resolve_conversation_context("When did he take over office?", [], state)
    assert "Xavier Bettel" in turn2.resolved_question

    # Turn 3: the user asks about a *different* named person -> focus must shift.
    state = update_state_from_answer(
        state,
        question="Who is Luc Frieden",
        resolved_query="Who is Luc Frieden",
        answer="Are you asking about a specific Luc Frieden? There could be multiple people.",
    )
    assert state.active_person == "Luc Frieden"

    # Turn 4: a later pronoun resolves to the person in focus and is NOT collapsed
    # into a generic role question that silently drops the named entity.
    turn4 = resolve_conversation_context(
        "He is the current prime minister of Luxembourg", [], state
    )
    assert "Luc Frieden" in turn4.resolved_question
    assert "Xavier Bettel" not in turn4.resolved_question
    assert turn4.resolved_question != "Who is the prime minister of Luxembourg?"


def test_question_focus_person_boundaries() -> None:
    from verilume.core.agents import _question_focus_person

    # A question about a named person -> that person.
    assert _question_focus_person("Who is Luc Frieden") == "Luc Frieden"
    assert _question_focus_person("tell me about Angela Merkel") == "Angela Merkel"
    # A role query names no person -> handled by the roles map, not the antecedent.
    assert _question_focus_person("who is the prime minister of luxembourg") == ""
    # A pronoun-only follow-up introduces no new person.
    assert _question_focus_person("when did he take office") == ""
    assert _question_focus_person("") == ""


def test_genuine_topic_switch_without_pronoun_still_switches() -> None:
    # Guard for the topic-switch change: a real switch to a new country with an
    # explicit role (no bound pronoun) must still drop the previous role holder.
    state = ConversationState(
        active_country="Luxembourg",
        active_person="Luc Frieden",
        active_role="prime minister",
        roles={"prime minister": "Luc Frieden"},
        active_topics=["Luxembourg politics"],
    )
    result = resolve_conversation_context("Who is the president of France?", [], state)
    assert "Luc Frieden" not in result.resolved_question
    assert "France" in result.resolved_question


def test_pronoun_points_to_specific_person_guard() -> None:
    from verilume.core.agents import _pronoun_points_to_specific_person

    focused = ConversationState(
        active_person="Luc Frieden",
        active_role="prime minister",
        active_country="Luxembourg",
        roles={"prime minister": "Xavier Bettel"},
    )
    assert _pronoun_points_to_specific_person(
        focused, "he is the current prime minister of luxembourg"
    )
    # No pronoun -> a genuine generic role follow-up should still collapse.
    assert not _pronoun_points_to_specific_person(focused, "and the deputy prime minister")
    # Pronoun but the person is just the role holder -> nothing special.
    anchored = ConversationState(
        active_person="Xavier Bettel",
        active_role="prime minister",
        roles={"prime minister": "Xavier Bettel"},
    )
    assert not _pronoun_points_to_specific_person(anchored, "when did he take office")


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
