from __future__ import annotations

import threading
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from verilume.core.schemas import LocalSource, RAGResponse, WebSource
from verilume.settings import AppSettings
from verilume.ui.chat import (
    _GEN_STATE_KEY,
    ASSISTANT_ICON,
    _answer_origin,
    _chat_placeholder,
    _consume_pending_prompt,
    _display_answer,
    _evidence_badges,
    _evidence_detail_rows,
    _format_timestamp,
    _group_web_source_rows,
    _history_bucket,
    _partition_message_history,
    _poll_generation_impl,
    _render_benchmark_report,
    _render_message_history,
    _render_sources,
    _recommendation_for_response,
    _regeneration_plan,
    _source_strength_rows,
    _start_generation,
    _supporting_source_count,
)
from verilume.utils.exporting import chat_to_markdown


class DummyContext:
    def __enter__(self) -> "DummyContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class FakeSessionState(dict):
    """Minimal stand-in for st.session_state supporting dict AND attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class ChatInteractionTests(unittest.TestCase):
    def test_regeneration_plan_keeps_latest_user_prompt_and_drops_answer(self) -> None:
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]

        prompt, trimmed = _regeneration_plan(messages)

        self.assertEqual(prompt, "Second question")
        self.assertEqual(trimmed, messages[:3])

    def test_timestamp_formatting(self) -> None:
        self.assertEqual(
            _format_timestamp("2026-06-16T19:20:00+02:00"),
            "2026-06-16 19:20",
        )

    def test_markdown_export_includes_message_timestamp(self) -> None:
        markdown = chat_to_markdown(
            [
                {
                    "role": "assistant",
                    "content": "Answer",
                    "timestamp": "2026-06-16T19:20:00+02:00",
                }
            ],
            "Verilume",
        )

        self.assertIn("_Timestamp: 2026-06-16 19:20_", markdown)

    def test_chat_placeholder_includes_search_mode_and_example(self) -> None:
        with patch(
            "verilume.ui.chat.st.session_state",
            {"chat_placeholder_example": "Search local files..."},
        ):
            placeholder = _chat_placeholder(AppSettings(search_mode="Local Only"))

        self.assertEqual(placeholder, "📄 Local  Search local files...")

    def test_pending_prompt_is_consumed_once(self) -> None:
        session_state = {"pending_prompt": "  List indexed documents  "}

        with patch("verilume.ui.chat.st.session_state", session_state):
            self.assertEqual(_consume_pending_prompt(), "List indexed documents")
            self.assertIsNone(_consume_pending_prompt())

        self.assertNotIn("pending_prompt", session_state)

    def test_partition_message_history_keeps_recent_messages_visible(self) -> None:
        archived_timestamp = (datetime.now().astimezone() - timedelta(days=4)).isoformat(
            timespec="seconds"
        )
        messages = [
            {
                "role": "assistant" if index % 2 else "user",
                "content": f"Message {index}",
                "timestamp": archived_timestamp,
            }
            for index in range(12)
        ]

        recent, archived = _partition_message_history(
            messages, archive_threshold=10, recent_count=6
        )

        self.assertEqual([index for index, _ in recent], [6, 7, 8, 9, 10, 11])
        self.assertEqual([index for index, _ in archived["Earlier"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual(archived["Today"], [])
        self.assertEqual(archived["Yesterday"], [])

    def test_archived_messages_render_sources_inline_to_avoid_nested_expanders(self) -> None:
        messages = [{"role": "assistant", "content": f"Message {index}"} for index in range(12)]
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", new=SimpleNamespace(messages=messages)),
            patch("verilume.ui.chat.st.markdown"),
            patch("verilume.ui.chat.st.expander", return_value=DummyContext()),
            patch("verilume.ui.chat._render_message") as render_message,
        ):
            _render_message_history(settings)

        inline_indexes = [
            call.args[2]
            for call in render_message.call_args_list
            if call.kwargs.get("source_display") == "inline"
        ]
        default_indexes = [
            call.args[2]
            for call in render_message.call_args_list
            if "source_display" not in call.kwargs
        ]

        self.assertEqual(inline_indexes, [0, 1, 2, 3, 4, 5])
        self.assertEqual(default_indexes, [6, 7, 8, 9, 10, 11])

    def test_history_bucket_uses_relative_day_labels(self) -> None:
        now = datetime.now().astimezone()
        today = now.isoformat(timespec="seconds")
        yesterday = (now - timedelta(days=1)).isoformat(timespec="seconds")
        older = (now - timedelta(days=4)).isoformat(timespec="seconds")

        self.assertEqual(_history_bucket(today), "Today")
        self.assertEqual(_history_bucket(yesterday), "Yesterday")
        self.assertEqual(_history_bucket(older), "Earlier")

    def test_recommendations_are_limited_to_model_availability_states(self) -> None:
        web_error = RAGResponse(
            answer="Web update failed, but here is a model answer.",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="generation-error",
        )
        model_unavailable = RAGResponse(
            answer="Select another model.",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="model-selection-warning",
        )

        self.assertIsNone(_recommendation_for_response(web_error, "test"))
        self.assertIsNotNone(_recommendation_for_response(model_unavailable, "test"))

    def test_display_answer_replaces_web_labels_with_source_badges(self) -> None:
        response = RAGResponse(
            answer="Luc Frieden is prime minister [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="The Government",
                    url="https://gouvernement.lu/en/gouvernement.html",
                    content="Luc Frieden Prime Minister.",
                )
            ],
            used_web=True,
            confidence="web-assisted",
        )

        rendered = _display_answer(response)

        self.assertIn("Luc Frieden is prime minister", rendered)
        self.assertIn("[W1]", rendered)

    def test_display_answer_replaces_ai_source_footer_with_origin_badge(self) -> None:
        response = RAGResponse(
            answer="Churn prediction estimates customer churn.\n\nSource: AI knowledge (not externally verified)",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="model-only",
        )

        self.assertEqual(_display_answer(response), "Churn prediction estimates customer churn.")
        self.assertEqual(
            _answer_origin(response),
            ("\U0001f9e0 AI Knowledge", "Medium", "AI knowledge"),
        )

    def test_answer_origin_uses_na_confidence_for_conversation(self) -> None:
        response = RAGResponse(
            answer="Hello! How can I assist you today?",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="model-only",
        )

        self.assertEqual(
            _answer_origin(response), ("\U0001f9e0 AI Knowledge", "N/A", "Conversation")
        )

    def test_answer_origin_keeps_confidence_for_non_conversational_greeting_topic(self) -> None:
        response = RAGResponse(
            answer="A greeting is a social expression used to acknowledge another person.",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="model-only",
        )

        self.assertEqual(
            _answer_origin(response),
            ("\U0001f9e0 AI Knowledge", "Medium", "AI knowledge"),
        )

    def test_answer_origin_uses_current_information_for_time_sensitive_web_answer(self) -> None:
        response = RAGResponse(
            answer="Luc Frieden is prime minister [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="FRIEDEN Luc - The Luxembourg Government",
                    url="https://gouvernement.lu/en/gouvernement/luc-frieden.html",
                    content="Prime Minister profile.",
                )
            ],
            used_web=True,
            confidence="current-information",
        )

        self.assertEqual(
            _answer_origin(response),
            ("\U0001f310 Current Information", "High", "Web"),
        )

    def test_answer_origin_uses_allowed_label_for_unverified_current_answer(self) -> None:
        response = RAGResponse(
            answer="I could not verify current information from web sources.",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="low",
            diagnostics={"time_sensitive": True},
        )

        self.assertEqual(
            _answer_origin(response),
            ("\U0001f310 Current Information", "Low", "Not verified"),
        )

    def test_answer_origin_uses_allowed_label_for_low_confidence_noncurrent_answer(self) -> None:
        response = RAGResponse(
            answer="I could not answer from the available evidence.",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="low",
        )

        self.assertEqual(
            _answer_origin(response),
            ("\U0001f9e0 AI Knowledge", "Low", "Insufficient evidence"),
        )

    def test_answer_origin_uses_hybrid_when_local_and_web_are_used(self) -> None:
        response = RAGResponse(
            answer="Answer [W1]",
            local_sources=[
                LocalSource(
                    label="S1",
                    document="doc.pdf",
                    page=1,
                    chunk_id="chunk",
                    text="local",
                    score=0.9,
                )
            ],
            web_sources=[
                WebSource(
                    label="W1",
                    title="University article",
                    url="https://www.uni.lu/article",
                    content="University source.",
                )
            ],
            used_web=True,
            confidence="local-web-assisted",
        )

        self.assertEqual(
            _answer_origin(response),
            ("\U0001f500 Hybrid", "High", "Hybrid"),
        )
        self.assertEqual(_supporting_source_count(response), 2)

    def test_answer_origin_prefers_actual_diagnostics_over_local_sufficient_flag(self) -> None:
        response = RAGResponse(
            answer="Spectral analysis decomposes a signal into its component frequencies.",
            local_sources=[
                LocalSource(
                    label="S1",
                    document="notes.pdf",
                    page=3,
                    chunk_id="chunk",
                    text="A local note exists but was not cited.",
                    score=0.88,
                )
            ],
            web_sources=[],
            used_web=False,
            confidence="model-only",
            diagnostics={
                "local_sufficient": True,
                "used_local": False,
                "used_model_knowledge": True,
                "used_web": False,
                "evidence_winner": "model_knowledge",
            },
        )

        self.assertEqual(
            _answer_origin(response),
            ("\U0001f9e0 AI Knowledge", "Medium", "AI knowledge"),
        )

    def test_answer_origin_uses_local_retrieval_for_local_grounded_answer_without_citations(
        self,
    ) -> None:
        response = RAGResponse(
            answer="According to the local context, model diagnostics checks model fit.",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="local-grounded",
            diagnostics={"local_sufficient": True},
        )

        self.assertEqual(
            _answer_origin(response), ("\U0001f4c4 Local Retrieval", "High", "Document")
        )
        self.assertEqual(_supporting_source_count(response), 1)

    def test_answer_origin_uses_local_retrieval_for_local_file_miss(self) -> None:
        response = RAGResponse(
            answer="I could not find this in the indexed local files.",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="low",
            diagnostics={"local_file_question": True},
        )

        self.assertEqual(
            _answer_origin(response),
            ("\U0001f4c4 Local Retrieval", "Low", "Document metadata"),
        )

    def test_answer_origin_scores_local_retrieval_from_chunk_score(self) -> None:
        response = RAGResponse(
            answer="Answer [S1]",
            local_sources=[
                LocalSource(
                    label="S1",
                    document="doc.pdf",
                    page=1,
                    chunk_id="chunk",
                    text="local",
                    score=0.65,
                )
            ],
            web_sources=[],
            used_web=False,
            confidence="local-grounded",
        )

        self.assertEqual(
            _answer_origin(response), ("\U0001f4c4 Local Retrieval", "Medium", "Document")
        )

    def test_evidence_badges_use_response_confidence_and_diagnostics(self) -> None:
        response = RAGResponse(
            answer="Current answer [W1]",
            local_sources=[],
            web_sources=[],
            used_web=True,
            confidence="current-information",
            diagnostics={
                "source_agreement": "high",
                "local_is_older_than_web": True,
                "evidence_winner": "web",
            },
        )

        badges = _evidence_badges(response)

        self.assertIn("Evidence: Current verified", badges)
        self.assertIn("Agreement: High", badges)
        self.assertIn("Freshness: newer web evidence", badges)
        self.assertIn("Winner: Web", badges)

    def test_evidence_detail_rows_show_requested_diagnostics(self) -> None:
        response = RAGResponse(
            answer="Answer",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="low",
            diagnostics={
                "evidence_note": "Local files win for private facts.",
                "freshness_note": "Freshness was not decisive.",
                "local_is_older_than_web": False,
            },
        )

        rows = dict(_evidence_detail_rows(response))

        self.assertEqual(rows["Evidence note"], "Local files win for private facts.")
        self.assertEqual(rows["Freshness note"], "Freshness was not decisive.")
        self.assertEqual(rows["Local older than web"], "No")

    def test_source_strength_rows_show_local_web_and_ai_percentages(self) -> None:
        response = RAGResponse(
            answer="Answer [S1] [W1]",
            local_sources=[
                LocalSource("S1", "doc.pdf", 1, "chunk", "local", 0.95),
            ],
            web_sources=[
                WebSource("W1", "University", "https://www.uni.lu/article", "web", score=0.88),
            ],
            used_web=True,
            confidence="local-web-assisted",
            diagnostics={
                "used_local": True,
                "used_web": True,
                "used_model_knowledge": True,
                "model_sufficient": True,
            },
        )

        rows = _source_strength_rows(response)

        self.assertEqual(rows[0], ("Local", 95, "local"))
        self.assertEqual(rows[1], ("Web", 88, "web"))
        self.assertEqual(rows[2], ("AI", 68, "ai"))

    def test_web_source_rows_are_grouped_by_source_type(self) -> None:
        rows = [
            {
                "Badge": "\U0001f393 University",
                "Source": "University profile",
                "Source type": "University",
                "URL": "https://www.uni.lu/profile",
                "Confidence": "High",
                "Preview": "Profile",
            },
            {
                "Badge": "\U0001f393 University",
                "Source": "Research Explorer",
                "Source type": "University",
                "URL": "https://research.uni.lu/profile",
                "Confidence": "High",
                "Preview": "Research",
            },
            {
                "Badge": "\U0001f464 Social",
                "Source": "LinkedIn",
                "Source type": "Social media",
                "URL": "https://www.linkedin.com/in/person",
                "Confidence": "Low",
                "Preview": "Social",
            },
        ]

        grouped = _group_web_source_rows(rows)

        self.assertEqual(grouped[0][0], "\U0001f393 University Sources (2)")
        self.assertEqual(len(grouped[0][1]), 2)
        self.assertEqual(grouped[1][0], "\U0001f464 Social Sources (1)")

    def test_inline_sources_do_not_create_nested_expanders(self) -> None:
        response = RAGResponse(
            answer="Answer [S1] [W1]",
            local_sources=[
                LocalSource(
                    label="S1",
                    document="doc.pdf",
                    page=1,
                    chunk_id="chunk",
                    text="local text",
                    score=0.9,
                )
            ],
            web_sources=[
                WebSource(
                    label="W1",
                    title="University article",
                    url="https://www.uni.lu/article",
                    content="University source.",
                )
            ],
            used_web=True,
            confidence="local-web-assisted",
        )

        with (
            patch("verilume.ui.chat.st.expander", side_effect=AssertionError("nested expander")),
            patch("verilume.ui.chat.st.container", return_value=DummyContext()) as container,
            patch("verilume.ui.chat.st.markdown"),
            patch("verilume.ui.chat.st.dataframe"),
        ):
            _render_sources(response, AppSettings(), display="inline")

        self.assertEqual(container.call_count, 2)

    def test_benchmark_report_renders_answers_without_nested_expanders(self) -> None:
        response = RAGResponse(
            answer="Benchmark answer",
            local_sources=[],
            web_sources=[],
            used_web=False,
            confidence="medium",
            diagnostics={
                "benchmark_report": {
                    "best_mode": "full",
                    "best_mode_label": "full",
                    "results": [
                        {
                            "mode": "full",
                            "answer": "Full answer [S1]",
                            "confidence": "high",
                            "source_count": 1,
                            "local_source_count": 1,
                            "web_source_count": 0,
                            "latency_seconds": 0.5,
                            "faithfulness_score": 0.8,
                        }
                    ],
                }
            },
        )
        expander_depth = 0

        class GuardedExpander:
            def __enter__(self) -> "GuardedExpander":
                nonlocal expander_depth
                expander_depth += 1
                return self

            def __exit__(self, exc_type, exc, traceback) -> bool:
                nonlocal expander_depth
                expander_depth -= 1
                return False

        def guarded_expander(*_args, **_kwargs) -> GuardedExpander:
            if expander_depth:
                raise AssertionError("nested expander")
            return GuardedExpander()

        with (
            patch("verilume.ui.chat.st.expander", side_effect=guarded_expander) as expander,
            patch("verilume.ui.chat.st.dataframe"),
            patch("verilume.ui.chat.st.caption"),
            patch("verilume.ui.chat.st.markdown") as markdown,
        ):
            _render_benchmark_report(response)

        self.assertEqual(expander.call_count, 1)
        markdown.assert_any_call("**Full answer**")
        markdown.assert_any_call("Full answer [S1]")


class BackgroundGenerationTests(unittest.TestCase):
    """Covers the threaded Stop/Regenerate plumbing in _start_generation/_poll_generation_impl."""

    def test_start_generation_spawns_thread_and_populates_session_state(self) -> None:
        fake_response = SimpleNamespace(answer="The answer", conversation_state=None)
        fake_service = SimpleNamespace(ask=lambda *args, **kwargs: fake_response)
        state = FakeSessionState(
            messages=[{"role": "user", "content": "Q", "timestamp": "t"}],
            conversation_state=SimpleNamespace(),
        )
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.get_rag_service", return_value=fake_service),
        ):
            _start_generation(settings, "Q")
            gen = state[_GEN_STATE_KEY]
            gen["thread"].join(timeout=2)

        self.assertTrue(state.generating)
        self.assertFalse(state.stop_requested)
        self.assertEqual(gen["prompt"], "Q")
        self.assertIsInstance(gen["stop_event"], threading.Event)
        self.assertTrue(gen["result"]["done"])
        self.assertIs(gen["result"]["response"], fake_response)

    def test_start_generation_records_error_from_worker_exception(self) -> None:
        def boom(*_args, **_kwargs) -> None:
            raise ValueError("bad payload")

        fake_service = SimpleNamespace(ask=boom)
        state = FakeSessionState(
            messages=[{"role": "user", "content": "Q", "timestamp": "t"}],
            conversation_state=SimpleNamespace(),
        )
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.get_rag_service", return_value=fake_service),
        ):
            _start_generation(settings, "Q")
            gen = state[_GEN_STATE_KEY]
            gen["thread"].join(timeout=2)

        self.assertTrue(gen["result"]["done"])
        self.assertIn("unexpected response", gen["result"]["error"])

    def test_poll_generation_impl_shows_exactly_one_progress_card_while_running(self) -> None:
        result_box = {"done": False, "stage": "Searching local evidence..."}
        state = FakeSessionState(
            {
                _GEN_STATE_KEY: {
                    "thread": SimpleNamespace(),
                    "stop_event": threading.Event(),
                    "result": result_box,
                    "prompt": "Q",
                }
            },
            stop_requested=False,
            generating=True,
            messages=[],
        )
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.st.columns", return_value=[DummyContext(), DummyContext()]),
            patch("verilume.ui.chat.st.button", return_value=False),
            patch("verilume.ui.chat.st.chat_message", return_value=DummyContext()) as chat_message,
            patch("verilume.ui.chat.st.markdown") as markdown,
        ):
            _poll_generation_impl(settings)

        chat_message.assert_called_once_with("assistant", avatar=ASSISTANT_ICON)
        markdown.assert_called_once()
        self.assertIn(_GEN_STATE_KEY, state)
        self.assertEqual(state.messages, [])

    def test_poll_generation_impl_stop_request_finalizes_immediately_while_worker_runs(
        self,
    ) -> None:
        # The worker is still running (done=False) and blocked in a network call.
        # A toolbar stop_requested must finalize NOW without waiting for the worker.
        stop_event = threading.Event()
        result_box = {"done": False, "stage": "Searching web evidence..."}
        state = FakeSessionState(
            {
                _GEN_STATE_KEY: {
                    "thread": SimpleNamespace(),
                    "stop_event": stop_event,
                    "result": result_box,
                    "prompt": "Q",
                }
            },
            stop_requested=True,
            generating=True,
            messages=[{"role": "user", "content": "Q", "timestamp": "t"}],
        )
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.st.columns", return_value=[DummyContext(), DummyContext()]),
            patch("verilume.ui.chat.st.button", return_value=False),
            patch("verilume.ui.chat.st.chat_message", return_value=DummyContext()),
            patch("verilume.ui.chat.st.markdown"),
            patch("verilume.ui.chat._render_assistant_meta"),
            patch("verilume.ui.chat.st.rerun") as rerun,
        ):
            _poll_generation_impl(settings)

        self.assertTrue(stop_event.is_set())
        self.assertFalse(state.generating)
        self.assertFalse(state.stop_requested)
        self.assertNotIn(_GEN_STATE_KEY, state)
        self.assertEqual(state.messages[-1]["content"], "Generation stopped by user.")
        rerun.assert_called_once()

    def test_poll_generation_impl_stop_button_in_fragment_finalizes_immediately(self) -> None:
        stop_event = threading.Event()
        result_box = {"done": False, "stage": "Interpreting question..."}
        state = FakeSessionState(
            {
                _GEN_STATE_KEY: {
                    "thread": SimpleNamespace(),
                    "stop_event": stop_event,
                    "result": result_box,
                    "prompt": "Q",
                }
            },
            stop_requested=False,
            generating=True,
            messages=[{"role": "user", "content": "Q", "timestamp": "t"}],
        )
        settings = AppSettings()

        # Simulate user clicking the in-fragment Stop button (st.button returns True).
        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.st.columns", return_value=[DummyContext(), DummyContext()]),
            patch("verilume.ui.chat.st.button", return_value=True),
            patch("verilume.ui.chat.st.chat_message", return_value=DummyContext()),
            patch("verilume.ui.chat.st.markdown"),
            patch("verilume.ui.chat._render_assistant_meta"),
            patch("verilume.ui.chat.st.rerun") as rerun,
        ):
            _poll_generation_impl(settings)

        self.assertTrue(stop_event.is_set())
        self.assertNotIn(_GEN_STATE_KEY, state)
        self.assertEqual(state.messages[-1]["content"], "Generation stopped by user.")
        rerun.assert_called_once()

    def test_poll_generation_impl_finalizes_stopped_generation(self) -> None:
        result_box = {"done": True, "stopped": True}
        state = FakeSessionState(
            {
                _GEN_STATE_KEY: {
                    "thread": SimpleNamespace(),
                    "stop_event": threading.Event(),
                    "result": result_box,
                    "prompt": "Q",
                }
            },
            stop_requested=True,
            generating=True,
            messages=[{"role": "user", "content": "Q", "timestamp": "t"}],
        )
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.st.chat_message", return_value=DummyContext()),
            patch("verilume.ui.chat.st.markdown"),
            patch("verilume.ui.chat._render_assistant_meta") as render_meta,
            patch("verilume.ui.chat.st.rerun") as rerun,
        ):
            _poll_generation_impl(settings)

        self.assertFalse(state.generating)
        self.assertFalse(state.stop_requested)
        self.assertNotIn(_GEN_STATE_KEY, state)
        self.assertEqual(len(state.messages), 2)
        self.assertEqual(state.messages[-1]["content"], "Generation stopped by user.")
        render_meta.assert_called_once()
        rerun.assert_called_once()

    def test_poll_generation_impl_finalizes_error_generation(self) -> None:
        result_box = {"done": True, "error": "Something went wrong generating the answer."}
        state = FakeSessionState(
            {
                _GEN_STATE_KEY: {
                    "thread": SimpleNamespace(),
                    "stop_event": threading.Event(),
                    "result": result_box,
                    "prompt": "Q",
                }
            },
            stop_requested=False,
            generating=True,
            messages=[{"role": "user", "content": "Q", "timestamp": "t"}],
        )
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.st.chat_message", return_value=DummyContext()),
            patch("verilume.ui.chat.st.warning") as warning,
            patch("verilume.ui.chat._render_assistant_meta") as render_meta,
            patch("verilume.ui.chat.st.rerun") as rerun,
        ):
            _poll_generation_impl(settings)

        self.assertNotIn(_GEN_STATE_KEY, state)
        self.assertEqual(len(state.messages), 2)
        self.assertEqual(
            state.messages[-1]["content"], "Something went wrong generating the answer."
        )
        warning.assert_called_once()
        render_meta.assert_called_once()
        rerun.assert_called_once()

    def test_poll_generation_impl_finalizes_successful_generation(self) -> None:
        response = SimpleNamespace(answer="The answer", conversation_state="new-state")
        result_box = {"done": True, "response": response}
        state = FakeSessionState(
            {
                _GEN_STATE_KEY: {
                    "thread": SimpleNamespace(),
                    "stop_event": threading.Event(),
                    "result": result_box,
                    "prompt": "Q",
                }
            },
            stop_requested=False,
            generating=True,
            messages=[{"role": "user", "content": "Q", "timestamp": "t"}],
            conversation_state=None,
        )
        settings = AppSettings()

        with (
            patch("verilume.ui.chat.st.session_state", state),
            patch("verilume.ui.chat.st.chat_message", return_value=DummyContext()),
            patch("verilume.ui.chat._render_assistant_meta") as render_meta,
            patch("verilume.ui.chat._render_answer") as render_answer,
            patch("verilume.ui.chat._render_sources") as render_sources,
            patch("verilume.ui.chat._trim_history") as trim_history,
            patch("verilume.ui.chat.st.rerun") as rerun,
        ):
            _poll_generation_impl(settings)

        self.assertFalse(state.generating)
        self.assertNotIn(_GEN_STATE_KEY, state)
        self.assertEqual(state.conversation_state, "new-state")
        self.assertEqual(len(state.messages), 2)
        self.assertEqual(state.messages[-1]["content"], "The answer")
        render_meta.assert_called_once()
        render_answer.assert_called_once()
        render_sources.assert_called_once()
        trim_history.assert_called_once_with(settings.max_chat_messages)
        rerun.assert_called_once()
