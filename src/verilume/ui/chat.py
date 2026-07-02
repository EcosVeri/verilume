"""Chat interface component."""

from __future__ import annotations

import json
import logging
import random
import re
import threading
from datetime import datetime, timedelta
from html import escape
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from verilume.core.conversation_state import ConversationState
from verilume.core.schemas import ChatMessage, RAGResponse
from verilume.rag import GenerationStopped, _claims_entirely_unverified, get_rag_service
from verilume.settings import AppSettings
from verilume.utils.exporting import chat_to_markdown, chat_to_pdf
from verilume.utils.formatting import (
    local_source_confidence,
    local_source_rows,
    source_badge,
    source_confidence,
    source_quality_stars,
    web_source_rows,
)

LOGGER = logging.getLogger(__name__)
ARCHIVE_MESSAGE_THRESHOLD = 10
RECENT_MESSAGE_COUNT = 6
HISTORY_BUCKETS = ("Today", "Yesterday", "Earlier")

# Neutral Material avatars keep the chat focused and avoid loud emoji badges.
USER_ICON = ":material/account_circle:"
ASSISTANT_ICON = ":material/auto_awesome:"
ANSWER_HEADER = "Verified Findings"
EVIDENCE_HEADER = "Evidence Analysis"
LOCAL_SOURCES_HEADER = "📄 Local"
WEB_SOURCES_HEADER = "🌍 Web"
CHAT_PLACEHOLDER_EXAMPLES = (
    "Ask about your documents...",
    "Search local files...",
    "Ask a research question...",
    "Compare local and web evidence...",
    "Explain a concept...",
    "Summarise uploaded documents...",
)
SEARCH_MODE_PLACEHOLDER_ICONS = {
    "Auto": "🌐",
    "Local Only": "📄",
    "Local + AI": "🧠",
    "Local + AI + Web": "🌐",
    "Web Only": "🌍",
    "Research Mode": "🔎",
}
SEARCH_MODE_DISPLAY_LABELS = {
    "Auto": "Auto",
    "Local Only": "Local",
    "Local + AI": "Local + AI",
    "Local + AI + Web": "Hybrid",
    "Web Only": "Web",
    "Research Mode": "Research",
    "local_only": "Local",
    "local_ai": "Local + AI",
    "local_ai_web": "Hybrid",
    "web_only": "Web",
    "research": "Research",
    "auto": "Auto",
}

# Session-state key that holds the active background generation dict.
_GEN_STATE_KEY = "_verilume_generation"


def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "generating" not in st.session_state:
        st.session_state.generating = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "regenerate_requested" not in st.session_state:
        st.session_state.regenerate_requested = False
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = ConversationState()
    if "chat_placeholder_example" not in st.session_state:
        st.session_state.chat_placeholder_example = random.choice(CHAT_PLACEHOLDER_EXAMPLES)


def render_chat(
    settings: AppSettings,
    _suggested_prompts: object | None = None,
) -> None:
    init_chat_state()

    regenerate_prompt = None
    regenerate_focus = None
    if st.session_state.regenerate_requested:
        st.session_state.regenerate_requested = False
        regenerate_prompt, regenerate_focus, st.session_state.messages = _regeneration_plan(
            st.session_state.messages
        )

    _render_toolbar(settings)
    _render_message_history(settings)

    # If a background generation is already running, poll it and return early.
    if st.session_state.get(_GEN_STATE_KEY) is not None:
        _poll_generation(settings)
        return

    if regenerate_prompt:
        _start_generation(settings, regenerate_prompt, focus_document=regenerate_focus)
        _poll_generation(settings)
        return

    if not st.session_state.messages:
        _render_welcome_screen()

    manual_prompt = st.chat_input(_chat_placeholder(settings))
    pending_prompt, focus_document = _consume_pending_prompt()
    prompt = pending_prompt or manual_prompt
    if not prompt:
        return
    if not pending_prompt:
        focus_document = None

    user_message: dict[str, Any] = {
        "role": "user",
        "content": prompt,
        "timestamp": _now_timestamp(),
    }
    if focus_document:
        user_message["focus_document"] = focus_document
    st.session_state.messages.append(user_message)
    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(prompt)

    _start_generation(settings, prompt, focus_document=focus_document)
    _poll_generation(settings)


def _chat_placeholder(settings: AppSettings) -> str:
    mode = str(getattr(settings, "search_mode", "Auto") or "Auto")
    icon = SEARCH_MODE_PLACEHOLDER_ICONS.get(mode, "🌐")
    label = _search_mode_display_label(mode)
    example = str(
        st.session_state.get("chat_placeholder_example")
        or random.choice(CHAT_PLACEHOLDER_EXAMPLES)
    )
    return f"{icon} {label}  {example}"


def _consume_pending_prompt() -> tuple[str | None, str | None]:
    """Pop the queued prompt plus the document it was derived from, if any."""
    prompt = st.session_state.pop("pending_prompt", None)
    focus_document = st.session_state.pop("pending_prompt_document", None)
    if prompt is None:
        return None, None
    prompt_text = str(prompt).strip()
    if not prompt_text:
        return None, None
    focus_text = str(focus_document).strip() if focus_document else None
    return prompt_text, focus_text or None


def _start_generation(
    settings: AppSettings, prompt: str, focus_document: str | None = None
) -> None:
    """Kick off RAG in a background thread and store the handle in session state."""
    history = _history_from_messages(st.session_state.messages[:-1])
    conv_state = st.session_state.conversation_state
    stop_event = threading.Event()
    # Capture the bound predicate once so its identity is stable: the worker's
    # cleanup must only clear the shared stop hook if it is still *this* request's
    # hook (a newer generation may have replaced it while we were abandoned).
    stop_fn = stop_event.is_set
    request_id = uuid4().hex
    result_box: dict[str, Any] = {
        "done": False,
        "stage": "Searching local evidence...",
        "request_id": request_id,
    }

    # Give the generator direct access to the stop signal so it can abort
    # in-flight Ollama streaming calls between tokens instead of blocking
    # for the full response before the next cooperative checkpoint fires.
    service = get_rag_service(settings)
    _generator = getattr(service, "generator", None)
    if _generator is not None and hasattr(_generator, "_should_stop"):
        _generator._should_stop = stop_fn

    def worker() -> None:
        try:
            resp = service.ask(
                prompt,
                history,
                conversation_state=conv_state,
                should_stop=stop_fn,
                on_stage=lambda label: result_box.update({"stage": label}),
                focus_document=focus_document,
            )
            result_box["response"] = resp
        except GenerationStopped:
            result_box["stopped"] = True
        except TimeoutError:
            LOGGER.exception("Chat generation timed out.")
            result_box["error"] = (
                "The model took too long to respond. Try again or switch to a faster model."
            )
        except (ValueError, KeyError) as exc:
            LOGGER.exception("Chat generation returned an invalid response.")
            result_box["error"] = (
                f"The model returned an unexpected response ({type(exc).__name__}). Try again."
            )
        except ConnectionError:
            LOGGER.exception("Chat generation lost network connection.")
            result_box["error"] = (
                "Connection error while reaching the model. Check your network or Ollama status."
            )
        except Exception:
            LOGGER.exception("Chat generation failed.")
            result_box["error"] = (
                "Something went wrong generating the answer. Check the terminal logs and try again."
            )
        finally:
            result_box["done"] = True
            # Only clear the shared stop hook if it is still ours. If the user
            # stopped this request and started a new one, that newer generation
            # already installed its own predicate and must not be clobbered.
            if _generator is not None and getattr(_generator, "_should_stop", None) is stop_fn:
                _generator._should_stop = None

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    st.session_state[_GEN_STATE_KEY] = {
        "thread": thread,
        "stop_event": stop_event,
        "result": result_box,
        "prompt": prompt,
        "request_id": request_id,
    }
    st.session_state.generating = True
    st.session_state.stop_requested = False


@st.fragment(run_every=0.4)
def _poll_generation(settings: AppSettings) -> None:
    """Fragment wrapper: only this subtree re-renders on each 0.4s tick.

    A full st.rerun() loop here would re-execute the whole script every tick
    and could leave a stale chat_message block visible alongside the freshly
    rendered one. The actual logic lives in _poll_generation_impl so it can be
    unit-tested without a live Streamlit script context (fragments no-op when
    called outside one).
    """
    _poll_generation_impl(settings)


def _poll_generation_impl(settings: AppSettings) -> None:
    gen = st.session_state.get(_GEN_STATE_KEY)
    if gen is None:
        return

    result_box: dict[str, Any] = gen["result"]
    stop_event: threading.Event = gen["stop_event"]

    if not result_box["done"]:
        # The Stop button lives inside this fragment because the fragment (unlike
        # the main-script toolbar) re-executes on every run_every tick, so the
        # button is genuinely live while a response is generating. Clicking it
        # triggers a fragment-scoped rerun — Streamlit's guaranteed-immediate
        # path — so the stop fires without racing the main-script scheduler.
        # (A response that finishes before the first ~0.4s tick simply finalizes
        # with no Stop shown — by then there is nothing left to stop.)
        stop_clicked = st.button(
            "⏹ Stop generating",
            key="_gen_stop",
            use_container_width=True,
            type="primary",
            help="Stop the current response",
        )
        # A stop can arrive from that button or from a queued stop_requested flag.
        # Either way, signal the worker AND finalize the UI right now — do not
        # wait for result_box["done"]. The background thread may still be blocked
        # inside a non-cooperative network/model call; that abandoned worker winds
        # itself down on its own cooperative checkpoints, and its late result is
        # discarded because we drop the generation handle here.
        if stop_clicked or st.session_state.stop_requested:
            stop_event.set()
            _finalize_stopped_generation()
            return
        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            st.markdown(
                _loading_stage_html(result_box.get("stage", "Searching local evidence...")),
                unsafe_allow_html=True,
            )
        return

    # Generation finished — clean up state flags.
    st.session_state.generating = False
    st.session_state.stop_requested = False
    del st.session_state[_GEN_STATE_KEY]

    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        if result_box.get("stopped"):
            message = "Generation stopped by user."
            msg_dict = {"role": "assistant", "content": message, "timestamp": _now_timestamp()}
            _render_assistant_meta(msg_dict, len(st.session_state.messages))
            st.markdown(message)
            st.session_state.messages.append(msg_dict)
        elif result_box.get("error"):
            message = str(result_box["error"])
            msg_dict = {"role": "assistant", "content": message, "timestamp": _now_timestamp()}
            _render_assistant_meta(msg_dict, len(st.session_state.messages))
            st.warning(message)
            st.session_state.messages.append(msg_dict)
        else:
            response = result_box["response"]
            assistant_message = {
                "role": "assistant",
                "content": response.answer,
                "response": response,
                "timestamp": _now_timestamp(),
            }
            if response.conversation_state is not None:
                st.session_state.conversation_state = response.conversation_state
            _render_assistant_meta(assistant_message, len(st.session_state.messages))
            _render_answer(response, f"live-{len(st.session_state.messages)}")
            _render_sources(response, settings, key_prefix=f"live-{len(st.session_state.messages)}")
            st.session_state.messages.append(assistant_message)
            _trim_history(settings.max_chat_messages)

    # Exit fragment-only rerun scope so the toolbar (Stop/Regenerate enablement)
    # and message history settle into their normal, non-generating state.
    st.rerun()


def _finalize_stopped_generation() -> None:
    """Drop the active generation handle and record a stopped message now.

    Called the instant a stop is requested, without waiting for the worker to
    observe its cooperative checkpoint. Dropping ``_GEN_STATE_KEY`` means the
    worker's eventual result (success or otherwise) is never read or rendered,
    so no stale answer can appear after the user has stopped.
    """
    st.session_state.generating = False
    st.session_state.stop_requested = False
    if _GEN_STATE_KEY in st.session_state:
        del st.session_state[_GEN_STATE_KEY]

    message = "Generation stopped by user."
    msg_dict = {"role": "assistant", "content": message, "timestamp": _now_timestamp()}
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        _render_assistant_meta(msg_dict, len(st.session_state.messages))
        st.markdown(message)
    st.session_state.messages.append(msg_dict)
    st.rerun()


@st.cache_data(show_spinner=False)
def _cached_chat_pdf(messages: tuple[Any, ...], title: str) -> bytes | None:
    """Generate PDF only when message content changes; returns None on failure."""
    try:
        return chat_to_pdf(list(messages), title)
    except Exception:
        LOGGER.debug("PDF export unavailable.", exc_info=True)
        return None


def _render_toolbar(settings: AppSettings) -> None:
    can_regenerate = _latest_user_index(st.session_state.messages) is not None
    # No Stop button here: this toolbar is rendered in the main script, which does
    # not re-execute during the polling fragment's run_every ticks, so a toolbar
    # widget's disabled state would be frozen from before generation started and
    # could never become clickable. The live Stop lives inside the polling
    # fragment (_poll_generation_impl), which re-renders every tick.
    col_b, col_c, col_d, col_e = st.columns([1, 1, 1, 1])
    with col_b:
        if st.button(
            "\u21ba Regenerate",
            disabled=st.session_state.generating or not can_regenerate,
            use_container_width=True,
            help="Re-run the last query",
        ):
            st.session_state.regenerate_requested = True
            st.rerun()
    with col_c:
        if st.button(
            "\u2715 Clear",
            use_container_width=True,
            help="Clear the chat history",
        ):
            st.session_state.messages = []
            st.session_state.conversation_state = ConversationState()
            st.rerun()
    markdown = chat_to_markdown(st.session_state.messages, settings.app_title)
    with col_d:
        st.download_button(
            "\u2193 Markdown",
            data=markdown,
            file_name="verilume-chat.md",
            mime="text/markdown",
            use_container_width=True,
            help="Download chat as Markdown",
        )
    with col_e:
        pdf = _cached_chat_pdf(tuple(st.session_state.messages), settings.app_title)
        if pdf is not None:
            st.download_button(
                "\u2193 PDF",
                data=pdf,
                file_name="verilume-chat.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download chat as PDF",
            )
        else:
            st.download_button(
                "\u2193 PDF",
                data=b"",
                file_name="verilume-chat.pdf",
                disabled=True,
                use_container_width=True,
                help="PDF export unavailable",
            )


def _render_welcome_screen() -> None:
    st.markdown(
        """
<div class="veri-welcome">
  <div class="veri-welcome-kicker">Welcome</div>
  <div class="veri-welcome-title">Search documents, research sources, and compare evidence.</div>
  <div class="veri-welcome-grid">
    <div class="veri-welcome-cell">
      <div class="veri-welcome-cell-title">📄 Search documents</div>
      <div class="veri-welcome-cell-desc">Find local facts and citations.</div>
    </div>
    <div class="veri-welcome-cell">
      <div class="veri-welcome-cell-title">📚 Summarise files</div>
      <div class="veri-welcome-cell-desc">Turn long PDFs into clear briefs.</div>
    </div>
    <div class="veri-welcome-cell">
      <div class="veri-welcome-cell-title">⚖ Compare evidence</div>
      <div class="veri-welcome-cell-desc">Separate local, AI, and web support.</div>
    </div>
    <div class="veri-welcome-cell">
      <div class="veri-welcome-cell-title">🌍 Current facts</div>
      <div class="veri-welcome-cell-desc">Use web sources when enabled.</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _loading_stage_html(label: str) -> str:
    current = _loading_stage_index(label)
    steps = ("Searching Local", "Searching Web", "Ranking", "Verifying", "Writing")
    rendered_steps = "".join(
        (
            f'<div class="veri-loading-step {"active" if index == current else "done" if index < current else ""}">'
            f"<span>{escape(step)}</span><i></i>"
            "</div>"
        )
        for index, step in enumerate(steps)
    )
    return f"""
<div class="veri-loading-panel">
  <div class="veri-loading-label">{escape(label)}</div>
  <div class="veri-loading-steps">{rendered_steps}</div>
</div>
    """


def _loading_stage_index(label: str) -> int:
    normalized = (label or "").lower()
    if "web" in normalized:
        return 1
    if "rank" in normalized:
        return 2
    if "verif" in normalized or "evidence" in normalized:
        return 3
    if (
        "generat" in normalized
        or "synthesis" in normalized
        or "answer" in normalized
        or "writ" in normalized
    ):
        return 4
    return 0


def _render_message(
    message: dict[str, Any],
    settings: AppSettings,
    index: int,
    source_display: str = "expander",
) -> None:
    role = message.get("role", "assistant")
    avatar = USER_ICON if role == "user" else ASSISTANT_ICON
    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            _render_assistant_meta(message, index)
        response = message.get("response")
        if isinstance(response, RAGResponse):
            _render_answer(response, f"history-{index}")
            _render_sources(response, settings, display=source_display, key_prefix=f"history-{index}")
        else:
            st.markdown(str(message.get("content", "")))


def _render_answer(response: RAGResponse, key_prefix: str) -> None:
    recommendation = _recommendation_for_response(response, key_prefix)
    if recommendation is None:
        # Keyed container emits an `.st-key-...` wrapper class so the answer
        # body can be visually enlarged (see .veri-answer-body in styles.py)
        # without affecting the smaller evidence/diagnostic text below it.
        with st.container(key=f"veri-answer-body-{key_prefix}"):
            st.markdown(_display_answer(response))
        _render_evidence_summary(response)
        return
    _render_recommendation(**recommendation)


def _render_sources(
    response: RAGResponse,
    settings: AppSettings,
    display: str = "expander",
    key_prefix: str = "src",
) -> None:
    if display == "inline":
        _render_sources_inline(response, settings)
        return
    _render_sources_expanded(response, settings, key_prefix)


def _render_sources_expanded(
    response: RAGResponse, settings: AppSettings, key_prefix: str = "src"
) -> None:
    _render_evidence_details(response)
    _render_benchmark_report(response, key_prefix)
    _render_evidence_comparison(response)
    _render_specialized_evidence_panels(response)
    if settings.show_local_sources and response.local_sources:
        st.markdown(
            f'<div class="veri-source-section veri-source-section-local">{LOCAL_SOURCES_HEADER}</div>',
            unsafe_allow_html=True,
        )
        _render_local_sources_table(response)
    if response.web_sources:
        with st.expander(f"{WEB_SOURCES_HEADER} ({len(response.web_sources)})", expanded=True):
            _render_web_source_groups(response)


def _render_sources_inline(response: RAGResponse, settings: AppSettings) -> None:
    _render_evidence_details_inline(response)
    _render_benchmark_report_inline(response)
    _render_evidence_comparison_inline(response)
    _render_specialized_evidence_panels(response)
    if settings.show_local_sources and response.local_sources:
        with st.container():
            st.markdown(
                f'<div class="veri-inline-source-heading veri-inline-source-heading-local">{LOCAL_SOURCES_HEADER}</div>',
                unsafe_allow_html=True,
            )
            _render_local_sources_table(response)
    if response.web_sources:
        with st.container():
            st.markdown(
                f'<div class="veri-inline-source-heading veri-inline-source-heading-web">{WEB_SOURCES_HEADER}</div>',
                unsafe_allow_html=True,
            )
            _render_web_source_groups(response)


def _render_evidence_badges(response: RAGResponse) -> None:
    badges = _evidence_badges(response)
    if not badges:
        return

    rendered = "".join(f"<span>{escape(label)}</span>" for label in badges)
    st.markdown(
        f'<div class="veri-evidence-badges">{rendered}</div>',
        unsafe_allow_html=True,
    )


def _render_evidence_summary(response: RAGResponse) -> None:
    origin, confidence, source_type = _answer_origin(response)
    diagnostics = response.diagnostics or {}
    source_count = _supporting_source_count(response)
    source_count_label = _supporting_source_count_label(source_count)
    badges = [origin, f"{_confidence_dot(confidence)} Confidence: {confidence}", source_type]
    if source_count_label:
        badges.append(source_count_label)
    badges.extend(_evidence_badges(response)[1:])
    rendered_badges = "".join(f"<span>{escape(label)}</span>" for label in badges if label)
    strength_rows = _source_strength_rows(response)
    rendered_strength = "".join(
        (
            '<div class="veri-source-strength-row">'
            '<div class="veri-source-strength-head">'
            f'<span class="veri-source-strength-label">{escape(label.upper())} SUPPORT</span>'
            f'<span class="veri-source-strength-grade veri-source-strength-grade-{kind}">'
            f"{escape(_strength_grade(score))}</span>"
            "</div>"
            '<div class="veri-source-strength-meter">'
            '<span class="veri-source-strength-track">'
            f'<span class="veri-source-strength-fill veri-source-strength-{kind}" style="width:{score}%"></span>'
            "</span>"
            f'<span class="veri-source-strength-value">{score}%</span>'
            "</div>"
            "</div>"
        )
        for label, score, kind in strength_rows
    )
    strength_block = (
        f'<div class="veri-source-strengths">{rendered_strength}</div>'
        if rendered_strength
        else ""
    )
    reason_title, win_reasons = _winner_reasons(diagnostics)
    reasons_block = ""
    if win_reasons:
        rendered_reasons = "".join(f"<li>✓ {escape(reason)}</li>" for reason in win_reasons)
        reasons_block = (
            '<div class="veri-evidence-reasons">'
            f"<strong>{escape(reason_title)}</strong>"
            f"<ul>{rendered_reasons}</ul>"
            "</div>"
        )
    search_mode = _summary_search_mode(diagnostics)
    searched = _summary_source_list(diagnostics.get("sources_searched"))
    used = _summary_source_list(diagnostics.get("sources_used"))
    winner = _friendly_token(str(diagnostics.get("evidence_winner") or ""))
    claim_support = _claim_support_summary(diagnostics)
    summary_rows = "".join(
        (
            '<div class="veri-evidence-summary-row">'
            f'<span>{escape(icon)} {escape(label)}</span>'
            f'<strong>{escape(value)}</strong>'
            "</div>"
        )
        for icon, label, value in (
            ("🔎", "Search mode", search_mode),
            ("📚", "Sources searched", searched),
            ("✅", "Sources used", used),
            ("🏆", "Winner", winner),
            ("🤝", "Claim support", claim_support),
        )
        if value
    )
    rows_block = f'<div class="veri-evidence-summary-rows">{summary_rows}</div>' if summary_rows else ""
    st.markdown(
        f"""
<div class="veri-evidence-summary">
  <div class="veri-evidence-summary-title">Evidence Summary</div>
  <div class="veri-evidence-summary-badges">{rendered_badges}</div>
  {rows_block}
  {reasons_block}
  {strength_block}
</div>
        """,
        unsafe_allow_html=True,
    )


def _strength_grade(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def _confidence_dot(confidence: str) -> str:
    level = str(confidence or "").strip().lower()
    if level.startswith("high"):
        return "🟢"
    if level.startswith("med"):
        return "🟡"
    if level.startswith("low"):
        return "🔴"
    return "⚪"


def _winner_reasons(diagnostics: dict[str, Any]) -> tuple[str, list[str]]:
    winner = str(diagnostics.get("evidence_winner") or "").lower()
    if winner == "web":
        return "Why Web Won", _summary_value_list(diagnostics.get("web_win_reasons"))
    if winner == "hybrid":
        reasons = _summary_value_list(diagnostics.get("hybrid_win_reasons"))
        if not reasons:
            reasons = _summary_value_list(diagnostics.get("local_win_reasons"))
        return "Why Hybrid Won", reasons
    return "Why Local Won", _summary_value_list(diagnostics.get("local_win_reasons"))


def _claim_support_summary(diagnostics: dict[str, Any]) -> str:
    summary = diagnostics.get("claim_support_summary") or {}
    if not isinstance(summary, dict):
        return ""
    supported = int(summary.get("supported") or 0)
    weak = int(summary.get("weakly_supported") or 0)
    unsupported = int(summary.get("unsupported") or 0)
    total = supported + weak + unsupported
    if total <= 0:
        return ""
    if weak or unsupported:
        return f"{supported}/{total} supported, {weak} weak, {unsupported} unsupported"
    return f"{supported}/{total} supported"


def _supporting_source_count_label(source_count: int) -> str:
    if not source_count:
        return ""
    noun = "source" if source_count == 1 else "sources"
    return f"{source_count} supporting {noun}"


def _summary_search_mode(diagnostics: dict[str, Any]) -> str:
    policy = diagnostics.get("search_policy") or {}
    if isinstance(policy, dict) and policy.get("mode"):
        return _search_mode_display_label(str(policy.get("mode")))
    return _search_mode_display_label(
        str(diagnostics.get("search_mode_key") or diagnostics.get("search_mode") or "")
    )


def _search_mode_display_label(mode: str) -> str:
    text = str(mode or "Auto").strip()
    if text in SEARCH_MODE_DISPLAY_LABELS:
        return SEARCH_MODE_DISPLAY_LABELS[text]
    key = text.lower().replace("-", "_").replace(" ", "_")
    return SEARCH_MODE_DISPLAY_LABELS.get(key, _friendly_token(text))


def _summary_source_list(value: Any) -> str:
    labels = []
    for item in _summary_value_list(value):
        text = str(item or "").strip()
        if not text:
            continue
        labels.append({"ai": "AI", "model_knowledge": "AI"}.get(text, _friendly_token(text)))
    return ", ".join(labels)


def _summary_value_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def _render_evidence_details(response: RAGResponse) -> None:
    curated = _curated_evidence_html(response)
    debug_rows = _evidence_detail_rows(response)
    if not curated and not debug_rows:
        return

    if curated:
        with st.expander(EVIDENCE_HEADER, expanded=False):
            st.markdown(curated, unsafe_allow_html=True)
    # Sibling (not nested) expander — Streamlit forbids expanders inside
    # expanders — keeps the raw diagnostics available on demand without
    # cluttering the reader-facing Evidence Analysis card.
    if debug_rows:
        with st.expander("Debug details", expanded=False):
            for label, value in debug_rows:
                st.markdown(f"**{label}:** {value}")


def _curated_evidence_html(response: RAGResponse) -> str:
    """Reader-facing Evidence Analysis: verification, agreement, freshness, knowledge."""
    diagnostics = response.diagnostics or {}
    cards: list[tuple[str, str]] = []
    verification = _verification_display(diagnostics)
    if verification:
        cards.append(("Verification", verification))
    agreement = str(diagnostics.get("source_agreement") or "").strip()
    if agreement:
        cards.append(("Agreement", agreement.title()))
    cards.append(("Freshness", _freshness_display(diagnostics)))
    rendered_cards = "".join(
        (
            '<div class="veri-evidence-summary-row">'
            f"<span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
        )
        for label, value in cards
        if value
    )
    used_local, used_model, used_web = _response_evidence_usage(response)
    knowledge = (("Local", used_local), ("AI", used_model), ("Web", used_web))
    chips = "".join(
        (
            f'<span class="veri-knowledge-chip {"on" if on else "off"}">'
            f'{"✓" if on else "✗"} {escape(name)}</span>'
        )
        for name, on in knowledge
    )
    if not rendered_cards and not chips:
        return ""
    knowledge_block = (
        '<div class="veri-knowledge-used">'
        '<span class="veri-knowledge-label">Knowledge used</span>'
        f'<div class="veri-knowledge-chips">{chips}</div>'
        "</div>"
    )
    return (
        f'<div class="veri-evidence-summary-rows veri-evidence-analysis-grid">{rendered_cards}</div>'
        f"{knowledge_block}"
    )


def _verification_display(diagnostics: dict[str, Any]) -> str:
    status = str(diagnostics.get("answer_verification_status") or "").strip()
    if not status:
        return ""
    friendly = _friendly_token(status)
    low = status.lower()
    if any(marker in low for marker in ("verified", "supported", "pass", "grounded")):
        return f"✓ {friendly}"
    if any(marker in low for marker in ("fail", "unsupported", "unverified", "error")):
        return f"⚠ {friendly}"
    return friendly


def _freshness_display(diagnostics: dict[str, Any]) -> str:
    if diagnostics.get("local_is_older_than_web") is True:
        return "Newer web available"
    if diagnostics.get("requires_date_reconciliation") is True:
        return "Checked"
    note = str(diagnostics.get("freshness_note") or "").strip()
    if note:
        return _friendly_token(note)
    return "Current"


def _render_evidence_details_inline(response: RAGResponse) -> None:
    rows = _evidence_detail_rows(response)
    if not rows:
        return

    summary = " · ".join(f"{label}: {value}" for label, value in rows[:3])
    st.caption(f"{EVIDENCE_HEADER} — {summary}")


def _render_evidence_comparison(response: RAGResponse) -> None:
    comparisons = _claim_comparisons(response)
    if not comparisons:
        return

    with st.expander("Evidence Comparison", expanded=False):
        for index, comparison in enumerate(comparisons, start=1):
            claim = comparison.get("claim") or {}
            st.markdown(f"**Claim {index}:** {claim.get('text', '')}")
            cols = st.columns(3)
            for col, source_type in zip(cols, ("local", "web", "ai"), strict=False):
                with col:
                    st.markdown(_comparison_stream_block(comparison, source_type))
            decision = str(comparison.get("decision") or "").strip()
            winner = str(comparison.get("winning_source_type") or "").strip()
            conflict = bool(comparison.get("conflict_detected"))
            suffix = " Conflict detected." if conflict else ""
            st.caption(f"Decision: {decision} Winner: {_friendly_token(winner)}.{suffix}")


def _render_benchmark_report(response: RAGResponse, key_prefix: str = "src") -> None:
    report = _benchmark_report(response)
    if not report or not _benchmark_rows(report):
        return

    results = [r for r in report.get("results", []) if isinstance(r, dict)]
    best_label = _benchmark_display_label(
        str(report.get("best_mode_label") or report.get("best_mode") or "")
    )
    caption = f"{len(results)} modes compared"
    if best_label:
        caption += f" · best: {best_label}"
    st.markdown(
        f'<div class="veri-benchmark-teaser">📊 Benchmark — {escape(caption)}</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "📊 View Benchmark Report",
        key=f"bench-{key_prefix}",
        use_container_width=True,
        help="Compare Local, AI, Web, and Full retrieval modes",
    ):
        try:
            _benchmark_dialog(report)
        except Exception:
            # Dialogs are disallowed in a few nested contexts; if so, fall back
            # to inline rendering so the report is never lost.
            LOGGER.debug("Benchmark dialog unavailable; rendering inline.", exc_info=True)
            with st.expander("Benchmark Results", expanded=True):
                _render_benchmark_body(report)


def _benchmark_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        faithfulness = result.get("faithfulness_score")
        rows.append(
            {
                "Mode": _benchmark_display_label(str(result.get("mode") or "")),
                "Confidence": result.get("confidence") or "",
                "Sources": int(result.get("source_count") or 0),
                "Local": int(result.get("local_source_count") or 0),
                "Web": int(result.get("web_source_count") or 0),
                "Time": f"{float(result.get('latency_seconds') or 0.0):.2f}s",
                "Faithfulness": (
                    f"{int(round(float(faithfulness) * 100))}%"
                    if faithfulness is not None
                    else "N/A"
                ),
            }
        )
    return rows


@st.dialog("Benchmark Report", width="large")
def _benchmark_dialog(report: dict[str, Any]) -> None:
    _render_benchmark_body(report)


def _render_benchmark_body(report: dict[str, Any]) -> None:
    rows = _benchmark_rows(report)
    if not rows:
        st.caption("No benchmark results available.")
        return
    table_height = min(260, 44 + (len(rows) + 1) * 36)
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        height=table_height,
    )
    best_label = _benchmark_display_label(
        str(report.get("best_mode_label") or report.get("best_mode") or "")
    )
    if best_label:
        st.caption(f"Best mode: {best_label}")
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        mode = _benchmark_display_label(str(result.get("mode") or ""))
        answer = str(result.get("answer") or "").strip()
        st.markdown(f"**{mode} answer**")
        st.markdown(answer or "_No answer returned._")


def _render_benchmark_report_inline(response: RAGResponse) -> None:
    report = _benchmark_report(response)
    if not report:
        return
    results = report.get("results") or []
    best_label = _benchmark_display_label(
        str(report.get("best_mode_label") or report.get("best_mode") or "")
    )
    st.caption(f"Benchmark Results — {len(results)} modes compared; best: {best_label or 'N/A'}")


def _benchmark_report(response: RAGResponse) -> dict[str, Any]:
    report = (response.diagnostics or {}).get("benchmark_report") or {}
    return report if isinstance(report, dict) else {}


def _benchmark_display_label(mode: str) -> str:
    labels = {
        "full": "Full",
        "Full": "Full",
        "local_only": "Local",
        "Local Only": "Local",
        "ai_only": "AI",
        "AI Only": "AI",
        "web_only": "Web",
        "Web Only": "Web",
    }
    text = str(mode or "").strip()
    return labels.get(text, text.replace("_", " ").title())


def _render_evidence_comparison_inline(response: RAGResponse) -> None:
    comparisons = _claim_comparisons(response)
    if not comparisons:
        return

    supported = sum(
        1
        for comparison in comparisons
        if str(comparison.get("winning_source_type") or "") not in {"", "unsupported"}
    )
    st.caption(f"Evidence Comparison — {supported}/{len(comparisons)} claims supported")


def _claim_comparisons(response: RAGResponse) -> list[dict[str, Any]]:
    comparisons = (response.diagnostics or {}).get("claim_comparisons") or []
    if not isinstance(comparisons, list):
        return []
    return [item for item in comparisons if isinstance(item, dict)]


def _comparison_stream_block(comparison: dict[str, Any], source_type: str) -> str:
    key = f"{source_type}_support"
    supports = comparison.get(key) or []
    title = {"local": "Local", "web": "Web", "ai": "AI"}[source_type]
    if not supports:
        return f"**{title}**  \nNot found"
    best = supports[0]
    status = _friendly_token(str(best.get("support") or "unclear"))
    confidence = _score_to_percent(float(best.get("confidence") or 0.0))
    label = str(best.get("source_label") or title)
    snippet = str(best.get("snippet") or "").strip()
    snippet_line = f"  \n{escape(snippet[:180])}" if snippet else ""
    return f"**{title}**  \n{status} ({confidence}%)  \n`{escape(label)}`{snippet_line}"


def _evidence_badges(response: RAGResponse) -> list[str]:
    confidence = (response.confidence or "").strip()
    diagnostics = response.diagnostics or {}
    badges = [_confidence_badge(confidence)]

    agreement = str(diagnostics.get("source_agreement") or "").strip()
    if agreement:
        badges.append(f"Agreement: {agreement.title()}")

    if diagnostics.get("local_is_older_than_web") is True:
        badges.append("Freshness: newer web evidence")
    elif diagnostics.get("requires_date_reconciliation") is True:
        badges.append("Freshness checked")

    winner = str(diagnostics.get("evidence_winner") or "").replace("_", " ").strip()
    if winner:
        badges.append(f"Winner: {winner.title()}")

    return [badge for badge in badges if badge]


def _confidence_badge(confidence: str) -> str:
    labels = {
        "current-information": "Evidence: Current verified",
        "high": "Evidence: High confidence",
        "medium": "Evidence: Medium confidence",
        "low": "Evidence: Low confidence",
        "local-grounded": "Evidence: Local grounded",
        "local-web-assisted": "Evidence: Local + web",
        "web-assisted": "Evidence: Web assisted",
        "model-only": "Evidence: AI knowledge",
        "model-selection-warning": "Evidence: Model unavailable",
        "needs-token": "Evidence: Token needed",
        "generation-error": "Evidence: Generation error",
        "": "Evidence: Unknown",
    }
    if confidence in labels:
        return labels[confidence]
    # Strip internal underscores/dashes but never expose raw enum tokens.
    clean = re.sub(r"[-_]+", " ", confidence).strip().title()
    # Reject anything that looks like an unhandled internal identifier.
    if re.search(r"[_A-Z]{2,}", confidence):
        return "Evidence: Unavailable"
    return f"Evidence: {clean}" if clean else "Evidence: Unknown"


def _evidence_detail_rows(response: RAGResponse) -> list[tuple[str, str]]:
    diagnostics = response.diagnostics or {}
    fields = (
        ("Evidence note", _friendly_evidence_note(diagnostics.get("evidence_note"))),
        ("Focused document", diagnostics.get("focus_document")),
        ("Search mode", _summary_search_mode(diagnostics)),
        ("Sources searched", _summary_source_list(diagnostics.get("sources_searched"))),
        ("Sources used", _summary_source_list(diagnostics.get("sources_used"))),
        ("Freshness note", diagnostics.get("freshness_note")),
        ("Local older than web", diagnostics.get("local_is_older_than_web")),
        ("Source agreement", diagnostics.get("source_agreement")),
        ("Evidence winner", diagnostics.get("evidence_winner")),
        ("Why local won", diagnostics.get("local_win_reasons")),
        ("Evidence streams", diagnostics.get("evidence_streams")),
        ("Used model knowledge", diagnostics.get("used_model_knowledge")),
        ("Model knowledge available", diagnostics.get("model_knowledge_available")),
        ("Web enabled", diagnostics.get("web_enabled")),
        ("Query type", diagnostics.get("query_type")),
        ("Web note", diagnostics.get("web_note")),
        ("Web error", diagnostics.get("web_error")),
        ("Verification", diagnostics.get("answer_verification_status")),
    )

    rows: list[tuple[str, str]] = []
    for label, value in fields:
        formatted = _format_diagnostic_value(value)
        if formatted:
            rows.append((label, formatted))
    return rows


def _format_diagnostic_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(_friendly_token(str(item)) for item in value if item)
    return _friendly_token(str(value))


def _friendly_token(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    return text.replace("_", " ").title() if re.fullmatch(r"[a-z_]+", text) else text


def _render_specialized_evidence_panels(response: RAGResponse) -> None:
    specialized = [
        source
        for source in response.local_sources
        if (source.metadata or {}).get("content_type") in {"formula", "structured_field", "ocr_block"}
    ]
    if not specialized:
        return
    grouped: dict[str, list[Any]] = {}
    for source in specialized:
        grouped.setdefault(str((source.metadata or {}).get("content_type")), []).append(source)
    labels = {
        "formula": "Formula Evidence",
        "structured_field": "Structured OCR Evidence",
        "ocr_block": "OCR Evidence",
    }
    for content_type, sources in grouped.items():
        rendered = "".join(_specialized_evidence_card(source) for source in sources)
        st.markdown(
            f"""
<div class="veri-specialized-panel">
  <div class="veri-specialized-title">{escape(labels.get(content_type, "Specialized Evidence"))}</div>
  <div class="veri-source-card-grid">{rendered}</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def _specialized_evidence_card(source: Any) -> str:
    metadata = source.metadata or {}
    content_type = metadata.get("content_type")
    page = f"Page {source.page}" if source.page else "Page not available"
    if content_type == "formula":
        title = metadata.get("formula_type") or "formula"
        rows = (
            ("Document", source.document),
            ("Page", page),
            ("Formula", metadata.get("repaired_formula") or metadata.get("raw_formula") or source.text),
            ("Variables", _format_diagnostic_value(metadata.get("formula_variables"))),
            ("Surrounding text", metadata.get("surrounding_text")),
            ("Confidence", metadata.get("formula_confidence")),
        )
    elif content_type == "structured_field":
        title = metadata.get("canonical_name") or "structured field"
        rows = (
            ("Document", source.document),
            ("Page", page),
            ("Document type", metadata.get("document_type")),
            ("Field", metadata.get("canonical_name")),
            ("Value", _source_value_line(source.text)),
            ("Raw label", metadata.get("raw_label") or "detected pattern"),
            ("Confidence", metadata.get("structured_confidence")),
        )
    else:
        title = "OCR block"
        rows = (
            ("Document", source.document),
            ("Page", page),
            ("Text block", source.text),
            ("Confidence", metadata.get("ocr_confidence")),
        )
    rendered_rows = "".join(
        f"<p><strong>{escape(label)}:</strong> {escape(_format_diagnostic_value(value))}</p>"
        for label, value in rows
        if _format_diagnostic_value(value)
    )
    return f"""
<div class="veri-specialized-card">
  <div class="veri-source-card-top">
    <div class="veri-source-card-title"><span>✦</span>{escape(_friendly_token(str(title)))}</div>
    <div class="veri-source-card-badge">{escape(source.label)}</div>
  </div>
  <div class="veri-specialized-body">{rendered_rows}</div>
</div>
    """


def _source_value_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.lower().startswith("value:"):
            return line.split(":", maxsplit=1)[1].strip()
    return text or ""


def _render_local_sources_table(response: RAGResponse) -> None:
    cards = []
    for rank, row in enumerate(local_source_rows(response.local_sources), start=1):
        cards.append(
            """
<div class="veri-source-card">
  <div class="veri-source-card-top">
    <div class="veri-source-card-title"><span class="veri-source-card-rank">{rank}</span>{document}</div>
    <div class="veri-source-card-badge">{citation}</div>
  </div>
  <div class="veri-source-card-meta">
    <span>{page}</span>
    <span>Confidence: {confidence}</span>
    <span>Score: {score}</span>
  </div>
  <div class="veri-source-card-preview">{preview}</div>
</div>
            """.format(
                rank=rank,
                citation=escape(str(row.get("Citation", ""))),
                confidence=escape(str(row.get("Confidence", ""))),
                document=escape(str(row.get("Document", ""))),
                page=escape(str(row.get("Page", ""))),
                preview=escape(str(row.get("Preview", ""))),
                score=escape(str(row.get("Score", ""))),
            )
        )
    st.markdown(
        f'<div class="veri-source-card-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_web_source_groups(response: RAGResponse) -> None:
    for group_title, source_type, grouped_sources in _group_web_source_rows(
        web_source_rows(response.web_sources)
    ):
        stars = source_quality_stars(source_type)
        st.markdown(
            f'<div class="veri-source-group-heading">{escape(group_title)}'
            f'<span class="veri-source-stars" title="Source quality rating">{stars}</span></div>',
            unsafe_allow_html=True,
        )
        cards = []
        for source in grouped_sources:
            cards.append(
                _web_source_card_html(
                    badge=str(source.get("Badge") or ""),
                    citation=str(source.get("Citation") or ""),
                    confidence=str(source.get("Confidence") or ""),
                    preview=str(source.get("Preview") or ""),
                    source=str(source.get("Source") or ""),
                    url=str(source.get("URL") or ""),
                )
            )
        st.markdown(
            f'<div class="veri-source-card-grid">{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )


def _safe_href(url: str) -> str:
    """Return an HTML-escaped href only for http/https URLs; fall back to '#'."""
    try:
        parsed = urlparse(url or "")
        if parsed.scheme not in {"http", "https"}:
            return "#"
    except Exception:
        return "#"
    return escape(url, quote=True)


def _web_source_card_html(
    *,
    badge: str,
    citation: str,
    confidence: str,
    preview: str,
    source: str,
    url: str,
) -> str:
    icon = badge.split(" ", maxsplit=1)[0] if badge else "🌍"
    safe_url = _safe_href(url)
    safe_source = escape(source or url or "Web source")
    display_url = escape(url) if safe_url != "#" else ""
    return f"""
<div class="veri-source-card">
  <div class="veri-source-card-top">
    <a class="veri-source-card-title" href="{safe_url}" target="_blank" rel="noopener noreferrer"><span>{escape(icon)}</span>{safe_source}</a>
    <div class="veri-source-card-badge">{escape(citation)}</div>
  </div>
  <div class="veri-source-card-meta">
    <span>Confidence: {escape(confidence)}</span>
    <span>{display_url}</span>
  </div>
  <div class="veri-source-card-preview">{escape(preview)}</div>
</div>
    """


def _group_web_source_rows(
    rows: list[dict[str, str | float | None]],
) -> list[tuple[str, str, list[dict[str, str | float | None]]]]:
    grouped: dict[str, list[dict[str, str | float | None]]] = {}
    for row in rows:
        source_type = str(row.get("Source type") or "Web")
        grouped.setdefault(source_type, []).append(row)
    return [
        (_web_source_group_title(source_type, group), source_type, group)
        for source_type, group in grouped.items()
    ]


def _web_source_group_title(source_type: str, rows: list[dict[str, str | float | None]]) -> str:
    badge = str(rows[0].get("Badge") or source_badge(source_type))
    emoji, _, label = badge.partition(" ")
    source_label = {
        "Social media": "Social",
        "Local document": "Local",
        "AI knowledge": "AI",
        "Model knowledge": "AI",
    }.get(source_type, label or source_type)
    return f"{emoji} {source_label} Sources ({len(rows)})"


def _render_assistant_meta(message: dict[str, Any], index: int) -> None:
    col_a, col_b = st.columns([7.5, 1.3])
    with col_a:
        st.markdown(
            f'<div class="veri-answer-heading">{ANSWER_HEADER}</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        _render_copy_answer_button(str(message.get("content", "")), index)
    timestamp = _format_timestamp(message.get("timestamp"))
    if timestamp:
        st.markdown(f'<div class="veri-answer-timestamp">{timestamp}</div>', unsafe_allow_html=True)
    stats = _answer_stats_html(message.get("response"))
    if stats:
        st.markdown(stats, unsafe_allow_html=True)


def _answer_stats_html(response: Any) -> str:
    """Signature header row: time · sources · confidence, with a divider.

    Only rendered from real response data; conversational/error messages that
    carry no RAGResponse fall through to a plain body with no header chips.
    """
    if not isinstance(response, RAGResponse):
        return ""
    chips: list[str] = []
    seconds = (response.diagnostics or {}).get("response_seconds")
    try:
        if seconds is not None and float(seconds) > 0:
            chips.append(("", f"\U0001f552 {float(seconds):.1f}s"))
    except (TypeError, ValueError):
        pass
    source_count = _supporting_source_count(response)
    if source_count:
        noun = "source" if source_count == 1 else "sources"
        chips.append(("", f"\U0001f4da {source_count} {noun}"))
    _, confidence, _ = _answer_origin(response)
    if confidence and confidence != "N/A":
        chips.append(("conf", f"{_confidence_dot(confidence)} {confidence} confidence"))
    if not chips:
        return ""
    rendered = "".join(
        f'<span class="veri-answer-stat-{cls}">{escape(text)}</span>' if cls
        else f"<span>{escape(text)}</span>"
        for cls, text in chips
    )
    return (
        f'<div class="veri-answer-stats">{rendered}</div>'
        '<hr class="veri-answer-divider"/>'
    )


def _render_answer_origin(response: RAGResponse) -> None:
    origin, confidence, source_type = _answer_origin(response)
    kind = _answer_origin_kind(origin)
    source_count = _supporting_source_count(response)
    source_count_label = ""
    if source_count:
        noun = "source" if source_count == 1 else "sources"
        source_count_label = f"<span>{source_count} supporting {noun}</span>"
    detail = ""
    if kind == "hybrid":
        detail = """
  <div class="veri-answer-origin-detail">
    <span>Sources:</span>
    <span>\U0001f4c4 Local document</span>
    <span>\U0001f310 Web evidence</span>
    <span>\U0001f9e0 AI synthesis</span>
  </div>
        """
    st.markdown(
        f"""
<div class="veri-answer-origin veri-answer-origin-{kind}">
  <span>{origin}</span>
  <span>Confidence: {confidence}</span>
  <span>Source type: {source_type}</span>
  {source_count_label}
  {detail}
</div>
        """,
        unsafe_allow_html=True,
    )


def _friendly_evidence_note(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "no evidence available.":
        return ""
    return text


_CONFIDENCE_ORDER = ("Low", "Medium", "High")


def _answer_origin(response: RAGResponse) -> tuple[str, str, str]:
    origin, confidence, source_type = _answer_origin_raw(response)
    return origin, _cap_display_confidence(confidence, response), source_type


def _cap_display_confidence(confidence: str, response: RAGResponse) -> str:
    """Never let source authority outrank the evidence signals.

    A high-authority source (e.g. a government page) makes the raw confidence
    "High" even when the answer contradicts newer evidence — this is exactly
    how a stale answer ended up badged "High confidence". When the evidence
    layer reports an unsupported answer or a freshness/source conflict, cap the
    badge so it can never claim more certainty than the evidence supports.
    """
    if confidence not in _CONFIDENCE_ORDER:
        return confidence
    diagnostics = response.diagnostics or {}
    ceiling = _CONFIDENCE_ORDER.index("High")

    # Claim verification supported none of the answer's claims — the answer is
    # entirely unverified, so the badge is Low regardless of source authority.
    if _claims_entirely_unverified(diagnostics):
        return "Low"

    agreement = str(diagnostics.get("source_agreement") or "").lower()
    # "unsupported" from the heuristic verifier is an uncertainty signal, not
    # proof of error — it shares the Medium ceiling with evidence conflicts,
    # mirroring the backend cap in rag._cap_confidence_with_evidence.
    conflict = (
        str(diagnostics.get("answer_verification_status") or "").lower() == "unsupported"
        or diagnostics.get("local_is_older_than_web") is True
        or diagnostics.get("evidence_conflict") is True
        or "conflict" in agreement
    )
    time_sensitive = bool(
        diagnostics.get("time_sensitive") or diagnostics.get("requires_date_reconciliation")
    )
    if conflict:
        ceiling = min(ceiling, _CONFIDENCE_ORDER.index("Medium"))
    elif time_sensitive and agreement in {"", "none", "single", "single_source"}:
        # A lone source on a question whose answer changes over time does not
        # justify High confidence.
        ceiling = min(ceiling, _CONFIDENCE_ORDER.index("Medium"))

    if _CONFIDENCE_ORDER.index(confidence) <= ceiling:
        return confidence
    return _CONFIDENCE_ORDER[ceiling]


def _answer_origin_raw(response: RAGResponse) -> tuple[str, str, str]:
    used_local, used_model, used_web = _response_evidence_usage(response)
    if sum(1 for enabled in (used_local, used_model, used_web) if enabled) >= 2:
        return "\U0001f500 Hybrid", _hybrid_confidence(response, used_model=used_model), "Hybrid"
    if used_local:
        return (
            "\U0001f4c4 Local Retrieval",
            local_source_confidence(response.local_sources),
            "Document",
        )
    if response.diagnostics.get("local_file_question"):
        return "\U0001f4c4 Local Retrieval", "Low", "Document metadata"
    if response.confidence == "current-information" and used_web and response.web_sources:
        return (
            "\U0001f310 Current Information",
            source_confidence(response.web_sources[0]),
            "Web",
        )
    if used_web and response.web_sources:
        return "\U0001f310 Web Search", source_confidence(response.web_sources[0]), "Web"
    if response.confidence == "low":
        if response.diagnostics.get("time_sensitive"):
            return "\U0001f310 Current Information", "Low", "Not verified"
        return "\U0001f9e0 AI Knowledge", "Low", "Insufficient evidence"
    if _is_conversational_response(response.answer):
        return "\U0001f9e0 AI Knowledge", "N/A", "Conversation"
    return "\U0001f9e0 AI Knowledge", "Medium", "AI knowledge"


def _display_answer(response: RAGResponse) -> str:
    return _strip_trailing_model_source(response.answer)


def _strip_trailing_model_source(answer: str) -> str:
    return re.sub(
        r"\n+\s*Source:\s*(?:model|ai) knowledge(?:\s*\([^)]*\))?\s*$",
        "",
        answer or "",
        flags=re.IGNORECASE,
    ).strip()


def _uses_local_retrieval(response: RAGResponse) -> bool:
    used_local, _, _ = _response_evidence_usage(response)
    return used_local


def _uses_web_search(response: RAGResponse) -> bool:
    _, _, used_web = _response_evidence_usage(response)
    return used_web


def _hybrid_confidence(response: RAGResponse, *, used_model: bool) -> str:
    values: list[str] = []
    if _uses_local_retrieval(response):
        values.append(local_source_confidence(response.local_sources))
    if _uses_web_search(response) and response.web_sources:
        values.append(source_confidence(response.web_sources[0]))
    if used_model:
        values.append("Medium")
    if "High" in values:
        return "High"
    if "Medium" in values:
        return "Medium"
    return "Low"


def _response_evidence_usage(response: RAGResponse) -> tuple[bool, bool, bool]:
    diagnostics = response.diagnostics or {}
    used_local = bool(
        diagnostics["used_local"]
        if "used_local" in diagnostics
        else (
            bool(response.local_sources)
            or response.confidence in {"local-grounded", "local-web-assisted"}
            or bool(diagnostics.get("local_sufficient"))
        )
    )
    used_model = bool(
        diagnostics["used_model_knowledge"]
        if "used_model_knowledge" in diagnostics
        else (response.confidence == "model-only" and not used_local and not response.used_web)
    )
    used_web = bool(
        diagnostics["used_web"]
        if "used_web" in diagnostics
        else (response.used_web or response.web_sources)
    )
    return used_local, used_model, used_web


def _answer_origin_kind(origin: str) -> str:
    if origin.startswith("\U0001f4c4"):
        return "local"
    if origin.startswith("\U0001f310"):
        return "web"
    if origin.startswith("\U0001f9e0"):
        return "ai"
    if origin.startswith("\U0001f500"):
        return "hybrid"
    return "evidence"


def _supporting_source_count(response: RAGResponse) -> int:
    local_count = len(response.local_sources) if _uses_local_retrieval(response) else 0
    if local_count == 0 and _uses_local_retrieval(response):
        diagnostic_count = response.diagnostics.get("local_count", 0)
        if isinstance(diagnostic_count, int) and diagnostic_count > 0:
            local_count = diagnostic_count
        elif response.diagnostics.get("local_sufficient"):
            # RAG confirmed at least one local source was sufficient; exact count unavailable.
            local_count = 1
    web_count = len(response.web_sources) if _uses_web_search(response) else 0
    return local_count + web_count


def _source_strength_rows(response: RAGResponse) -> list[tuple[str, int, str]]:
    used_local, used_model, used_web = _response_evidence_usage(response)
    rows: list[tuple[str, int, str]] = []
    if used_local:
        rows.append(("Local", _local_strength_percent(response), "local"))
    if used_web:
        rows.append(("Web", _web_strength_percent(response), "web"))
    if used_model:
        rows.append(("AI", _ai_strength_percent(response), "ai"))
    return rows


def _local_strength_percent(response: RAGResponse) -> int:
    scores = [float(source.score or 0.0) for source in response.local_sources]
    if not scores:
        diagnostic_score = response.diagnostics.get("best_local_score")
        try:
            scores = [float(diagnostic_score)]
        except (TypeError, ValueError):
            scores = []
    if not scores:
        return 72 if response.confidence == "local-grounded" else 45
    return _score_to_percent(max(scores))


def _web_strength_percent(response: RAGResponse) -> int:
    scores = [float(source.score) for source in response.web_sources if source.score is not None]
    if scores:
        return _score_to_percent(max(scores))
    if response.web_sources:
        return _confidence_label_to_percent(source_confidence(response.web_sources[0]))
    return 72 if response.used_web else 45


def _ai_strength_percent(response: RAGResponse) -> int:
    if response.diagnostics.get("model_sufficient") is False:
        return 48
    if response.confidence == "model-only":
        return 72
    return 68


def _score_to_percent(score: float) -> int:
    if score > 1.0:
        score = score / (1.0 + score)
    return max(1, min(100, int(round(score * 100))))


def _confidence_label_to_percent(label: str) -> int:
    return {"High": 91, "Medium": 72, "Low": 45}.get(label, 60)


def _is_conversational_response(answer: str) -> bool:
    text = " ".join((answer or "").lower().split())
    if not text or len(text) > 420:
        return False
    if re.search(r"\b(?:hello|hi|hey)\b", text):
        return True
    phrase_markers = (
        "how can i assist",
        "how can i help",
        "start of our conversation",
        "i don't see any greeting",
        "i do not see any greeting",
        "good morning",
        "good afternoon",
        "good evening",
    )
    return any(marker in text for marker in phrase_markers)


def _render_copy_answer_button(answer: str, index: int) -> None:
    """Render a copy button via a single lightweight iframe.

    The style block is minimal — shared styles live in styles.py under
    .veri-copy-btn so Streamlit's main document stylesheet handles them.
    Each iframe only carries the button markup and a tiny data-payload script.
    """
    button_id = f"veri-copy-{index}"
    # Encode the answer as JSON, then escape "</" so a "</script>" sequence in
    # the answer (which can echo untrusted web content) cannot terminate the
    # script block early and inject HTML into the component frame. json.dumps
    # alone does NOT escape "</"; "<\/" is identical to "</" inside a JS string.
    answer_json = json.dumps(answer).replace("</", "<\\/")
    html = f"""
<style>
html,body{{margin:0;background:transparent}}
button{{background:rgba(54,209,196,.10);border:1px solid rgba(54,209,196,.42);border-radius:8px;
color:#36d1c4;cursor:pointer;font:700 13px system-ui,sans-serif;min-height:32px;padding:0 12px;
transition:border-color .16s,color .16s,transform .16s;width:100%}}
button:hover{{border-color:rgba(255,200,87,.72);color:#ffc857;transform:translateY(-1px)}}
</style>
<button id="{button_id}">Copy answer</button>
<script>
(function(){{
  var btn=document.getElementById({json.dumps(button_id)});
  var txt={answer_json};
  btn.addEventListener("click",function(){{
    var p=navigator.clipboard&&window.isSecureContext
      ?navigator.clipboard.writeText(txt)
      :new Promise(function(res,rej){{
          var a=document.createElement("textarea");
          a.value=txt;a.style.position="fixed";a.style.left="-9999px";
          document.body.appendChild(a);a.focus();a.select();
          document.execCommand("copy")?res():rej();
          document.body.removeChild(a);
        }});
    p.then(function(){{btn.textContent="Copied ✓";setTimeout(function(){{btn.textContent="Copy answer"}},1400)}})
     .catch(function(){{btn.textContent="Copy failed";setTimeout(function(){{btn.textContent="Copy answer"}},1800)}});
  }});
}})();
</script>"""
    components.html(html, height=38, scrolling=False)


def _render_message_history(settings: AppSettings) -> None:
    recent_entries, archived_entries = _partition_message_history(st.session_state.messages)
    if any(archived_entries.values()):
        st.markdown(
            '<div class="veri-history-label">Previous conversations</div>', unsafe_allow_html=True
        )
        for label in HISTORY_BUCKETS:
            entries = archived_entries[label]
            if not entries:
                continue
            with st.expander(label, expanded=False):
                for index, message in entries:
                    _render_message(message, settings, index, source_display="inline")
    for index, message in recent_entries:
        _render_message(message, settings, index)


def _partition_message_history(
    messages: list[dict[str, Any]],
    archive_threshold: int = ARCHIVE_MESSAGE_THRESHOLD,
    recent_count: int = RECENT_MESSAGE_COUNT,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, list[tuple[int, dict[str, Any]]]]]:
    entries = list(enumerate(messages))
    grouped_archived = {label: [] for label in HISTORY_BUCKETS}
    if len(entries) <= archive_threshold:
        return entries, grouped_archived

    archived_entries = entries[:-recent_count]
    recent_entries = entries[-recent_count:]
    for entry in archived_entries:
        grouped_archived[_history_bucket(entry[1].get("timestamp"))].append(entry)
    return recent_entries, grouped_archived


def _history_bucket(value: Any) -> str:
    try:
        timestamp = datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return "Earlier"
    today = datetime.now().astimezone().date()
    if timestamp == today:
        return "Today"
    if timestamp == today - timedelta(days=1):
        return "Yesterday"
    return "Earlier"


def _render_recommendation(
    title: str,
    body: str,
    suggestions: list[str],
    button_label: str | None = None,
    button_key: str | None = None,
    focus_section: str | None = None,
) -> None:
    items = "".join(f"<li>{item}</li>" for item in suggestions)
    st.markdown(
        f"""
<div class="veri-recommendation-card">
  <div class="veri-recommendation-kicker">Recommendations</div>
  <div class="veri-recommendation-title">⚠ {title}</div>
  <div class="veri-recommendation-body">{body}</div>
  <ul class="veri-recommendation-list">{items}</ul>
</div>
        """,
        unsafe_allow_html=True,
    )
    if button_label and button_key and focus_section:
        if st.button(button_label, key=button_key):
            st.session_state["focus_sidebar_section"] = focus_section
            st.rerun()


def _recommendation_for_response(response: RAGResponse, key_prefix: str) -> dict[str, Any] | None:
    if response.confidence == "model-selection-warning":
        return {
            "title": "Model unavailable",
            "body": "The selected model could not generate a response.",
            "suggestions": [
                "Switch to Qwen 2.5 7B",
                "Verify your Hugging Face token",
                "Use a custom model ID",
            ],
            "button_label": "Switch model",
            "button_key": f"{key_prefix}-switch-model",
            "focus_section": "models",
        }
    if response.confidence == "needs-token":
        return {
            "title": "Hugging Face token needed",
            "body": "Generation is unavailable until a valid Hugging Face token is added.",
            "suggestions": [
                "Add or replace your Hugging Face token",
                "Switch to a model your token can access",
                "Use a custom model ID if you have one",
            ],
            "button_label": "Open models",
            "button_key": f"{key_prefix}-open-models",
            "focus_section": "models",
        }
    return None


def _history_from_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role in {"user", "assistant"}:
            history.append(ChatMessage(role=role, content=str(message.get("content", ""))))
    return history


def _trim_history(max_messages: int) -> None:
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]


def _regeneration_plan(
    messages: list[dict[str, Any]],
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    user_index = _latest_user_index(messages)
    if user_index is None:
        return None, None, messages
    prompt = str(messages[user_index].get("content", "")).strip()
    if not prompt:
        return None, None, messages
    # Preserve the document focus of suggestion-derived prompts so regenerating
    # re-runs the same document-scoped retrieval, not an unscoped re-guess.
    focus = str(messages[user_index].get("focus_document") or "").strip() or None
    return prompt, focus, messages[: user_index + 1]


def _latest_user_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def _now_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_timestamp(value: Any) -> str:
    if not value:
        return ""
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    return timestamp.strftime("%Y-%m-%d %H:%M")
