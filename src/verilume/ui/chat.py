"""Chat interface component."""

from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timedelta
from html import escape
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from verilume.core.conversation_state import ConversationState
from verilume.core.schemas import ChatMessage, RAGResponse
from verilume.rag import GenerationStopped, get_rag_service
from verilume.settings import AppSettings
from verilume.utils.exporting import chat_to_markdown, chat_to_pdf
from verilume.utils.formatting import (
    local_source_confidence,
    local_source_rows,
    source_badge,
    source_confidence,
    web_source_rows,
)

LOGGER = logging.getLogger(__name__)
ARCHIVE_MESSAGE_THRESHOLD = 10
RECENT_MESSAGE_COUNT = 6
HISTORY_BUCKETS = ("Today", "Yesterday", "Earlier")

# Neutral Material avatars keep the chat focused and avoid loud emoji badges.
USER_ICON = ":material/account_circle:"
ASSISTANT_ICON = ":material/auto_awesome:"
ANSWER_HEADER = "Verilume Findings"
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
    if st.session_state.regenerate_requested:
        st.session_state.regenerate_requested = False
        regenerate_prompt, st.session_state.messages = _regeneration_plan(st.session_state.messages)

    _render_toolbar(settings)
    _render_message_history(settings)

    if regenerate_prompt:
        _generate_assistant_response(settings, regenerate_prompt)
        return

    if not st.session_state.messages:
        _render_welcome_screen()

    manual_prompt = st.chat_input(_chat_placeholder(settings))
    prompt = _consume_pending_prompt() or manual_prompt
    if not prompt:
        return

    _handle_prompt(settings, prompt)


def _handle_prompt(settings: AppSettings, prompt: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": _now_timestamp()}
    )
    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(prompt)

    _generate_assistant_response(settings, prompt)


def _chat_placeholder(settings: AppSettings) -> str:
    mode = str(getattr(settings, "search_mode", "Auto") or "Auto")
    icon = SEARCH_MODE_PLACEHOLDER_ICONS.get(mode, "🌐")
    label = _search_mode_display_label(mode)
    example = str(
        st.session_state.get("chat_placeholder_example")
        or random.choice(CHAT_PLACEHOLDER_EXAMPLES)
    )
    return f"{icon} {label}  {example}"


def _consume_pending_prompt() -> str | None:
    prompt = st.session_state.pop("pending_prompt", None)
    if prompt is None:
        return None
    prompt_text = str(prompt).strip()
    return prompt_text or None


def _generate_assistant_response(settings: AppSettings, prompt: str) -> None:
    history = _history_from_messages(st.session_state.messages[:-1])
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        placeholder = st.empty()
        stage_placeholder = st.empty()
        st.session_state.generating = True
        st.session_state.stop_requested = False
        try:
            def update_stage(label: str) -> None:
                stage_placeholder.markdown(_loading_stage_html(label), unsafe_allow_html=True)

            update_stage("Searching local evidence...")
            response = get_rag_service(settings).ask(
                prompt,
                history,
                conversation_state=st.session_state.conversation_state,
                should_stop=lambda: st.session_state.stop_requested,
                on_stage=update_stage,
            )
            placeholder.empty()
            stage_placeholder.empty()
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
            _render_sources(response, settings)
            st.session_state.messages.append(assistant_message)
            _trim_history(settings.max_chat_messages)
        except GenerationStopped:
            message = "Generation stopped by user."
            assistant_message = {
                "role": "assistant",
                "content": message,
                "timestamp": _now_timestamp(),
            }
            placeholder.empty()
            stage_placeholder.empty()
            _render_assistant_meta(assistant_message, len(st.session_state.messages))
            st.markdown(message)
            st.session_state.messages.append(assistant_message)
        except Exception:
            LOGGER.exception("Chat generation failed.")
            message = (
                "Sorry, something went wrong while generating the answer. "
                "Please check the terminal logs and try again."
            )
            assistant_message = {
                "role": "assistant",
                "content": message,
                "timestamp": _now_timestamp(),
            }
            placeholder.empty()
            stage_placeholder.empty()
            _render_assistant_meta(assistant_message, len(st.session_state.messages))
            st.markdown(message)
            st.session_state.messages.append(assistant_message)
        finally:
            st.session_state.generating = False


def _render_toolbar(settings: AppSettings) -> None:
    can_regenerate = _latest_user_index(st.session_state.messages) is not None
    col_a, col_b, col_c, col_d, col_e = st.columns([1.1, 1.2, 0.8, 1, 2])
    with col_a:
        if st.button(
            "\u23f9 Stop response",
            disabled=not st.session_state.generating,
            use_container_width=True,
        ):
            st.session_state.stop_requested = True
            if st.session_state.generating:
                st.warning("Stop requested. The current provider call will finish this turn.")
    with col_b:
        if st.button(
            "Regenerate response",
            disabled=st.session_state.generating or not can_regenerate,
            use_container_width=True,
        ):
            st.session_state.regenerate_requested = True
            st.rerun()
    with col_c:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_state = ConversationState()
            st.rerun()
    markdown = chat_to_markdown(st.session_state.messages, settings.app_title)
    with col_d:
        st.download_button(
            "Markdown",
            data=markdown,
            file_name="verilume-chat.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_e:
        try:
            pdf = chat_to_pdf(st.session_state.messages, settings.app_title)
            st.download_button(
                "PDF",
                data=pdf,
                file_name="verilume-chat.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.download_button(
                "PDF unavailable",
                data=b"",
                file_name="verilume-chat.pdf",
                disabled=True,
                use_container_width=True,
            )


def _render_welcome_screen() -> None:
    st.markdown(
        """
<div class="veri-welcome">
  <div class="veri-welcome-kicker">Welcome</div>
  <div class="veri-welcome-title">Search documents, research sources, and compare evidence.</div>
  <div class="veri-welcome-grid">
    <div><strong>📄 Search documents</strong><span>Find local facts and citations.</span></div>
    <div><strong>📚 Summarise files</strong><span>Turn long PDFs into clear briefs.</span></div>
    <div><strong>⚖ Compare evidence</strong><span>Separate local, AI, and web support.</span></div>
    <div><strong>🌍 Current facts</strong><span>Use web sources when enabled.</span></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _loading_stage_html(label: str) -> str:
    current = _loading_stage_index(label)
    steps = ("Searching Local", "Searching Web", "Ranking", "Generating")
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
    if "rank" in normalized or "evidence" in normalized or "verif" in normalized:
        return 2
    if "generat" in normalized or "synthesis" in normalized or "answer" in normalized:
        return 3
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
            _render_sources(response, settings, display=source_display)
        else:
            st.markdown(str(message.get("content", "")))


def _render_answer(response: RAGResponse, key_prefix: str) -> None:
    recommendation = _recommendation_for_response(response, key_prefix)
    if recommendation is None:
        st.markdown(_display_answer(response))
        _render_evidence_summary(response)
        return
    _render_recommendation(**recommendation)


def _render_sources(
    response: RAGResponse, settings: AppSettings, display: str = "expander"
) -> None:
    if display == "inline":
        _render_sources_inline(response, settings)
        return
    _render_sources_expanded(response, settings)


def _render_sources_expanded(response: RAGResponse, settings: AppSettings) -> None:
    _render_evidence_details(response)
    _render_benchmark_report(response)
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
    badges = [origin, f"Confidence: {confidence}", source_type]
    if source_count_label:
        badges.append(source_count_label)
    badges.extend(_evidence_badges(response)[1:])
    rendered_badges = "".join(f"<span>{escape(label)}</span>" for label in badges if label)
    strength_rows = _source_strength_rows(response)
    rendered_strength = "".join(
        (
            '<div class="veri-source-strength-row">'
            f'<span class="veri-source-strength-label">{escape(label)}</span>'
            '<span class="veri-source-strength-track">'
            f'<span class="veri-source-strength-fill veri-source-strength-{kind}" style="width:{score}%"></span>'
            "</span>"
            f'<span class="veri-source-strength-value">{score}%</span>'
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
            f'<span>{escape(label)}</span>'
            f'<strong>{escape(value)}</strong>'
            "</div>"
        )
        for label, value in (
            ("Search mode", search_mode),
            ("Sources searched", searched),
            ("Sources used", used),
            ("Winner", winner),
            ("Claim support", claim_support),
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
    rows = _evidence_detail_rows(response)
    if not rows:
        return

    with st.expander(EVIDENCE_HEADER, expanded=False):
        for label, value in rows:
            st.markdown(f"**{label}:** {value}")


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


def _render_benchmark_report(response: RAGResponse) -> None:
    report = _benchmark_report(response)
    if not report:
        return

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
    if not rows:
        return

    with st.expander("Benchmark Results", expanded=True):
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
    }
    return labels.get(confidence, f"Evidence: {confidence.replace('-', ' ').title()}")


def _evidence_detail_rows(response: RAGResponse) -> list[tuple[str, str]]:
    diagnostics = response.diagnostics or {}
    fields = (
        ("Evidence note", _friendly_evidence_note(diagnostics.get("evidence_note"))),
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
    for row in local_source_rows(response.local_sources):
        cards.append(
            """
<div class="veri-source-card">
  <div class="veri-source-card-top">
    <div class="veri-source-card-title"><span>📄</span>{document}</div>
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
    for group_title, grouped_sources in _group_web_source_rows(
        web_source_rows(response.web_sources)
    ):
        st.markdown(
            f'<div class="veri-source-group-heading">{escape(group_title)}</div>',
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
    safe_url = escape(url, quote=True)
    safe_source = escape(source or url or "Web source")
    return f"""
<div class="veri-source-card">
  <div class="veri-source-card-top">
    <a class="veri-source-card-title" href="{safe_url}" target="_blank" rel="noopener noreferrer"><span>{escape(icon)}</span>{safe_source}</a>
    <div class="veri-source-card-badge">{escape(citation)}</div>
  </div>
  <div class="veri-source-card-meta">
    <span>Confidence: {escape(confidence)}</span>
    <span>{escape(url)}</span>
  </div>
  <div class="veri-source-card-preview">{escape(preview)}</div>
</div>
    """


def _group_web_source_rows(
    rows: list[dict[str, str | float | None]],
) -> list[tuple[str, list[dict[str, str | float | None]]]]:
    grouped: dict[str, list[dict[str, str | float | None]]] = {}
    for row in rows:
        source_type = str(row.get("Source type") or "Web")
        grouped.setdefault(source_type, []).append(row)
    return [
        (_web_source_group_title(source_type, group), group)
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


def _answer_origin(response: RAGResponse) -> tuple[str, str, str]:
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
        else:
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
    button_id = f"copy-answer-{index}"
    html = f"""
<style>
html, body {{
  background: transparent;
  margin: 0;
}}

button {{
  background: rgba(54, 209, 196, .10);
  border: 1px solid rgba(54, 209, 196, .42);
  border-radius: 8px;
  color: #36d1c4;
  cursor: pointer;
  font: 700 13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  min-height: 32px;
  padding: 0 12px;
  transition: border-color .16s ease, color .16s ease, transform .16s ease;
  width: 100%;
}}

button:hover {{
  border-color: rgba(255, 200, 87, .72);
  color: #ffc857;
  transform: translateY(-1px);
}}
</style>
<button id="{button_id}">Copy answer</button>
<script>
const button = document.getElementById({json.dumps(button_id)});
const answer = {json.dumps(answer)};
button.addEventListener("click", async () => {{
  try {{
    if (navigator.clipboard && window.isSecureContext) {{
      await navigator.clipboard.writeText(answer);
    }} else {{
      const area = document.createElement("textarea");
      area.value = answer;
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.focus();
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
    }}
    button.textContent = "Copied";
    setTimeout(() => button.textContent = "Copy answer", 1400);
  }} catch (error) {{
    button.textContent = "Copy failed";
    setTimeout(() => button.textContent = "Copy answer", 1800);
  }}
}});
</script>
        """
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


def _regeneration_plan(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    user_index = _latest_user_index(messages)
    if user_index is None:
        return None, messages
    prompt = str(messages[user_index].get("content", "")).strip()
    if not prompt:
        return None, messages
    return prompt, messages[: user_index + 1]


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
