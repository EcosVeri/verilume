"""Chat interface component."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from html import escape
from typing import Any
from urllib.parse import quote

import streamlit as st

from verilume.core.conversation_context import ConversationState
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
    web_source_type,
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
LOCAL_SOURCES_HEADER = "Document Citations"
WEB_SOURCES_HEADER = "Web Evidence"


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


def render_chat(settings: AppSettings) -> None:
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

    prompt = st.chat_input("Ask Verilume anything...")
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": _now_timestamp()}
    )
    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(prompt)

    _generate_assistant_response(settings, prompt)


def _generate_assistant_response(settings: AppSettings, prompt: str) -> None:
    history = _history_from_messages(st.session_state.messages[:-1])
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        placeholder = st.empty()
        st.session_state.generating = True
        st.session_state.stop_requested = False
        try:
            with st.status("Evidence collection", expanded=False) as status:

                def update_stage(label: str) -> None:
                    status.write(label)

                response = get_rag_service(settings).ask(
                    prompt,
                    history,
                    conversation_state=st.session_state.conversation_state,
                    should_stop=lambda: st.session_state.stop_requested,
                    on_stage=update_stage,
                )
                status.update(label="Evidence collection complete", state="complete")
            placeholder.empty()
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
            _render_assistant_meta(assistant_message, len(st.session_state.messages))
            st.markdown(message)
            st.session_state.messages.append(assistant_message)
        finally:
            st.session_state.generating = False


def _render_toolbar(settings: AppSettings) -> None:
    can_regenerate = _latest_user_index(st.session_state.messages) is not None
    col_a, col_b, col_c, col_d, col_e = st.columns([1.1, 1.2, 0.8, 1, 2])
    with col_a:
        if st.button("\u23f9 Stop response", width="stretch"):
            st.session_state.stop_requested = True
            if st.session_state.generating:
                st.warning("Stop requested. The current provider call will finish this turn.")
    with col_b:
        if st.button(
            "Regenerate response",
            disabled=st.session_state.generating or not can_regenerate,
            width="stretch",
        ):
            st.session_state.regenerate_requested = True
            st.rerun()
    with col_c:
        if st.button("Clear", width="stretch"):
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
            width="stretch",
        )
    with col_e:
        try:
            pdf = chat_to_pdf(st.session_state.messages, settings.app_title)
            st.download_button(
                "PDF",
                data=pdf,
                file_name="verilume-chat.pdf",
                mime="application/pdf",
                width="stretch",
            )
        except Exception:
            st.download_button(
                "PDF unavailable",
                data=b"",
                file_name="verilume-chat.pdf",
                disabled=True,
                width="stretch",
            )


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
        _render_answer_origin(response)
        _render_evidence_badges(response)
        st.markdown(_display_answer(response))
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
        ("Freshness note", diagnostics.get("freshness_note")),
        ("Local older than web", diagnostics.get("local_is_older_than_web")),
        ("Source agreement", diagnostics.get("source_agreement")),
        ("Evidence winner", diagnostics.get("evidence_winner")),
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
        return ", ".join(str(item) for item in value if item)
    return str(value)


def _render_local_sources_table(response: RAGResponse) -> None:
    st.dataframe(local_source_rows(response.local_sources), width="stretch", hide_index=True)


def _render_web_source_groups(response: RAGResponse) -> None:
    for group_title, grouped_sources in _group_web_source_rows(
        web_source_rows(response.web_sources)
    ):
        st.markdown(f"**{group_title}**")
        for source in grouped_sources:
            preview = str(source["Preview"] or "").strip()
            preview_line = f"  \n  {preview}" if preview else ""
            st.markdown(
                f"- [{source['Source']}]({source['URL']}) "
                f"· Confidence: **{source['Confidence']}**"
                f"{preview_line}"
            )


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
    uses_local = _uses_local_retrieval(response)
    uses_web = _uses_web_search(response)
    if uses_local and uses_web:
        return "\U0001f500 Hybrid", _hybrid_confidence(response), _hybrid_source_type(response)
    if uses_local:
        return (
            "\U0001f4c4 Local Retrieval",
            local_source_confidence(response.local_sources),
            "Document",
        )
    if response.diagnostics.get("local_file_question"):
        return "\U0001f4c4 Local Retrieval", "Low", "Document metadata"
    if response.confidence == "current-information" and response.web_sources:
        return (
            "\U0001f310 Current Information",
            source_confidence(response.web_sources[0]),
            "Recent web evidence",
        )
    if uses_web and response.web_sources:
        source_type = web_source_type(response.web_sources[0])
        return "\U0001f310 Web Search", source_confidence(response.web_sources[0]), source_type
    if response.confidence == "low":
        if response.diagnostics.get("time_sensitive"):
            return "\U0001f310 Current Information", "Low", "Not verified"
        return "\U0001f9e0 AI Knowledge", "Low", "Insufficient evidence"
    if _is_conversational_response(response.answer):
        return "\U0001f9e0 AI Knowledge", "N/A", "Conversation"
    return "\U0001f9e0 AI Knowledge", "Medium", "Not externally verified"


def _display_answer(response: RAGResponse) -> str:
    answer = _strip_trailing_model_source(response.answer)
    labels: dict[str, str] = {}
    for source in response.web_sources:
        labels[source.label] = source_badge(web_source_type(source))
    for source in response.local_sources:
        labels[source.label] = source_badge("Local document")

    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        return f"**{labels[label]}**" if label in labels else match.group(0)

    return re.sub(r"\[([SW]\d+)\]", replace, answer)


def _strip_trailing_model_source(answer: str) -> str:
    return re.sub(
        r"\n+\s*Source:\s*(?:model|ai) knowledge(?:\s*\([^)]*\))?\s*$",
        "",
        answer or "",
        flags=re.IGNORECASE,
    ).strip()


def _uses_local_retrieval(response: RAGResponse) -> bool:
    return (
        bool(response.local_sources)
        or response.confidence in {"local-grounded", "local-web-assisted"}
        or bool(response.diagnostics.get("local_sufficient"))
    )


def _uses_web_search(response: RAGResponse) -> bool:
    return bool(response.used_web or response.web_sources)


def _hybrid_confidence(response: RAGResponse) -> str:
    values = [local_source_confidence(response.local_sources)]
    if response.web_sources:
        values.append(source_confidence(response.web_sources[0]))
    if "High" in values:
        return "High"
    if "Medium" in values:
        return "Medium"
    return "Low"


def _hybrid_source_type(response: RAGResponse) -> str:
    values = ["Document"]
    if response.web_sources:
        values.append(web_source_type(response.web_sources[0]))
    values.append("AI synthesis")
    return " + ".join(dict.fromkeys(values))


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
    local_count = len(response.local_sources)
    if local_count == 0 and _uses_local_retrieval(response):
        diagnostic_count = response.diagnostics.get("local_count", 0)
        if isinstance(diagnostic_count, int) and diagnostic_count > 0:
            local_count = diagnostic_count
        else:
            local_count = 1
    return local_count + len(response.web_sources)


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
<button id="{button_id}" style="
  background:#1a1f27;
  border:1px solid #2b303a;
  border-radius:10px;
  color:#f5f2e8;
  cursor:pointer;
  font:600 13px system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  min-height:32px;
  padding:0 12px;
  width:100%;
">Copy answer</button>
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
    st.iframe(
        f"data:text/html;charset=utf-8,{quote(html, safe='')}",
        height=38,
        width="stretch",
    )


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
