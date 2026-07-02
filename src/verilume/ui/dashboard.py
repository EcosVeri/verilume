"""Persistent dashboard for the Streamlit workspace."""

from __future__ import annotations

import logging
import re
from html import escape
from typing import Any, Sequence

import streamlit as st

from verilume.core.document_index import build_document_index
from verilume.core.prompt_suggestions import (
    PromptSuggestion,
    generate_suggested_prompts,
)
from verilume.ingest import document_metadata_from_manifest
from verilume.settings import AppSettings

LOGGER = logging.getLogger(__name__)

DEFAULT_DASHBOARD_COLLAPSED = True


@st.cache_data(ttl=120, show_spinner=False)
def _generate_prompts_cached(
    recent_activity: tuple[str, ...],
    settings: AppSettings,
) -> list[PromptSuggestion]:
    try:
        document_index = build_document_index(document_metadata_from_manifest(settings))
    except Exception:
        LOGGER.debug("Failed to build document index for prompt suggestions.", exc_info=True)
        document_index = []
    return generate_suggested_prompts(document_index, list(recent_activity), settings)


def render_dashboard(
    settings: AppSettings,
    library_stats: dict[str, int],
    recent_documents: Sequence[str],
    recent_activity: Sequence[str],
    suggested_prompts: Sequence[PromptSuggestion] | None = None,
) -> None:
    """Render the dashboard independently from chat empty-state UI."""
    if "dashboard_collapsed" not in st.session_state:
        st.session_state["dashboard_collapsed"] = DEFAULT_DASHBOARD_COLLAPSED

    # Auto-collapse once the conversation starts (ChatGPT-style) to give the
    # chat more room. Fires a single time so it never fights a later manual
    # expand by the user.
    if st.session_state.get("messages") and not st.session_state.get("_dashboard_autocollapsed"):
        st.session_state["dashboard_collapsed"] = True
        st.session_state["_dashboard_autocollapsed"] = True

    dashboard_collapsed = bool(st.session_state.get("dashboard_collapsed", False))
    chevron = "⌄ Dashboard" if dashboard_collapsed else "⌃ Dashboard"
    st.markdown(
        '<div class="veri-dashboard-divider">'
        f'<span class="veri-dashboard-divider-label">{escape(settings.app_title)} workspace</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    _col_spacer, col_toggle = st.columns([5, 1])
    with col_toggle:
        st.markdown('<div class="veri-dark-button-anchor veri-dashboard-toggle-wrap"></div>', unsafe_allow_html=True)
        if st.button(
            chevron,
            help=f"Show or hide the {settings.app_title} dashboard.",
            key="dashboard-collapse-toggle",
            use_container_width=True,
        ):
            st.session_state["dashboard_collapsed"] = not dashboard_collapsed
            st.rerun()

    if dashboard_collapsed:
        return

    render_metric_cards(library_stats)

    doc_col, activity_col, prompt_col = st.columns(3)
    with doc_col:
        render_recent_documents(recent_documents)
    with activity_col:
        render_recent_activity(recent_activity)
    with prompt_col:
        # Generate prompts here, after the early-return, so they are skipped
        # entirely when the dashboard is collapsed (saves compute on every rerun).
        effective_prompts = suggested_prompts if suggested_prompts is not None else _generate_prompts_cached(
            tuple(recent_activity), settings
        )
        render_suggested_prompts(effective_prompts)


def render_metric_cards(library_stats: dict[str, int]) -> None:
    cards = (
        ("\U0001f4c4", library_stats.get("uploaded_documents", 0), "Documents"),
        ("\U0001f4da", library_stats.get("pdf_pages", 0), "Pages"),
        ("\U0001f9e9", library_stats.get("chunks_indexed", 0), "Chunks"),
    )
    rendered = "".join(
        (
            '<div class="veri-metric-card">'
            f'<div class="veri-metric-icon">{icon}</div>'
            f'<div class="veri-metric-value">{int(value or 0):,}</div>'
            f'<div class="veri-metric-label">{label}</div>'
            "</div>"
        )
        for icon, value, label in cards
    )
    st.markdown(f'<div class="veri-metric-grid">{rendered}</div>', unsafe_allow_html=True)


def render_recent_documents(recent_documents: Sequence[str]) -> None:
    doc_items = "".join(
        f'<div class="veri-mini-row"><span>\U0001f4c4</span><strong>{escape(name)}</strong></div>'
        for name in recent_documents
    ) or '<div class="veri-mini-muted">Upload documents to start building your local library.</div>'
    st.markdown(
        f"""
<div class="veri-workspace-card">
  <div class="veri-workspace-kicker">Recent Documents</div>
  {doc_items}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_activity(recent_activity: Sequence[str]) -> None:
    activity_items = "".join(
        f'<div class="veri-mini-row"><span>\u2318</span><strong>{escape(item[:72])}</strong></div>'
        for item in recent_activity
    ) or '<div class="veri-mini-muted">Recent searches will appear here.</div>'
    st.markdown(
        f"""
<div class="veri-workspace-card">
  <div class="veri-workspace-kicker">Recent Activity</div>
  {activity_items}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_suggested_prompts(
    suggested_prompts: Sequence[PromptSuggestion] | None = None,
) -> None:
    suggestions = list(
        suggested_prompts
        if suggested_prompts is not None
        else generate_suggested_prompts([], [], AppSettings())
    )
    st.markdown(
        """
<div class="veri-workspace-card veri-dashboard-prompt-card">
  <div class="veri-workspace-kicker">Suggested Prompts</div>
  <div class="veri-mini-muted">Fresh actions for the current library.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    for index, suggestion in enumerate(suggestions, start=1):
        st.markdown('<div class="veri-dark-button-anchor veri-prompt-button-wrap"></div>', unsafe_allow_html=True)
        if st.button(
            f"{_prompt_icon(suggestion)} {suggestion.title}",
            key=f"suggest_dashboard_{index}_{_prompt_key_fragment(suggestion.title)}",
            help=suggestion.category.replace("_", " ").title(),
            use_container_width=True,
        ):
            st.session_state["pending_prompt"] = suggestion.prompt
            # Carry the source document through the click so retrieval can be
            # focused on the document the suggestion was generated from.
            focus = getattr(suggestion, "document_filename", None)
            if focus:
                st.session_state["pending_prompt_document"] = focus
            else:
                st.session_state.pop("pending_prompt_document", None)
            st.rerun()


def render_empty_document_state(library_stats: dict[str, int]) -> None:
    if library_stats.get("uploaded_documents", 0) > 0:
        return
    st.markdown(
        """
<div class="veri-empty-state">
  <div class="veri-empty-state-title">No local documents indexed yet</div>
  <div class="veri-empty-state-body">
    Upload documents in the sidebar to ground answers in your own files.
    Web research and model knowledge remain available when configured.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def recent_activity_from_messages(messages: Sequence[dict[str, Any]], limit: int = 3) -> list[str]:
    return [
        str(message.get("content", "")).strip()
        for message in reversed(messages)
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ][:limit]


_PROMPT_CATEGORY_ICONS = {
    "onboarding": "🚀",
    "collection": "📚",
    "inventory": "📋",
    "listing": "📋",
    "comparison": "📊",
    "search": "🔍",
    "recent_activity": "🕘",
    "recent_upload": "🆕",
    "summary": "📄",
    "formula": "🧮",
    "extraction": "🔎",
    "structure": "🗂️",
    "table": "📊",
    "presentation": "📝",
    "scientific_paper": "🧠",
}

_PROMPT_TITLE_KEYWORD_ICONS = (
    ("speaker note", "📝"),
    ("lecture note", "📝"),
    ("literature review", "📝"),
    ("list", "📋"),
    ("compare", "📊"),
    ("summar", "📄"),
    ("explain", "🧠"),
)


def _prompt_icon(suggestion: PromptSuggestion) -> str:
    """Return a display-only icon for a suggested prompt button."""
    title = suggestion.title.lower()
    for keyword, icon in _PROMPT_TITLE_KEYWORD_ICONS:
        if keyword in title:
            return icon
    return _PROMPT_CATEGORY_ICONS.get(suggestion.category, "💡")


def _prompt_key_fragment(prompt: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return normalized or "prompt"
