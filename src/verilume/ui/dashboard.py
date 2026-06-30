"""Persistent dashboard for the Streamlit workspace."""

from __future__ import annotations

import re
from html import escape
from typing import Any, Sequence

import streamlit as st

from verilume.core.prompt_suggestions import (
    PromptSuggestion,
    generate_suggested_prompts,
)
from verilume.settings import AppSettings

DEFAULT_DASHBOARD_COLLAPSED = True


@st.cache_data(ttl=120, show_spinner=False)
def _generate_prompts_cached(
    recent_documents: tuple[str, ...],
    recent_activity: tuple[str, ...],
    settings: AppSettings,
) -> list[PromptSuggestion]:
    return generate_suggested_prompts(list(recent_documents), list(recent_activity), settings)


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
            tuple(recent_documents), tuple(recent_activity), settings
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
            suggestion.title,
            key=f"suggest_dashboard_{index}_{_prompt_key_fragment(suggestion.title)}",
            help=suggestion.category.replace("_", " ").title(),
            use_container_width=True,
        ):
            st.session_state["pending_prompt"] = suggestion.prompt
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


def _prompt_key_fragment(prompt: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return normalized or "prompt"
