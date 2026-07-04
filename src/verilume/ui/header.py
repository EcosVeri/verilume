"""Header component."""

from __future__ import annotations

import streamlit as st

from verilume.settings import AppSettings, save_user_config


def render_header(settings: AppSettings, stats: dict[str, int]) -> None:
    if not settings.enable_web_search:
        web_state, web_dot = "Web off", "muted"
    elif settings.web_search_ready():
        web_state, web_dot = f"{settings.web_search_provider_label()} ready", "green"
    else:
        web_state, web_dot = f"{settings.web_search_provider_label()} setup needed", "amber"
    model_ready = settings.generation_ready()
    if settings.generation_backend == "ollama":
        token_state = "Ollama ready" if model_ready else "Ollama model needed"
    else:
        token_state = "HF ready" if model_ready else "HF token needed"
    model_dot = "green" if model_ready else "amber"
    docs = stats.get("uploaded_documents", 0)
    chunks = stats.get("chunks_indexed", 0)
    docs_dot = "green" if docs else "amber"
    chunks_dot = "green" if chunks else "amber"
    header_col, toggle_col = st.columns([0.88, 0.12], vertical_alignment="top")
    with header_col:
        st.markdown(
            f"""
<div class="veri-header">
  <div class="veri-brand">Verilume</div>
  <div class="veri-title">{settings.app_icon} {settings.app_title}</div>
  <div class="veri-subtitle">Search documents, verify evidence, and compare sources with transparent AI.</div>
  <div class="veri-pill-row">
    <span class="veri-pill"><span class="veri-dot {model_dot}"></span>{token_state}</span>
    <span class="veri-pill"><span class="veri-dot {docs_dot}"></span>{docs} docs</span>
    <span class="veri-pill"><span class="veri-dot {chunks_dot}"></span>{chunks} chunks</span>
    <span class="veri-pill"><span class="veri-dot {web_dot}"></span>{web_state}</span>
  </div>
</div>
        """,
            unsafe_allow_html=True,
        )
    with toggle_col:
        render_theme_toggle(settings)


def render_theme_toggle(settings: AppSettings) -> AppSettings:
    current = settings.appearance or "dark"
    next_appearance = "light" if current == "dark" else "dark"
    icon = "☀️" if current == "dark" else "🌙"
    label = "Switch to light mode" if current == "dark" else "Switch to dark mode"
    st.markdown('<div class="veri-theme-toggle-wrap">', unsafe_allow_html=True)
    if st.button(icon, help=label, key="appearance_toggle"):
        updated = settings.with_overrides(appearance=next_appearance)
        save_user_config(updated)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    return settings
