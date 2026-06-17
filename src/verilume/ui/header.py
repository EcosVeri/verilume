"""Header component."""

from __future__ import annotations

import streamlit as st

from verilume.settings import AppSettings


def render_header(settings: AppSettings, stats: dict[str, int]) -> None:
    if not settings.enable_web_search:
        web_state = "Web off"
    elif settings.web_search_ready():
        web_state = f"{settings.web_search_provider_label()} ready"
    else:
        web_state = f"{settings.web_search_provider_label()} setup needed"
    token_state = "HF ready" if settings.hf_token else "HF token needed"
    st.markdown(
        f"""
<div class="veri-header">
  <div class="veri-brand">Verilume</div>
  <div class="veri-title">{settings.app_icon} {settings.app_title}</div>
  <div class="veri-subtitle">Answers powered by local retrieval, trusted AI generation, and transparent sources.</div>
  <div class="veri-pill-row">
    <span class="veri-pill"><span class="veri-dot green"></span>{token_state}</span>
    <span class="veri-pill"><span class="veri-dot amber"></span>{stats.get("uploaded_documents", 0)} docs</span>
    <span class="veri-pill"><span class="veri-dot"></span>{stats.get("chunks_indexed", 0)} chunks</span>
    <span class="veri-pill"><span class="veri-dot coral"></span>{web_state}</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
