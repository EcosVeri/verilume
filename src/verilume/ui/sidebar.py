"""Sidebar controls."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from verilume.settings import (
    ANSWER_STYLE_CHOICES,
    DEFAULT_MODEL_CHOICES,
    WEB_SEARCH_API_KEY_FIELDS,
    WEB_SEARCH_PROVIDER_LABELS,
    AppSettings,
    save_user_config,
)


@dataclass(slots=True)
class SidebarState:
    settings: AppSettings
    uploaded_files: list
    build_clicked: bool
    reset_clicked: bool


def render_sidebar(base_settings: AppSettings, stats: dict[str, int]) -> SidebarState:
    with st.sidebar:
        st.markdown('<div class="veri-side-brand">Verilume</div>', unsafe_allow_html=True)
        st.subheader("Verilume Studio")
        focused_section = st.session_state.pop("focus_sidebar_section", "")
        if "custom_model" not in st.session_state:
            st.session_state.custom_model = ""
        with st.expander(
            "Models",
            expanded=focused_section == "models" or not base_settings.hf_token,
        ):
            hf_token = st.text_input(
                "Hugging Face token",
                value=st.session_state.get("hf_token", base_settings.hf_token),
                type="password",
            )
            st.session_state.hf_token = hf_token

            options = list(dict.fromkeys([base_settings.hf_llm_model, *DEFAULT_MODEL_CHOICES]))
            selected_index = options.index(base_settings.hf_llm_model) if base_settings.hf_llm_model in options else 0
            model_choice = st.selectbox("Hugging Face model", options=options, index=selected_index)
            custom_model = st.text_input("Custom model", key="custom_model")

            hf_model = model_choice
            if custom_model.strip():
                if _is_valid_hf_model_id(custom_model.strip()):
                    hf_model = custom_model.strip()
                else:
                    st.warning("Custom model must be a Hugging Face model ID like owner/model-name.")

        provider_ids = list(WEB_SEARCH_PROVIDER_LABELS)
        provider_labels = [WEB_SEARCH_PROVIDER_LABELS[provider] for provider in provider_ids]
        current_provider = base_settings.web_search_provider
        provider_index = provider_ids.index(current_provider) if current_provider in provider_ids else 0
        with st.expander("Search", expanded=focused_section == "search"):
            enable_web = st.toggle("Web search", value=base_settings.enable_web_search)
            provider_label = st.selectbox(
                "Web search provider",
                options=provider_labels,
                index=provider_index,
            )
            web_search_provider = provider_ids[provider_labels.index(provider_label)]
            web_search_overrides = _render_web_provider_fields(web_search_provider, base_settings)

        with st.expander(
            "Documents",
            expanded=focused_section == "documents" or not stats.get("uploaded_documents", 0),
        ):
            st.markdown(
                """
<div class="veri-upload-card">
  <div class="veri-upload-title">Build your knowledge base</div>
  <div class="veri-upload-grid">
    <div>
      <div class="veri-upload-label">Supported</div>
      <div class="veri-upload-value">PDF • DOCX • TXT • MD • CSV</div>
    </div>
    <div>
      <div class="veri-upload-label">Maximum size</div>
      <div class="veri-upload-value">200 MB per file</div>
    </div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Build your knowledge base",
                type=["pdf", "txt", "md", "markdown", "csv", "docx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                build_clicked = st.button("Build KB", type="primary", use_container_width=True)
            with col_b:
                reset_clicked = st.button("Reset DB", use_container_width=True)

            st.caption("Library snapshot")
            c1, c2 = st.columns(2)
            c1.metric("Documents", stats.get("uploaded_documents", 0))
            c2.metric("PDF pages", stats.get("pdf_pages", 0))
            c3, c4 = st.columns(2)
            c3.metric("Chunks", stats.get("chunks_indexed", 0))
            c4.metric("Types", stats.get("supported_types", 0))

        with st.expander("Retrieval", expanded=focused_section == "retrieval"):
            answer_style_index = ANSWER_STYLE_CHOICES.index(base_settings.answer_style)
            answer_style = st.radio(
                "Answer style",
                options=ANSWER_STYLE_CHOICES,
                index=answer_style_index,
                horizontal=True,
            )
            show_local = st.toggle("Show local citations", value=base_settings.show_local_sources)
            retriever_k = st.slider("Retriever K", min_value=2, max_value=12, value=base_settings.retriever_k)
            threshold = st.slider(
                "Score threshold",
                min_value=0.0,
                max_value=0.95,
                value=float(base_settings.retrieval_score_threshold),
                step=0.05,
            )

        config_settings = base_settings.with_overrides(
            hf_token=hf_token.strip(),
            hf_llm_model=hf_model,
            web_search_provider=web_search_provider,
            enable_web_search=enable_web,
            answer_style=answer_style,
            show_local_sources=show_local,
            retriever_k=retriever_k,
            retrieval_score_threshold=threshold,
            **web_search_overrides,
        )
        if st.button("Save settings", type="primary", use_container_width=True):
            path = save_user_config(config_settings)
            st.success(f"Configuration saved to {path}")

    settings = base_settings.with_overrides(
        hf_token=hf_token.strip(),
        hf_llm_model=hf_model,
        web_search_provider=web_search_provider,
        enable_web_search=enable_web,
        answer_style=answer_style,
        show_local_sources=show_local,
        retriever_k=retriever_k,
        retrieval_score_threshold=threshold,
        **web_search_overrides,
    )
    return SidebarState(
        settings=settings,
        uploaded_files=uploaded_files or [],
        build_clicked=build_clicked,
        reset_clicked=reset_clicked,
    )


def _render_web_provider_fields(provider: str, settings: AppSettings) -> dict[str, str]:
    overrides: dict[str, str] = {}
    key_field = WEB_SEARCH_API_KEY_FIELDS.get(provider)

    if provider == "duckduckgo":
        st.caption("DuckDuckGo Instant Answer does not require an API key.")
        return overrides

    if provider == "custom":
        custom_provider = _plain_input(
            "Custom provider",
            "custom_web_search_provider",
            settings.custom_web_search_provider,
        )
        custom_endpoint = _plain_input(
            "Custom search endpoint",
            "custom_web_search_endpoint",
            settings.custom_web_search_endpoint,
        )
        custom_key = _secret_input(
            "Web search API key",
            "custom_web_search_api_key",
            settings.custom_web_search_api_key,
        )
        overrides["custom_web_search_provider"] = custom_provider.strip()
        overrides["custom_web_search_endpoint"] = custom_endpoint.strip()
        overrides["custom_web_search_api_key"] = custom_key.strip()
        return overrides

    if key_field:
        api_key = _secret_input(
            "Web search API key",
            key_field,
            str(getattr(settings, key_field, "") or ""),
        )
        overrides[key_field] = api_key.strip()

    if provider == "google_cse":
        cse_id = _plain_input("Google CSE ID", "google_cse_id", settings.google_cse_id)
        overrides["google_cse_id"] = cse_id.strip()

    return overrides


def _secret_input(label: str, state_key: str, default: str) -> str:
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    return st.text_input(label, type="password", key=state_key)


def _plain_input(label: str, state_key: str, default: str) -> str:
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    return st.text_input(label, key=state_key)


def _is_valid_hf_model_id(model_id: str) -> bool:
    parts = model_id.strip().split("/")
    if not 1 <= len(parts) <= 2:
        return False
    return all(_is_valid_hf_model_part(part) for part in parts)


def _is_valid_hf_model_part(value: str) -> bool:
    if not value or len(value) > 96:
        return False
    if value[0] in {"-", "."} or value[-1] in {"-", "."}:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return all(char in allowed for char in value)
