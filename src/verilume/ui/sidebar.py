"""Sidebar controls for Verilume."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import streamlit as st

from verilume.ingest import removable_documents, supported_extensions
from verilume.settings import (
    ANSWER_STYLE_CHOICES,
    DEFAULT_HF_MODEL_CHOICES,
    DEFAULT_OLLAMA_MODEL_CHOICES,
    GENERATION_BACKEND_LABELS,
    SEARCH_MODE_CHOICES,
    WEB_SEARCH_PROVIDER_LABELS,
    AppSettings,
    save_user_config,
)


@dataclass(slots=True)
class SidebarState:
    settings: AppSettings
    uploaded_files: list[Any]
    build_clicked: bool
    remove_documents: list[str]
    remove_clicked: bool
    reset_clicked: bool


def render_sidebar(
    base_settings: AppSettings,
    stats: dict[str, int],
) -> SidebarState:
    with st.sidebar:
        st.markdown(_sidebar_status_html(stats), unsafe_allow_html=True)

        uploaded_files: list[Any] = []
        build_clicked = False
        remove_clicked = False
        reset_clicked = False

        overrides: dict[str, Any] = {}

        # -------------------------
        # Models
        # -------------------------

        models_expanded = _section_expanded("models", default=False)

        with st.expander("🧠 Models · Configure AI", expanded=models_expanded):
            backend_options = list(GENERATION_BACKEND_LABELS.keys())
            backend_labels = [GENERATION_BACKEND_LABELS[key] for key in backend_options]

            current_backend = base_settings.generation_backend
            backend_index = (
                backend_options.index(current_backend) if current_backend in backend_options else 0
            )

            selected_backend_label = st.selectbox(
                "Generation backend",
                options=backend_labels,
                index=backend_index,
            )

            selected_backend = backend_options[backend_labels.index(selected_backend_label)]
            overrides["generation_backend"] = selected_backend

            if selected_backend == "huggingface":
                overrides.update(_render_huggingface_settings(base_settings))
            else:
                overrides.update(_render_ollama_settings(base_settings))

        # -------------------------
        # Search
        # -------------------------

        search_expanded = _section_expanded("search", default=False)

        with st.expander("🔎 Search · Choose sources", expanded=search_expanded):
            search_mode = st.radio(
                "Search mode",
                options=list(SEARCH_MODE_CHOICES),
                index=list(SEARCH_MODE_CHOICES).index(base_settings.search_mode)
                if base_settings.search_mode in SEARCH_MODE_CHOICES
                else 0,
            )

            enable_web_search = st.toggle(
                "Web search",
                value=base_settings.enable_web_search,
            )

            show_benchmark_controls = st.checkbox(
                "Show benchmark controls",
                value=base_settings.benchmark_mode,
                help="Developer comparison mode for retrieval and generation strategies.",
            )
            benchmark_mode = base_settings.benchmark_mode
            if show_benchmark_controls:
                benchmark_mode = st.toggle(
                    "Benchmark mode",
                    value=base_settings.benchmark_mode,
                    help="Compare Full, Local Only, AI Only, and Web Only strategies.",
                )

            provider_keys = list(WEB_SEARCH_PROVIDER_LABELS.keys())
            provider_labels = [WEB_SEARCH_PROVIDER_LABELS[key] for key in provider_keys]

            current_provider = base_settings.web_search_provider
            provider_index = (
                provider_keys.index(current_provider) if current_provider in provider_keys else 0
            )

            selected_provider_label = st.selectbox(
                "Web search provider",
                options=provider_labels,
                index=provider_index,
            )

            selected_provider = provider_keys[provider_labels.index(selected_provider_label)]

            overrides["search_mode"] = search_mode
            overrides["enable_web_search"] = enable_web_search
            overrides["benchmark_mode"] = benchmark_mode
            overrides["web_search_provider"] = selected_provider

            overrides.update(_render_web_provider_settings(base_settings, selected_provider))

            web_search_max_results = st.slider(
                "Web results",
                min_value=1,
                max_value=10,
                value=base_settings.web_search_max_results,
            )

            web_search_timeout_seconds = st.slider(
                "Web timeout seconds",
                min_value=5,
                max_value=60,
                value=int(base_settings.web_search_timeout_seconds),
            )

            overrides["web_search_max_results"] = web_search_max_results
            overrides["web_search_timeout_seconds"] = float(web_search_timeout_seconds)

        # -------------------------
        # Documents
        # -------------------------

        docs_expanded = _section_expanded("documents", default=False)

        with st.expander("📚 Knowledge Base · Upload + Browse", expanded=docs_expanded):
            st.markdown(
                """
<div class="veri-upload-info">
  <strong>Build your knowledge base</strong>
  <div class="veri-upload-types">
    <span>📄 PDF</span>
    <span>📘 DOCX</span>
    <span>📑 PPTX</span>
    <span>📊 CSV</span>
    <span>🖼 Images</span>
    <span>✍ TXT / MD</span>
  </div>
  <div class="veri-upload-limit">Maximum size: 200 MB per file</div>
</div>
                """,
                unsafe_allow_html=True,
            )

            upload_types = sorted(extension.lstrip(".") for extension in supported_extensions())
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=upload_types,
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            build_clicked = st.button(
                "🟨 Build Knowledge Base",
                type="primary",
                use_container_width=True,
            )
            reset_clicked = st.button(
                "Reset DB",
                use_container_width=True,
            )

            existing_documents = removable_documents(base_settings.docs_dir)
            _render_document_explorer(existing_documents)
            remove_documents_selected: list[str] = []
            if existing_documents:
                remove_documents_selected = st.multiselect(
                    "Indexed documents",
                    options=existing_documents,
                    help="Remove selected documents from the local files and vector database.",
                )
                remove_clicked = st.button(
                    "Remove selected",
                    use_container_width=True,
                    disabled=not remove_documents_selected,
                )
            else:
                st.caption("No indexed documents available to remove.")

            st.caption("Library snapshot")

            col_1, col_2 = st.columns(2)
            col_1.metric("Documents", stats.get("uploaded_documents", 0))
            col_2.metric("PDF pages", stats.get("pdf_pages", 0))

            col_3, col_4 = st.columns(2)
            col_3.metric("Chunks", stats.get("chunks_indexed", 0))
            col_4.metric("Types", stats.get("file_types", 0))

        # -------------------------
        # Retrieval
        # -------------------------

        retrieval_expanded = _section_expanded("retrieval", default=False)

        with st.expander("⚙ Retrieval · Ranking", expanded=retrieval_expanded):
            answer_style = st.radio(
                "Answer style",
                options=list(ANSWER_STYLE_CHOICES),
                index=list(ANSWER_STYLE_CHOICES).index(base_settings.answer_style)
                if base_settings.answer_style in ANSWER_STYLE_CHOICES
                else 1,
                horizontal=True,
            )

            show_local_sources = st.toggle(
                "Show local citations",
                value=base_settings.show_local_sources,
            )

            retriever_k = st.slider(
                "Retriever K",
                min_value=2,
                max_value=12,
                value=base_settings.retriever_k,
            )

            retrieval_score_threshold = st.slider(
                "Score threshold",
                min_value=0.0,
                max_value=0.95,
                value=float(base_settings.retrieval_score_threshold),
                step=0.05,
            )

            enable_query_rewrite = st.toggle(
                "Query rewriting",
                value=base_settings.enable_query_rewrite,
            )

            overrides["answer_style"] = answer_style
            overrides["show_local_sources"] = show_local_sources
            overrides["retriever_k"] = retriever_k
            overrides["retrieval_score_threshold"] = retrieval_score_threshold
            overrides["enable_query_rewrite"] = enable_query_rewrite

        settings = base_settings.with_overrides(**overrides)

        if st.button("Save settings", type="primary", use_container_width=True):
            save_user_config(settings)
            st.success("Settings saved.")
            st.rerun()

        return SidebarState(
            settings=settings,
            uploaded_files=uploaded_files or [],
            build_clicked=build_clicked,
            remove_documents=remove_documents_selected,
            remove_clicked=remove_clicked,
            reset_clicked=reset_clicked,
        )


def _section_expanded(section: str, default: bool) -> bool:
    focused = st.session_state.pop("focus_sidebar_section", None)

    if focused == section:
        return True

    return default


def _sidebar_status_html(stats: dict[str, int]) -> str:
    documents = int(stats.get("uploaded_documents", 0) or 0)
    pages = int(stats.get("pdf_pages", 0) or 0)
    chunks = int(stats.get("chunks_indexed", 0) or 0)
    return (
        '<div class="veri-sidebar-panel">'
        '<div class="veri-sidebar-brandline">VERILUME</div>'
        '<div class="veri-sidebar-title">Verilume Studio</div>'
        '<div class="veri-sidebar-subtitle">Local-first AI Research Assistant</div>'
        '<div class="veri-sidebar-version">Version 1.0</div>'
        '<div class="veri-sidebar-status">'
        f'<div class="veri-sidebar-stat"><span>📄</span><strong>{documents}</strong><em>Docs</em></div>'
        f'<div class="veri-sidebar-stat"><span>📚</span><strong>{pages}</strong><em>Pages</em></div>'
        f'<div class="veri-sidebar-stat"><span>🧩</span><strong>{chunks}</strong><em>Chunks</em></div>'
        "</div>"
        "</div>"
    )


def _render_document_explorer(documents: list[str]) -> None:
    if not documents:
        return
    items = "".join(
        f'<div class="veri-document-row"><span>📄</span><strong>{escape(document)}</strong></div>'
        for document in documents[:6]
    )
    st.markdown(
        f"""
<div class="veri-document-explorer">
  <div class="veri-document-explorer-title">Document Explorer</div>
  {items}
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_huggingface_settings(base_settings: AppSettings) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    hf_token = st.text_input(
        "Hugging Face token",
        value=base_settings.hf_token,
        type="password",
    )

    model_options = DEFAULT_HF_MODEL_CHOICES

    current_model = base_settings.hf_llm_model
    model_index = model_options.index(current_model) if current_model in model_options else 0

    selected_model = st.selectbox(
        "Hugging Face model",
        options=model_options,
        index=model_index,
    )

    custom_model = st.text_input(
        "Custom Hugging Face model",
        value="" if current_model in model_options else current_model,
        placeholder="organisation/model-name",
    )

    active_model = custom_model.strip() or selected_model

    hf_provider = st.text_input(
        "HF provider",
        value=base_settings.hf_provider,
        help="Use 'auto' unless you know the provider name.",
    )

    hf_temperature = st.slider(
        "HF temperature",
        min_value=0.0,
        max_value=1.5,
        value=float(base_settings.hf_temperature),
        step=0.05,
    )

    hf_max_new_tokens = st.slider(
        "HF max tokens",
        min_value=128,
        max_value=4096,
        value=int(base_settings.hf_max_new_tokens),
        step=64,
    )

    overrides["hf_token"] = hf_token
    overrides["hf_llm_model"] = active_model
    overrides["hf_provider"] = hf_provider
    overrides["hf_temperature"] = hf_temperature
    overrides["hf_max_new_tokens"] = hf_max_new_tokens

    _render_active_model_panel("Active Hugging Face model", active_model)

    return overrides


def _render_ollama_settings(base_settings: AppSettings) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    ollama_base_url = st.text_input(
        "Ollama base URL",
        value=base_settings.ollama_base_url,
        placeholder="http://localhost:11434",
    )

    model_options = DEFAULT_OLLAMA_MODEL_CHOICES

    current_model = base_settings.ollama_model
    model_index = model_options.index(current_model) if current_model in model_options else 0

    selected_model = st.selectbox(
        "Ollama model",
        options=model_options,
        index=model_index,
    )

    custom_model = st.text_input(
        "Custom Ollama model",
        value="" if current_model in model_options else current_model,
        placeholder="llama3.1:8b",
    )

    active_model = custom_model.strip() or selected_model

    ollama_temperature = st.slider(
        "Ollama temperature",
        min_value=0.0,
        max_value=1.5,
        value=float(base_settings.ollama_temperature),
        step=0.05,
    )

    ollama_num_predict = st.slider(
        "Ollama max tokens",
        min_value=128,
        max_value=4096,
        value=int(base_settings.ollama_num_predict),
        step=64,
    )

    ollama_timeout_seconds = st.slider(
        "Ollama timeout seconds",
        min_value=10,
        max_value=300,
        value=int(base_settings.ollama_timeout_seconds),
        step=10,
    )

    overrides["ollama_base_url"] = ollama_base_url
    overrides["ollama_model"] = active_model
    overrides["ollama_temperature"] = ollama_temperature
    overrides["ollama_num_predict"] = ollama_num_predict
    overrides["ollama_timeout_seconds"] = float(ollama_timeout_seconds)

    _render_active_model_panel("Active Ollama model", active_model)

    return overrides


def _render_active_model_panel(label: str, model: str) -> None:
    st.markdown(
        _active_model_html(label, model),
        unsafe_allow_html=True,
    )


def _active_model_html(label: str, model: str) -> str:
    return (
        '<div class="veri-active-model">'
        f'<div class="veri-active-model-label">{escape(label)}</div>'
        f'<div class="veri-active-model-name">{escape(model)}</div>'
        "</div>"
    )


def _render_web_provider_settings(
    base_settings: AppSettings,
    provider: str,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    if provider == "duckduckgo":
        st.caption("DuckDuckGo does not require an API key.")
        return overrides

    if provider == "tavily":
        overrides["tavily_api_key"] = st.text_input(
            "Web search API key",
            value=base_settings.tavily_api_key,
            type="password",
        )
        return overrides

    if provider == "brave":
        overrides["brave_api_key"] = st.text_input(
            "Brave API key",
            value=base_settings.brave_api_key,
            type="password",
        )
        return overrides

    if provider == "exa":
        overrides["exa_api_key"] = st.text_input(
            "Exa API key",
            value=base_settings.exa_api_key,
            type="password",
        )
        return overrides

    if provider == "serpapi":
        overrides["serpapi_api_key"] = st.text_input(
            "SerpAPI key",
            value=base_settings.serpapi_api_key,
            type="password",
        )
        return overrides

    if provider == "bing":
        overrides["bing_api_key"] = st.text_input(
            "Bing API key",
            value=base_settings.bing_api_key,
            type="password",
        )
        return overrides

    if provider == "google_cse":
        overrides["google_cse_api_key"] = st.text_input(
            "Google CSE API key",
            value=base_settings.google_cse_api_key,
            type="password",
        )

        overrides["google_cse_id"] = st.text_input(
            "Google CSE ID",
            value=base_settings.google_cse_id,
            type="password",
        )

        return overrides

    if provider == "custom":
        overrides["custom_web_search_provider"] = st.text_input(
            "Custom provider name",
            value=base_settings.custom_web_search_provider,
            placeholder="My Search Provider",
        )

        overrides["custom_web_search_endpoint"] = st.text_input(
            "Custom endpoint",
            value=base_settings.custom_web_search_endpoint,
            placeholder="https://api.example.com/search?q={query}&key={api_key}",
        )

        overrides["custom_web_search_api_key"] = st.text_input(
            "Custom API key",
            value=base_settings.custom_web_search_api_key,
            type="password",
        )

        st.caption(
            "Endpoint may use `{query}` and `{api_key}` placeholders. "
            "If `{query}` is absent, Verilume sends `?q=<query>`."
        )

        return overrides

    return overrides
