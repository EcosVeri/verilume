"""Streamlit entrypoint for Verilume."""

from __future__ import annotations

import logging

import streamlit as st

from verilume.ingest import DocumentIngestor, save_uploaded_file
from verilume.rag import get_rag_service
from verilume.settings import AppSettings, ensure_app_dirs
from verilume.ui.chat import render_chat
from verilume.ui.header import render_header
from verilume.ui.sidebar import SidebarState, render_sidebar
from verilume.ui.styles import inject_styles
from verilume.utils.document_stats import collect_document_stats
from verilume.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
ACTIVE_SETTINGS_KEY = "_verilume_active_settings"


def main() -> None:
    configure_logging()
    base_settings = AppSettings.from_env()
    ensure_app_dirs(base_settings)
    st.set_page_config(
        page_title=base_settings.app_title,
        page_icon=base_settings.app_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    if not _password_ok(base_settings):
        return

    stats = _collect_document_stats_cached(base_settings)
    sidebar = render_sidebar(base_settings, stats)
    _clear_rag_cache_if_settings_changed(sidebar.settings)
    _handle_ingestion(sidebar)

    stats = _collect_document_stats_cached(sidebar.settings)
    render_header(sidebar.settings, stats)
    _render_metrics(stats)
    _render_empty_document_state(stats)
    render_chat(sidebar.settings)


def _password_ok(settings: AppSettings) -> bool:
    if not settings.app_password:
        return True
    provided = st.text_input("App password", type="password")
    if provided == settings.app_password:
        return True
    if provided:
        st.error("Incorrect password.")
    return False


def _handle_ingestion(sidebar: SidebarState) -> None:
    if not (sidebar.build_clicked or sidebar.reset_clicked):
        return

    settings = sidebar.settings
    for uploaded_file in sidebar.uploaded_files:
        save_uploaded_file(uploaded_file.name, uploaded_file.getvalue(), settings.docs_dir)

    progress = st.progress(0)
    caption = st.empty()

    def update(label: str, current: int, total: int) -> None:
        if total <= 0:
            progress.progress(1.0)
        else:
            progress.progress(min(1.0, current / total))
        caption.caption(f"{label}: {current}/{total}")

    with st.status("Building knowledge base", expanded=True) as status:
        try:
            result = DocumentIngestor(settings).ingest(reset=sidebar.reset_clicked, progress=update)
            if result.errors:
                for error in result.errors:
                    st.warning(error)
            st.write(
                f"Indexed {result.chunks_indexed} chunks from {result.files_indexed} files; "
                f"skipped {result.files_skipped} unchanged files."
            )
            get_rag_service.cache_clear()
            _collect_document_stats_cached.clear()
            status.update(label="Knowledge base ready", state="complete")
        except Exception:
            LOGGER.exception("Knowledge base build failed.")
            st.error(
                "The knowledge base build failed. Please check the terminal logs and try again."
            )
            status.update(label="Knowledge base build failed", state="error")


def _clear_rag_cache_if_settings_changed(settings: AppSettings) -> None:
    previous_settings = st.session_state.get(ACTIVE_SETTINGS_KEY)
    if previous_settings == settings:
        return

    get_rag_service.cache_clear()
    st.session_state[ACTIVE_SETTINGS_KEY] = settings


@st.cache_data(ttl=60)
def _collect_document_stats_cached(settings: AppSettings) -> dict[str, int]:
    return collect_document_stats(settings)


def _render_metrics(stats: dict[str, int]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Uploaded documents", stats.get("uploaded_documents", 0))
    col2.metric("PDF pages", stats.get("pdf_pages", 0))
    col3.metric("Chunks indexed", stats.get("chunks_indexed", 0))


def _render_empty_document_state(stats: dict[str, int]) -> None:
    if stats.get("uploaded_documents", 0) > 0:
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


if __name__ == "__main__":
    main()
