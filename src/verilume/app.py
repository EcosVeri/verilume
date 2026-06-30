"""Streamlit entrypoint for Verilume."""

from __future__ import annotations

import logging
import secrets

import streamlit as st

from verilume.ingest import DocumentIngestor, removable_documents, remove_documents, save_uploaded_file
from verilume.rag import get_rag_service
from verilume.settings import AppSettings, ensure_app_dirs
from verilume.ui.chat import render_chat
from verilume.ui.dashboard import (
    recent_activity_from_messages,
    render_dashboard,
    render_empty_document_state,
)
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
    inject_styles(base_settings.appearance)

    if not _password_ok(base_settings):
        return

    stats = _collect_document_stats_cached(base_settings)
    sidebar = render_sidebar(base_settings, stats)
    _clear_rag_cache_if_settings_changed(sidebar.settings)
    _handle_document_removal(sidebar)
    _handle_ingestion(sidebar)

    # Re-fetch stats only when the docs dir may have changed.
    if sidebar.settings.docs_dir != base_settings.docs_dir:
        stats = _collect_document_stats_cached(sidebar.settings)
    recent_documents = removable_documents(sidebar.settings.docs_dir)[-4:][::-1]
    recent_activity = recent_activity_from_messages(st.session_state.get("messages", []))

    header_slot = st.container()
    dashboard_slot = st.container()
    document_state_slot = st.container()
    chat_slot = st.container()

    with header_slot:
        render_header(sidebar.settings, stats)
    with dashboard_slot:
        # suggested_prompts=None tells render_dashboard to generate them lazily
        # after the collapsed-check so they are skipped entirely when collapsed.
        render_dashboard(
            sidebar.settings,
            stats,
            recent_documents,
            recent_activity,
            suggested_prompts=None,
        )
    with document_state_slot:
        render_empty_document_state(stats)
    with chat_slot:
        render_chat(sidebar.settings)


_MAX_PASSWORD_ATTEMPTS = 5


def _password_ok(settings: AppSettings) -> bool:
    if not settings.app_password:
        return True

    attempts = st.session_state.setdefault("_pw_attempts", 0)
    if attempts >= _MAX_PASSWORD_ATTEMPTS:
        st.error("Too many incorrect attempts. Please restart the app.")
        return False

    provided = st.text_input("App password", type="password")
    if not provided:
        return False

    # Constant-time comparison prevents timing side-channel attacks.
    if secrets.compare_digest(provided, settings.app_password):
        st.session_state["_pw_attempts"] = 0
        return True

    st.session_state["_pw_attempts"] = attempts + 1
    st.error("Incorrect password.")
    return False


def _handle_ingestion(sidebar: SidebarState) -> None:
    if not (sidebar.build_clicked or sidebar.reset_clicked):
        return

    settings = sidebar.settings
    for uploaded_file in sidebar.uploaded_files:
        save_uploaded_file(uploaded_file.name, uploaded_file.getvalue(), settings.docs_dir)

    _release_rag_retriever(settings)

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
        except Exception as exc:
            LOGGER.exception("Knowledge base build failed.")
            get_rag_service.cache_clear()
            _collect_document_stats_cached.clear()
            st.error(f"Knowledge base build failed: {_safe_error_category(exc)}")
            status.update(label="Knowledge base build failed", state="error")


def _handle_document_removal(sidebar: SidebarState) -> None:
    if not sidebar.remove_clicked or not sidebar.remove_documents:
        return

    _release_rag_retriever(sidebar.settings)

    with st.status("Removing selected documents", expanded=True) as status:
        try:
            removed = remove_documents(sidebar.settings, sidebar.remove_documents)
            get_rag_service.cache_clear()
            _collect_document_stats_cached.clear()
            if removed:
                st.write(f"Removed {len(removed)} document(s): {', '.join(removed)}")
            else:
                st.warning("No selected documents could be removed.")
            status.update(label="Document removal complete", state="complete")
        except Exception as exc:
            LOGGER.exception("Document removal failed.")
            get_rag_service.cache_clear()
            _collect_document_stats_cached.clear()
            st.error(f"Document removal failed: {_safe_error_category(exc)}")
            status.update(label="Document removal failed", state="error")


def _clear_rag_cache_if_settings_changed(settings: AppSettings) -> None:
    previous_settings = st.session_state.get(ACTIVE_SETTINGS_KEY)
    if previous_settings == settings:
        return

    get_rag_service.cache_clear()
    st.session_state[ACTIVE_SETTINGS_KEY] = settings


def _release_rag_retriever(settings: AppSettings) -> None:
    try:
        service = get_rag_service(settings)
    except Exception:
        get_rag_service.cache_clear()
        return

    close = getattr(service, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            LOGGER.debug("Failed to close cached RAG service before local store mutation.", exc_info=True)
    else:
        retriever = getattr(service, "retriever", None)
        close_retriever = getattr(retriever, "close", None)
        if callable(close_retriever):
            try:
                close_retriever(clear_system_cache=True)
            except TypeError:
                close_retriever()
            except Exception:
                LOGGER.debug(
                    "Failed to close cached retriever before local store mutation.",
                    exc_info=True,
                )
    get_rag_service.cache_clear()



@st.cache_data(ttl=60)
def _collect_document_stats_cached(settings: AppSettings) -> dict[str, int]:
    return collect_document_stats(settings)


def _safe_error_category(exc: BaseException) -> str:
    """Return a user-friendly error category without leaking internal paths or stack frames."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "permission" in msg or "access" in msg:
        return "Permission denied. Check folder access rights."
    if "disk" in msg or "space" in msg or "quota" in msg or "no space" in msg:
        return "Disk full or quota exceeded."
    if "encode" in msg or "decode" in msg or "unicode" in msg or "utf" in msg:
        return "Unsupported file encoding."
    if "timeout" in msg or "timed out" in msg:
        return "Operation timed out. Try again."
    if "memory" in msg or "oom" in msg:
        return "Out of memory. Try indexing fewer files at once."
    if "corrupt" in msg or "invalid" in msg or "parse" in msg:
        return "File appears corrupt or unreadable."
    if "network" in msg or "connection" in msg or "refused" in msg:
        return "Network error. Check your connection."
    return f"Unexpected error ({name}). See terminal logs for details."


if __name__ == "__main__":
    main()
