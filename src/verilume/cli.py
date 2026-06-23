"""Command line interface for Verilume."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from verilume.ingest import DocumentIngestor
from verilume.rag import get_rag_service
from verilume.settings import AppSettings
from verilume.utils.document_stats import collect_document_stats
from verilume.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="verilume")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Launch the Streamlit desktop app")
    ingest_parser = subparsers.add_parser("ingest", help="Build or update the local knowledge base")
    ingest_parser.add_argument("--reset", action="store_true", help="Clear Chroma before ingesting")
    subparsers.add_parser("stats", help="Show document statistics")
    subparsers.add_parser("config", help="Print effective configuration without secrets")
    subparsers.add_parser("doctor", help="Run deployment health checks")

    args = parser.parse_args(argv)
    command = args.command or "run"
    settings = AppSettings.from_env()

    if command == "run":
        return run_streamlit()
    if command == "ingest":
        result = DocumentIngestor(settings).ingest(reset=args.reset)
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0 if not result.errors else 1
    if command == "stats":
        print(json.dumps(collect_document_stats(settings), indent=2))
        return 0
    if command == "config":
        print(json.dumps(settings.public_dict(), indent=2, default=str))
        return 0
    if command == "doctor":
        return run_doctor(settings)

    parser.print_help()
    return 2


def run_streamlit() -> int:
    app_path = files("verilume").joinpath("app.py")
    if getattr(sys, "frozen", False):
        _patch_streamlit_for_frozen_bundle()

        from streamlit.web.cli import main as streamlit_main

        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--global.developmentMode",
            "false",
            "--server.fileWatcherType",
            "none",
        ]
        try:
            return int(streamlit_main() or 0)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 0
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.fileWatcherType",
                "none",
            ],
            check=False,
        )
        return process.returncode
    except KeyboardInterrupt:
        return 130


def _patch_streamlit_for_frozen_bundle() -> None:
    """Make Streamlit serve bundled frontend assets from a PyInstaller .app."""
    from streamlit import config, development, file_util

    static_dir = _find_frozen_resource("streamlit", "static")
    if static_dir is not None:
        file_util.get_static_dir = lambda: str(static_dir)

    config.set_option("global.developmentMode", False)
    development.is_development_mode = False


def _find_frozen_resource(*parts: str) -> Path | None:
    relative_path = Path(*parts)
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / relative_path)

    executable = Path(sys.executable).resolve()
    for parent in (executable.parent, *executable.parents):
        if parent.name == "Contents":
            candidates.extend(
                [
                    parent / "Resources" / relative_path,
                    parent / "Frameworks" / relative_path,
                ]
            )
            break

    candidates.append(Path.cwd() / relative_path)
    return next((path for path in candidates if path.exists()), None)


def run_doctor(settings: AppSettings) -> int:
    stats = collect_document_stats(settings)
    report = {
        "docs_dir_exists": settings.docs_dir.exists(),
        "chroma_dir_exists": settings.chroma_dir.exists(),
        "manifest_exists": settings.manifest_path.exists(),
        "huggingface_token_present": bool(settings.hf_token),
        "web_search_enabled": settings.enable_web_search,
        "web_search_provider": settings.web_search_provider_label(),
        "web_search_provider_configured": settings.web_search_ready(),
        "uploaded_documents": stats["uploaded_documents"],
        "indexed_chunks": stats["chunks_indexed"],
        "retriever_count": get_rag_service(settings).retriever.count(),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
