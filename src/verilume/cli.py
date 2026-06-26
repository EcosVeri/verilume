"""Command line interface for Verilume."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from verilume.ingest import DocumentIngestor
from verilume.rag import get_rag_service
from verilume.settings import AppSettings
from verilume.utils.document_stats import collect_document_stats
from verilume.utils.logging import configure_logging

STREAMLIT_HOST = "127.0.0.1"
STREAMLIT_BROWSER_ADDRESS = "localhost"
STREAMLIT_PORT = 8501
STREAMLIT_PORT_ENV = "VERILUME_PORT"
STREAMLIT_PORT_ATTEMPTS = 3


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
    streamlit_args = _streamlit_cli_args(app_path, port=_resolve_streamlit_port())
    if getattr(sys, "frozen", False):
        _patch_streamlit_for_frozen_bundle()

        from streamlit.web.cli import main as streamlit_main

        sys.argv = [
            "streamlit",
            *streamlit_args,
            "--global.developmentMode",
            "false",
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
                *streamlit_args,
            ],
            check=False,
        )
        return process.returncode
    except KeyboardInterrupt:
        return 130


def _streamlit_cli_args(app_path: object, *, port: int = STREAMLIT_PORT) -> list[str]:
    return [
        "run",
        str(app_path),
        "--server.address",
        STREAMLIT_HOST,
        "--server.port",
        str(port),
        "--browser.serverAddress",
        STREAMLIT_BROWSER_ADDRESS,
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]


def _resolve_streamlit_port() -> int:
    configured = os.environ.get(STREAMLIT_PORT_ENV, "").strip()
    if configured:
        try:
            return int(configured)
        except ValueError:
            print(
                f"Ignoring invalid {STREAMLIT_PORT_ENV}={configured!r}; "
                f"using an available port from {STREAMLIT_PORT}.",
                file=sys.stderr,
            )
    available_port = _first_available_port(
        STREAMLIT_HOST,
        STREAMLIT_PORT,
        attempts=STREAMLIT_PORT_ATTEMPTS,
    )
    if available_port is not None:
        return available_port
    return _force_reclaim_streamlit_port(STREAMLIT_HOST, STREAMLIT_PORT)


def _first_available_port(host: str, start_port: int, *, attempts: int = 20) -> int | None:
    for port in range(start_port, start_port + max(1, attempts)):
        if _port_is_available(host, port):
            return port
    return None


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _force_reclaim_streamlit_port(host: str, port: int) -> int:
    pids = _listening_pids(port)
    if not pids:
        print(
            f"Ports {STREAMLIT_PORT}-{STREAMLIT_PORT + STREAMLIT_PORT_ATTEMPTS - 1} "
            f"are busy; forcing Verilume to try port {port}.",
            file=sys.stderr,
        )
        return port

    pid_list = ", ".join(str(pid) for pid in pids)
    print(
        f"Ports {STREAMLIT_PORT}-{STREAMLIT_PORT + STREAMLIT_PORT_ATTEMPTS - 1} "
        f"are busy. Stopping process(es) {pid_list} on port {port} so Verilume can launch.",
        file=sys.stderr,
    )
    _terminate_processes(pids)
    if not _wait_for_port(host, port, available=True, timeout_seconds=5.0):
        print(
            f"Port {port} did not release after SIGTERM; forcing shutdown for process(es) {pid_list}.",
            file=sys.stderr,
        )
        _kill_processes(pids)
        _wait_for_port(host, port, available=True, timeout_seconds=3.0)
    return port


def _listening_pids(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    current_pid = os.getpid()
    return [pid for pid in pids if pid != current_pid]


def _terminate_processes(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"Could not stop process {pid}: {exc}", file=sys.stderr)


def _kill_processes(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"Could not force-stop process {pid}: {exc}", file=sys.stderr)


def _wait_for_port(
    host: str,
    port: int,
    *,
    available: bool,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while time.monotonic() <= deadline:
        if _port_is_available(host, port) is available:
            return True
        time.sleep(0.1)
    return _port_is_available(host, port) is available


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
