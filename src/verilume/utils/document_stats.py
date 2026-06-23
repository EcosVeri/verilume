"""Document and vector-store statistics."""

from __future__ import annotations

from pathlib import Path

from verilume.ingest import SUPPORTED_EXTENSIONS, load_manifest, supported_files
from verilume.settings import AppSettings


def collect_document_stats(settings: AppSettings) -> dict[str, int]:
    files = supported_files(settings.docs_dir)
    manifest = load_manifest(settings.manifest_path)
    pdf_pages = 0
    chunks_indexed = 0

    for path in files:
        key = str(path)
        if key in manifest:
            pdf_pages += int(manifest[key].get("pdf_pages") or 0)
            chunks_indexed += int(manifest[key].get("chunks") or 0)
        elif path.suffix.lower() == ".pdf":
            pdf_pages += _count_pdf_pages(path)

    return {
        "uploaded_documents": len(files),
        "pdf_pages": pdf_pages,
        "chunks_indexed": chunks_indexed,
        "supported_types": len(SUPPORTED_EXTENSIONS),
    }


def _count_pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return 0
