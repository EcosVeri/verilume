"""Document and vector-store statistics."""

from __future__ import annotations

from pathlib import Path

import chromadb

from verilume.ingest import SUPPORTED_EXTENSIONS, load_manifest, supported_files
from verilume.settings import AppSettings


def collect_document_stats(settings: AppSettings) -> dict[str, int]:
    files = supported_files(settings.docs_dir)
    manifest = load_manifest(settings.manifest_path)
    pdf_pages = 0
    chunks_indexed = 0
    file_types = len({path.suffix.lower() for path in files})

    for path in files:
        key = str(path)
        if key in manifest:
            pdf_pages += int(manifest[key].get("pdf_pages") or 0)
            chunks_indexed += int(manifest[key].get("chunks") or 0)
        elif path.suffix.lower() == ".pdf":
            pdf_pages += _count_pdf_pages(path)

    chunks_indexed = max(chunks_indexed, _persisted_collection_count(settings))

    return {
        "uploaded_documents": len(files),
        "pdf_pages": pdf_pages,
        "chunks_indexed": chunks_indexed,
        "file_types": file_types,
        "supported_types": len(SUPPORTED_EXTENSIONS),
    }


def _persisted_collection_count(settings: AppSettings) -> int:
    client = None
    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return int(collection.count())
    except Exception:
        return 0
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            try:
                clear_cache = getattr(client, "clear_system_cache", None)
                if callable(clear_cache):
                    clear_cache()
            except Exception:
                pass


def _count_pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return 0
