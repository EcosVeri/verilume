"""Local document ingestion into Chroma."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from verilume.core.embeddings import EmbeddingService
from verilume.core.retrieval import ChromaRetriever
from verilume.core.schemas import DocumentChunk, IngestResult
from verilume.settings import AppSettings, ensure_app_dirs

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".csv", ".docx"}
ProgressCallback = Callable[[str, int, int], None]


class ReadonlyChromaError(RuntimeError):
    """Raised when Chroma's SQLite store rejects writes."""


def save_uploaded_file(file_name: str, data: bytes, docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    clean_name = Path(file_name).name
    target = docs_dir / clean_name
    target.write_bytes(data)
    return target


def supported_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        return []
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=20)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_manifest(path: Path, manifest: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_pdf(path: Path) -> tuple[list[tuple[int | None, str]], int]:
    reader = PdfReader(str(path))
    pages: list[tuple[int | None, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((index, text))
    return pages, len(reader.pages)


def _extract_docx(path: Path) -> tuple[list[tuple[int | None, str]], int]:
    document = Document(str(path))
    parts: list[str] = []
    parts.extend(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    for table in document.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if values:
                parts.append(" | ".join(values))
    return [(None, "\n".join(parts))], 0


def extract_pages(path: Path) -> tuple[list[tuple[int | None, str]], int]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    return [(None, _read_text_file(path))], 0


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_sliding_chunks(paragraph, chunk_size, chunk_overlap))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk.strip()]


def _sliding_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += step
    return chunks


def _parse_file(path: Path, digest: str, settings: AppSettings) -> tuple[Path, list[DocumentChunk], int]:
    pages, pdf_pages = extract_pages(path)
    chunks: list[DocumentChunk] = []
    for page, text in pages:
        for chunk in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunk_index = len(chunks)
            chunk_id = f"{digest}-{page or 0}-{chunk_index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk,
                    source_path=path,
                    document=path.name,
                    page=page,
                    chunk_index=chunk_index,
                    file_hash=digest,
                    metadata={
                        "extension": path.suffix.lower(),
                        "source_path": str(path),
                    },
                )
            )
    return path, chunks, pdf_pages


class DocumentIngestor:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingService(settings.embed_model, settings.embed_device)
        self.retriever = ChromaRetriever(
            settings.chroma_dir,
            settings.collection_name,
            self.embeddings,
        )

    def ingest(
        self,
        reset: bool | None = None,
        progress: ProgressCallback | None = None,
    ) -> IngestResult:
        reset = self.settings.reset_db if reset is None else reset
        try:
            return self._ingest_once(reset=reset, progress=progress, force_all=False)
        except ReadonlyChromaError:
            LOGGER.warning("Chroma reported a readonly database; rebuilding and retrying once.")
            if progress:
                progress("Repairing Chroma database", 0, 1)
            self._reset()
            if progress:
                progress("Repairing Chroma database", 1, 1)
            return self._ingest_once(reset=False, progress=progress, force_all=True)

    def _ingest_once(
        self,
        reset: bool,
        progress: ProgressCallback | None,
        force_all: bool,
    ) -> IngestResult:
        ensure_app_dirs(self.settings)
        if reset:
            self._reset()

        files = supported_files(self.settings.docs_dir)
        if not files:
            manifest = load_manifest(self.settings.manifest_path)
            self._remove_missing_documents(manifest)
            write_manifest(self.settings.manifest_path, manifest)
            return IngestResult(0, 0, 0, 0, 0)

        manifest = {} if force_all else load_manifest(self.settings.manifest_path)
        self._remove_missing_documents(manifest)
        manifest = {path: value for path, value in manifest.items() if Path(path).exists()}
        force_reindex = force_all or self.retriever.count() == 0
        pending: list[tuple[Path, str]] = []
        skipped = 0

        for path in files:
            digest = file_hash(path)
            key = str(path)
            entry = manifest.get(key, {})
            if not force_reindex and entry.get("hash") == digest:
                skipped += 1
                continue
            pending.append((path, digest))

        all_chunks: list[DocumentChunk] = []
        pdf_pages = 0
        errors: list[str] = []
        if progress:
            progress("Parsing documents", 0, len(pending))

        with ThreadPoolExecutor(max_workers=max(1, self.settings.max_workers)) as pool:
            futures = {
                pool.submit(_parse_file, path, digest, self.settings): (path, digest)
                for path, digest in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                path, digest = futures[future]
                try:
                    parsed_path, chunks, pages = future.result()
                    self.retriever.delete_document(str(parsed_path))
                    all_chunks.extend(chunks)
                    pdf_pages += pages
                    manifest[str(parsed_path)] = {
                        "hash": digest,
                        "chunks": len(chunks),
                        "pdf_pages": pages,
                    }
                except Exception as exc:
                    message = f"{path.name}: {exc}"
                    LOGGER.exception("Failed to ingest %s", path)
                    errors.append(message)
                if progress:
                    progress("Parsing documents", done, len(pending))

        indexed_chunks = self._index_chunks(all_chunks, progress)
        write_manifest(self.settings.manifest_path, manifest)
        return IngestResult(
            files_seen=len(files),
            files_indexed=len(pending) - len(errors),
            files_skipped=skipped,
            chunks_indexed=indexed_chunks,
            pdf_pages=pdf_pages,
            errors=errors,
        )

    def _index_chunks(
        self,
        chunks: list[DocumentChunk],
        progress: ProgressCallback | None,
    ) -> int:
        if not chunks:
            return 0

        indexed = 0
        total = len(chunks)
        for start in range(0, total, self.settings.batch_size):
            batch = chunks[start : start + self.settings.batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embeddings.embed_documents(texts, batch_size=self.settings.batch_size)
            try:
                self.retriever.add_chunks(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=texts,
                    metadatas=[_metadata(chunk) for chunk in batch],
                    embeddings=embeddings,
                )
            except Exception as exc:
                if _is_readonly_chroma_error(exc):
                    raise ReadonlyChromaError(str(exc)) from exc
                raise
            indexed += len(batch)
            if progress:
                progress("Embedding chunks", indexed, total)
        return indexed

    def _reset(self) -> None:
        self.retriever.reconnect()
        if self.settings.manifest_path.exists():
            self.settings.manifest_path.unlink()
        if self.settings.chroma_dir.exists():
            _make_tree_writable(self.settings.chroma_dir)
            shutil.rmtree(self.settings.chroma_dir, ignore_errors=True)
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _make_tree_writable(self.settings.chroma_dir)
        self.retriever.reconnect()
        self.retriever.reset()

    def _remove_missing_documents(self, manifest: dict[str, dict]) -> None:
        missing_paths = [stored_path for stored_path in manifest if not Path(stored_path).exists()]
        for stored_path in missing_paths:
            self.retriever.delete_document(stored_path)
            manifest.pop(stored_path, None)


def _metadata(chunk: DocumentChunk) -> dict:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "document": chunk.document,
            "source_path": str(chunk.source_path),
            "page": chunk.page or 0,
            "chunk_index": chunk.chunk_index,
            "file_hash": chunk.file_hash,
        }
    )
    return metadata


def _is_readonly_chroma_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "readonly database" in text or "read-only database" in text


def _make_tree_writable(path: Path) -> None:
    if not path.exists():
        return
    for item in [path, *path.rglob("*")]:
        try:
            mode = item.stat().st_mode
            item.chmod(mode | 0o200)
        except OSError:
            pass
