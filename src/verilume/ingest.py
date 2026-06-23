"""Local document ingestion into Chroma."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from verilume.core.embeddings import EmbeddingService
from verilume.core.retrieval import ChromaRetriever
from verilume.core.schemas import DocumentChunk, IngestResult
from verilume.settings import AppSettings, ensure_app_dirs

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


class FileHandler(ABC):
    """Extract text from one or more file types."""

    extensions: set[str] = set()

    @abstractmethod
    def extract_pages(self, path: Path) -> tuple[list[tuple[int | None, str]], int]:
        """Return extracted text by page and PDF page count when available."""


class PdfHandler(FileHandler):
    extensions = {".pdf"}

    def extract_pages(self, path: Path) -> tuple[list[tuple[int | None, str]], int]:
        return _extract_pdf(path)


class DocxHandler(FileHandler):
    extensions = {".docx"}

    def extract_pages(self, path: Path) -> tuple[list[tuple[int | None, str]], int]:
        return _extract_docx(path)


class TextHandler(FileHandler):
    extensions = {".txt", ".md", ".markdown", ".csv"}

    def extract_pages(self, path: Path) -> tuple[list[tuple[int | None, str]], int]:
        return [(None, _read_text_file(path))], 0


FILE_HANDLERS: dict[str, FileHandler] = {}
SUPPORTED_EXTENSIONS: set[str] = set()


def register_file_handler(handler: FileHandler) -> None:
    for extension in handler.extensions:
        key = extension.lower()
        FILE_HANDLERS[key] = handler
        SUPPORTED_EXTENSIONS.add(key)


def supported_extensions() -> set[str]:
    return set(FILE_HANDLERS)


register_file_handler(PdfHandler())
register_file_handler(DocxHandler())
register_file_handler(TextHandler())


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
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int | None, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((index, text))
    return pages, len(reader.pages)


def _extract_docx(path: Path) -> tuple[list[tuple[int | None, str]], int]:
    from docx import Document

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
    handler = FILE_HANDLERS.get(path.suffix.lower())
    if handler is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return handler.extract_pages(path)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return chunk_text_semantic(text, chunk_size, chunk_overlap)


def chunk_text_semantic(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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
            chunks.extend(_chunk_by_sentences(paragraph, chunk_size, chunk_overlap))
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


def chunk_text_by_paragraphs(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text or "")
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _chunk_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return _sliding_chunks(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if sentence_length > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_length = 0
            chunks.extend(_sliding_chunks(sentence, chunk_size, chunk_overlap))
            continue
        if current and current_length + sentence_length + 1 > chunk_size:
            chunks.append(" ".join(current).strip())
            current, current_length = _overlap_sentences(current, chunk_overlap)
        current.append(sentence)
        current_length += sentence_length + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk.strip()]


def _overlap_sentences(sentences: list[str], chunk_overlap: int) -> tuple[list[str], int]:
    if chunk_overlap <= 0:
        return [], 0
    overlap: list[str] = []
    length = 0
    for sentence in reversed(sentences):
        next_length = length + len(sentence) + 1
        if next_length > chunk_overlap:
            break
        overlap.insert(0, sentence)
        length = next_length
    return overlap, length


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


def _parse_file(
    path: Path, digest: str, settings: AppSettings
) -> tuple[Path, list[DocumentChunk], int]:
    pages, pdf_pages = extract_pages(path)
    document_metadata = _extract_document_metadata(path, pages, settings)
    chunks: list[DocumentChunk] = []
    for page, text in pages:
        chunker = chunk_text_semantic
        if getattr(settings, "chunk_strategy", "semantic") in {"paragraph", "legacy"}:
            chunker = chunk_text_by_paragraphs
        for chunk in chunker(text, settings.chunk_size, settings.chunk_overlap):
            chunk_index = len(chunks)
            chunk_id = f"{digest}-{page or 0}-{chunk_index}"
            chunk_metadata = {
                "extension": path.suffix.lower(),
                "source_path": str(path),
                **document_metadata,
            }
            section_heading = _section_heading(chunk)
            if section_heading:
                chunk_metadata["section_heading"] = section_heading
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk,
                    source_path=path,
                    document=path.name,
                    page=page,
                    chunk_index=chunk_index,
                    file_hash=digest,
                    metadata=chunk_metadata,
                )
            )
    return path, chunks, pdf_pages


def _extract_document_metadata(
    path: Path,
    pages: list[tuple[int | None, str]],
    settings: AppSettings | None = None,
) -> dict[str, str]:
    sample = "\n".join(text for _page, text in pages[:3])[:12000]
    metadata: dict[str, str] = {}
    title = _infer_document_title(path, sample)
    authors = _infer_authors(sample)
    abstract = _extract_abstract(sample, settings)
    keywords = _extract_keywords(sample, settings)
    document_kind = _infer_document_kind(path, sample)
    for key, value in {
        "document_title": title,
        "authors": authors,
        "abstract": abstract,
        "keywords": keywords,
        "document_kind": document_kind,
    }.items():
        cleaned = _metadata_text(value)
        if cleaned:
            metadata[key] = cleaned
    return metadata


def _infer_document_title(path: Path, text: str) -> str:
    lines = _metadata_lines(text)
    for line in lines[:18]:
        lowered = line.lower()
        if lowered.startswith(("abstract", "keywords", "doi", "http")):
            continue
        if 8 <= len(line) <= 180 and len(line.split()) >= 2:
            return line
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _infer_authors(text: str) -> str:
    lines = _metadata_lines(text)
    candidates: list[str] = []
    for line in lines[1:10]:
        lowered = line.lower()
        if lowered.startswith(("abstract", "keywords", "introduction")):
            break
        if re.search(r"\b(?:university|department|faculty|school|http|doi)\b", lowered):
            continue
        if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", line):
            candidates.append(line)
    return "; ".join(candidates[:3])


def _extract_abstract(text: str, settings: AppSettings | None = None) -> str:
    pattern = (
        getattr(settings, "metadata_abstract_pattern", None)
        or AppSettings().metadata_abstract_pattern
    )
    limit = int(
        getattr(settings, "metadata_abstract_limit", None)
        or AppSettings().metadata_abstract_limit
    )
    match = re.search(
        pattern,
        text or "",
    )
    if not match:
        return ""
    return _metadata_text(match.group("abstract"), limit=limit)


def _extract_keywords(text: str, settings: AppSettings | None = None) -> str:
    pattern = (
        getattr(settings, "metadata_keywords_pattern", None)
        or AppSettings().metadata_keywords_pattern
    )
    limit = int(
        getattr(settings, "metadata_keywords_limit", None)
        or AppSettings().metadata_keywords_limit
    )
    match = re.search(pattern, text or "")
    if not match:
        match = re.search(r"(?im)^\s*keywords?\s*[:\-]\s*(?P<keywords>.+)$", text or "")
    return _metadata_text(match.group("keywords"), limit=limit) if match else ""


def _infer_document_kind(path: Path, text: str) -> str:
    haystack = f"{path.name} {text[:4000]}".lower()
    if any(marker in haystack for marker in ("doctoral thesis", "phd thesis", "dissertation")):
        return "thesis"
    if any(marker in haystack for marker in ("abstract", "journal", "doi", "references")):
        return "research_paper"
    if any(marker in haystack for marker in ("certificate", "attestation", "exam payment")):
        return "certificate"
    if any(marker in haystack for marker in ("manual", "guide", "training", "certification prep")):
        return "manual"
    return "document"


def _section_heading(chunk: str) -> str:
    for line in _metadata_lines(chunk)[:4]:
        if len(line) > 100:
            continue
        if re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,:()/-]{3,}$", line):
            return line
    return ""


def _metadata_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in (text or "").splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]


def _metadata_text(value: str, limit: int = 1200) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "")).strip()
    return cleaned[:limit].strip()


class DocumentIngestor:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingService(
            settings.embed_model,
            settings.embed_device,
            cache_dir=settings.embedding_cache_dir,
            cache_enabled=settings.embedding_cache_enabled,
        )
        self.retriever = ChromaRetriever(
            settings.chroma_dir,
            settings.collection_name,
            self.embeddings,
            settings=settings,
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

        indexed_chunks = 0
        pdf_pages = 0
        errors: list[str] = []
        if progress:
            progress("Parsing documents", 0, len(pending))

        executor_class = (
            ProcessPoolExecutor
            if self.settings.process_parse_documents and len(pending) > 1
            else ThreadPoolExecutor
        )
        with executor_class(max_workers=max(1, self.settings.max_workers)) as pool:
            futures = {
                pool.submit(_parse_file, path, digest, self.settings): (path, digest)
                for path, digest in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                path, digest = futures[future]
                try:
                    parsed_path, chunks, pages = future.result()
                    self.retriever.delete_document(str(parsed_path))
                    indexed_chunks += self._index_chunks(chunks, progress)
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

        if indexed_chunks:
            self.retriever.refresh_lexical_index()
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
        start = 0
        while start < total:
            next_batch_size = _adaptive_embedding_batch_size(
                chunks[start : start + self.settings.batch_size],
                self.settings.batch_size,
            )
            batch = chunks[start : start + next_batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embeddings.embed_documents(texts, batch_size=next_batch_size)
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
            start += len(batch)
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


def _adaptive_embedding_batch_size(chunks: list[DocumentChunk], base_batch_size: int) -> int:
    base = max(1, int(base_batch_size))
    if not chunks:
        return base
    average_chars = sum(len(chunk.text or "") for chunk in chunks) / max(1, len(chunks))
    if average_chars <= 450:
        return min(512, base * 2)
    if average_chars >= 2200:
        base = max(16, base // 2)

    available_memory = _available_memory_bytes()
    if available_memory is None:
        return base
    estimated_bytes_per_chunk = max(512, int(average_chars * 4)) + 1536 * 4
    memory_safe_batch = max(1, int(available_memory * 0.15) // estimated_bytes_per_chunk)
    return max(1, min(base, memory_safe_batch))


def _available_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
        except (OSError, ValueError):
            return None
    return None


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
