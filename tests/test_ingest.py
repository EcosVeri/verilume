from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from verilume.ingest import DocumentIngestor
from verilume.settings import AppSettings


class FakeRetriever:
    def __init__(self) -> None:
        self.deleted_paths: list[str] = []

    def delete_document(self, source_path: str) -> None:
        self.deleted_paths.append(source_path)


class IngestCleanupTests(unittest.TestCase):
    def test_missing_documents_are_removed_from_manifest_and_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            existing_path = tmp_path / "exists.pdf"
            missing_path = tmp_path / "missing.pdf"
            existing_path.write_text("hello", encoding="utf-8")

            settings = AppSettings(
                docs_dir=tmp_path,
                chroma_dir=tmp_path / "chroma",
                manifest_path=tmp_path / "manifest.json",
            )
            ingestor = DocumentIngestor(settings)
            ingestor.retriever = FakeRetriever()

            manifest = {
                str(existing_path): {"hash": "a"},
                str(missing_path): {"hash": "b"},
            }
            ingestor._remove_missing_documents(manifest)

            self.assertEqual(list(manifest), [str(existing_path)])
            self.assertEqual(ingestor.retriever.deleted_paths, [str(missing_path)])
