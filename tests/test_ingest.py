from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from verilume.core.schemas import DocumentChunk
from verilume.ingest import (
    DocumentIngestor,
    _adaptive_embedding_batch_size,
    _extract_document_metadata,
    chunk_text_semantic,
)
from verilume.settings import AppSettings


class FakeRetriever:
    def __init__(self) -> None:
        self.deleted_paths: list[str] = []

    def delete_document(self, source_path: str) -> None:
        self.deleted_paths.append(source_path)


def _chunk(text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{len(text)}",
        text=text,
        source_path=Path("doc.pdf"),
        document="doc.pdf",
        page=1,
        chunk_index=0,
        file_hash="hash",
    )


class IngestCleanupTests(unittest.TestCase):
    def test_document_metadata_is_extracted_from_research_text(self) -> None:
        text = """
        Replica Exchange Hamiltonian Monte Carlo for Hydrological Models
        Damian Mingo Ndiwago
        University of Luxembourg

        Abstract
        Replica Exchange Hamiltonian Monte Carlo combines Hamiltonian Monte Carlo with
        replica exchange to improve exploration of multimodal posterior distributions
        in Bayesian hydrological modelling.

        Keywords: Hamiltonian Monte Carlo; Replica Exchange; Bayesian inference

        1 Introduction
        The method is evaluated on HBV models.
        """

        metadata = _extract_document_metadata(Path("hremc-paper.pdf"), [(1, text)])

        self.assertEqual(
            metadata["document_title"],
            "Replica Exchange Hamiltonian Monte Carlo for Hydrological Models",
        )
        self.assertIn("Damian Mingo Ndiwago", metadata["authors"])
        self.assertIn("combines Hamiltonian Monte Carlo", metadata["abstract"])
        self.assertIn("Replica Exchange", metadata["keywords"])
        self.assertEqual(metadata["document_kind"], "research_paper")

    def test_adaptive_embedding_batch_size_uses_chunk_length(self) -> None:
        self.assertEqual(
            _adaptive_embedding_batch_size([_chunk("short text")] * 4, 128),
            256,
        )
        self.assertEqual(
            _adaptive_embedding_batch_size([_chunk("medium text " * 90)] * 4, 128),
            128,
        )
        self.assertEqual(
            _adaptive_embedding_batch_size([_chunk("long text " * 260)] * 4, 128),
            64,
        )

    def test_semantic_chunking_prefers_sentence_boundaries(self) -> None:
        text = (
            "Alpha sentence one. Beta sentence two carries important context. "
            "Gamma sentence three closes the thought."
        )

        chunks = chunk_text_semantic(text, chunk_size=62, chunk_overlap=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.endswith(".") for chunk in chunks))
        self.assertIn("Beta sentence two", " ".join(chunks))

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
