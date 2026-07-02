"""Integration test: removing a document drops it from local search.

Exercises the real Chroma-backed retriever (not a fake) end to end, so it
proves that ``delete_document`` actually removes a document's chunks from the
vector database and that subsequent local searches no longer surface them.
Lexical search is used so the test stays deterministic and embedding-free.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from verilume.core.retrieval import ChromaRetriever


class _StubEmbeddings:
    """Placeholder embedder; lexical search never calls it."""

    def embed_query(self, text: str):  # pragma: no cover - defensive
        raise AssertionError("lexical search must not require embeddings")


def _index_document(retriever: ChromaRetriever, source_path: str, document: str, text: str) -> None:
    retriever.add_chunks(
        ids=[f"{document}-0"],
        documents=[text],
        metadatas=[{"source_path": source_path, "document": document, "page": 1}],
        embeddings=[[0.1, 0.2]],
    )


# A handful of unrelated documents so BM25's idf term stays meaningful; on a
# one-document corpus every score collapses toward zero regardless of removal.
_FILLER = {
    "solar.pdf": "solar panels convert solar sunlight into solar electricity on rooftops",
    "ocean.pdf": "ocean currents move warm ocean water across the ocean basins",
    "coffee.pdf": "roasted coffee beans give coffee its aroma and coffee flavour",
    "railway.pdf": "the railway network links railway stations along the railway line",
}


class RemovalDropsFromLocalSearchTests(unittest.TestCase):
    def test_removed_document_no_longer_appears_in_local_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            retriever = ChromaRetriever(Path(tmp) / "chroma", "test_removal", _StubEmbeddings())
            try:
                for document, text in _FILLER.items():
                    _index_document(retriever, str(Path(tmp) / document), document, text)

                keep_path = str(Path(tmp) / "keep.pdf")
                remove_path = str(Path(tmp) / "remove.pdf")
                _index_document(
                    retriever,
                    keep_path,
                    "keep.pdf",
                    "photosynthesis lets photosynthesis plants turn photosynthesis sunlight into energy",
                )
                _index_document(
                    retriever,
                    remove_path,
                    "remove.pdf",
                    "the zeppelin airship zeppelin drifted over the zeppelin harbour at dawn",
                )

                count_before = retriever.count()

                # The target document is retrievable before removal.
                before = retriever.search("zeppelin airship harbour", k=5, mode="lexical")
                self.assertIn("remove.pdf", {source.document for source in before})

                retriever.delete_document(remove_path)

                # It is gone from search and the DB shrank by exactly its chunk.
                after = retriever.search("zeppelin airship harbour", k=5, mode="lexical")
                self.assertNotIn("remove.pdf", {source.document for source in after})
                self.assertEqual(retriever.count(), count_before - 1)

                # Removal is targeted: an untouched document still resolves.
                kept = retriever.search("photosynthesis plants sunlight", k=5, mode="lexical")
                self.assertIn("keep.pdf", {source.document for source in kept})
            finally:
                retriever.close(clear_system_cache=True)


if __name__ == "__main__":
    unittest.main()
