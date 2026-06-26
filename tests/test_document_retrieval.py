from __future__ import annotations

import unittest

from verilume.core.claim_verification import verify_claim_support
from verilume.core.document_index import IndexedDocument
from verilume.core.document_retrieval import (
    detect_requested_document,
    document_matches_to_sources,
    requested_document_names,
)
from verilume.core.schemas import LocalSource


class DocumentRetrievalTests(unittest.TestCase):
    def test_detect_requested_document_matches_extensionless_filename(self) -> None:
        documents = [
            IndexedDocument(
                document_id="/docs/kruschke2012.pdf",
                filename="kruschke2012.pdf",
                title="Doing Bayesian Data Analysis",
                page_count=672,
                chunk_count=220,
                document_type="research_paper",
                summary="Bayesian data analysis textbook.",
                keywords=["Bayesian", "MCMC"],
            )
        ]

        result = detect_requested_document("what is in kruschke2012", documents)

        self.assertFalse(result.ambiguous)
        self.assertEqual(result.best.document.filename, "kruschke2012.pdf")
        self.assertEqual(result.best.reason, "extension-insensitive filename match")

    def test_detect_requested_document_flags_ambiguous_partial_matches(self) -> None:
        documents = [
            IndexedDocument("1", "dic.pdf", "Dictionary", 4, 6, "document", "Dictionary.", []),
            IndexedDocument("2", "dic_notes.pdf", "Dictionary notes", 5, 7, "document", "Notes.", []),
        ]

        result = detect_requested_document("summarise dic", documents)

        self.assertTrue(result.ambiguous)
        self.assertEqual([match.document.filename for match in result.matches], ["dic.pdf", "dic_notes.pdf"])

    def test_document_matches_convert_to_sources_with_summary_metadata(self) -> None:
        documents = [
            IndexedDocument(
                "1",
                "gji_196_1_357.pdf",
                "Geophysical Journal Article",
                12,
                18,
                "research_paper",
                "Article summary.",
                ["geophysics"],
            )
        ]

        sources = document_matches_to_sources(
            detect_requested_document("summarise gji_196_1_357.pdf", documents).matches
        )

        self.assertEqual(sources[0].document, "gji_196_1_357.pdf")
        self.assertTrue(sources[0].metadata["document_summary"])
        self.assertIn("Article summary", sources[0].text)

    def test_requested_document_names_extracts_extensionless_summary_target(self) -> None:
        self.assertEqual(requested_document_names("summarise kruschke2012"), ("kruschke2012",))


class ClaimVerificationTests(unittest.TestCase):
    def test_claim_support_marks_cited_local_claim_supported(self) -> None:
        source = LocalSource(
            label="S1",
            document="dic.pdf",
            page=1,
            chunk_id="c1",
            text="The uploaded dictionary file contains terminology and definitions.",
            score=0.95,
        )

        support = verify_claim_support(
            "The uploaded dictionary contains terminology and definitions [S1].",
            local_sources=[source],
            web_sources=[],
        )

        self.assertEqual(support[0].verdict, "supported")
        self.assertEqual(support[0].supporting_source_ids, ["S1"])


if __name__ == "__main__":
    unittest.main()
