from __future__ import annotations

import unittest

from verilume.core.schemas import LocalSource, WebSource
from verilume.utils.formatting import (
    local_source_confidence,
    source_confidence,
    web_source_rows,
    web_source_type,
)


class FormattingTests(unittest.TestCase):
    def test_web_source_type_detects_government_and_github_sources(self) -> None:
        government = WebSource(
            label="W1",
            title="The Government",
            url="https://gouvernement.lu/en/gouvernement.html",
            content="Prime Minister profile.",
        )
        github = WebSource(
            label="W2",
            title="Damian Ndiwago",
            url="https://github.com/DamingoNdiwa",
            content="Research code repository.",
        )

        self.assertEqual(web_source_type(government), "Government")
        self.assertEqual(source_confidence(government), "High")
        self.assertEqual(web_source_type(github), "GitHub")
        self.assertEqual(source_confidence(github), "High")

    def test_web_source_rows_use_scan_friendly_source_badges(self) -> None:
        rows = web_source_rows(
            [
                WebSource(
                    label="W1",
                    title="Research Explorer profile",
                    url="https://research.uni.lu/profile",
                    content="University research profile.",
                )
            ]
        )

        self.assertEqual(rows[0]["Source type"], "University")
        self.assertEqual(rows[0]["Confidence"], "High")
        self.assertIn("University", str(rows[0]["Badge"]))
        self.assertEqual(rows[0]["Source"], "Research Explorer profile")

    def test_local_source_confidence_uses_best_retrieval_score(self) -> None:
        self.assertEqual(
            local_source_confidence(
                [
                    LocalSource("S1", "doc.pdf", 1, "c1", "text", 0.59),
                    LocalSource("S2", "doc.pdf", 2, "c2", "text", 0.82),
                ]
            ),
            "High",
        )
        self.assertEqual(
            local_source_confidence([LocalSource("S1", "doc.pdf", 1, "c1", "text", 0.7)]),
            "Medium",
        )
        self.assertEqual(
            local_source_confidence([LocalSource("S1", "doc.pdf", 1, "c1", "text", 0.5)]),
            "Low",
        )

    def test_web_source_confidence_scores_high_medium_low_domains(self) -> None:
        self.assertEqual(
            source_confidence(WebSource("W1", "ACM paper", "https://dl.acm.org/paper", "paper")),
            "High",
        )
        self.assertEqual(
            source_confidence(WebSource("W2", "News", "https://www.bbc.com/news/example", "news")),
            "Medium",
        )
        self.assertEqual(
            source_confidence(WebSource("W3", "Post", "https://www.facebook.com/post", "post")),
            "Low",
        )


if __name__ == "__main__":
    unittest.main()
