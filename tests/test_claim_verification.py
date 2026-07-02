"""Tests for claim-to-source support verification, incl. date grounding."""

from __future__ import annotations

import unittest

from verilume.core.claim_verification import verify_claim_support
from verilume.core.schemas import LocalSource, WebSource


def _web(label: str, title: str, content: str) -> WebSource:
    return WebSource(label=label, title=title, url=f"https://example.com/{label}", content=content)


def _local(label: str, document: str, text: str) -> LocalSource:
    return LocalSource(label=label, document=document, page=1, chunk_id=f"{label}-0", text=text, score=0.8)


class ClaimVerificationDateGroundingTests(unittest.TestCase):
    def test_date_present_in_source_stays_supported(self) -> None:
        answer = "Jonas Gahr Store became the prime minister of Norway on 14 October 2021 [W1]."
        supports = verify_claim_support(
            answer,
            local_sources=[],
            web_sources=[
                _web(
                    "W1",
                    "Store Cabinet",
                    "Jonas Gahr Store became prime minister of Norway on 14 October 2021 leading the Labour Party.",
                )
            ],
        )
        self.assertTrue(supports)
        support = supports[0]
        self.assertTrue(support.date_grounded)
        self.assertEqual(support.verdict, "supported")

    def test_fabricated_date_absent_from_sources_is_downgraded(self) -> None:
        # Mirrors the observed failure: the model stitched together a date that
        # appears in no source, yet term overlap alone rated the claim supported.
        answer = "Jonas Gahr Store became the prime minister of Norway on 1 October 2025 [W1]."
        supports = verify_claim_support(
            answer,
            local_sources=[],
            web_sources=[
                _web(
                    "W1",
                    "Prime Minister Jonas Gahr Store",
                    "Jonas Gahr Store has served as prime minister of Norway since 2021 leading the Labour Party. "
                    "He served as government minister from 1 October 2009 to 30 September 2013.",
                )
            ],
        )
        self.assertTrue(supports)
        support = supports[0]
        self.assertFalse(support.date_grounded)
        self.assertNotEqual(support.verdict, "supported")

    def test_claim_without_a_date_is_unaffected(self) -> None:
        answer = "The current prime minister of Norway is Jonas Gahr Store [W1]."
        supports = verify_claim_support(
            answer,
            local_sources=[],
            web_sources=[
                _web(
                    "W1",
                    "World Leaders",
                    "Jonas Gahr Store has served as prime minister of Norway and leader of the Labour Party.",
                )
            ],
        )
        self.assertTrue(supports)
        self.assertTrue(supports[0].date_grounded)

    def test_month_day_year_format_is_grounded(self) -> None:
        answer = "The report was published on October 1, 2025 [W1]."
        supports = verify_claim_support(
            answer,
            local_sources=[],
            web_sources=[_web("W1", "Report", "The annual report was published on 1 October 2025 by the agency.")],
        )
        self.assertTrue(supports[0].date_grounded)


if __name__ == "__main__":
    unittest.main()
