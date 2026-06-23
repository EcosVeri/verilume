from __future__ import annotations

import unittest
from datetime import date

from verilume.core.evidence import (
    EvidenceAuthority,
    EvidenceItem,
    EvidenceSourceType,
    QueryType,
    build_final_answer_payload,
    classify_question,
    evidence_from_sources,
    rank_evidence,
    reconcile_dates,
    resolve_evidence_conflicts,
    verified_evidence_for_generation,
)
from verilume.core.schemas import LocalSource, WebSource


class EvidenceLayerTests(unittest.TestCase):
    def test_classifies_current_entity_lookup_as_web_validated(self) -> None:
        query = classify_question("Who is the current Prime Minister of Luxembourg?")

        self.assertEqual(query.primary_type, QueryType.TIME_SENSITIVE)
        self.assertTrue(query.time_sensitive_question)
        self.assertTrue(query.personal_company_entity_lookup)
        self.assertTrue(query.requires_web_validation)
        self.assertTrue(query.requires_date_reconciliation)
        self.assertFalse(query.ai_knowledge_allowed_as_final)

    def test_classifies_lowercase_entity_statement_as_web_validated_lookup(self) -> None:
        query = classify_question("sofia loizidou")

        self.assertEqual(query.primary_type, QueryType.GENERAL)
        self.assertTrue(query.personal_company_entity_lookup)
        self.assertTrue(query.requires_web_validation)
        self.assertFalse(query.requires_date_reconciliation)

    def test_converts_existing_sources_to_evidence_items(self) -> None:
        local = LocalSource(
            label="S1",
            document="briefing.pdf",
            page=3,
            chunk_id="chunk-1",
            text="The internal project owner is Maya.",
            score=0.89,
            metadata={"source_path": "/tmp/briefing.pdf", "document_date": "2025-02-14"},
        )
        web = WebSource(
            label="W1",
            title="Official update",
            url="https://gouvernement.lu/news",
            content="Official update from 2026.",
            score=0.8,
            published_date="2026-06-01",
        )

        items = evidence_from_sources(
            local_sources=[local],
            web_sources=[web],
            ai_answer="Background model answer.",
        )

        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].citation(), "[S1]")
        self.assertEqual(items[0].document_date, date(2025, 2, 14))
        self.assertEqual(items[0].page, 3)
        self.assertEqual(items[1].citation(), "[W1]")
        self.assertEqual(items[1].authority, EvidenceAuthority.OFFICIAL)
        self.assertTrue(items[2].is_ai_knowledge)

    def test_newer_official_web_evidence_ranks_above_stale_local_for_current_fact(self) -> None:
        local = EvidenceItem(
            source_type=EvidenceSourceType.LOCAL_CHUNK,
            title="old_government_note.pdf",
            content="In 2024, the prime minister was Example Old.",
            document_date="2024",
            semantic_relevance_score=0.92,
            citation_label="S1",
        )
        web = EvidenceItem(
            source_type=EvidenceSourceType.WEB,
            title="Official government biography",
            content="The current prime minister is Example New.",
            url="https://gouvernement.lu/en/government.html",
            document_date="2026-06-01",
            semantic_relevance_score=0.82,
            citation_label="W1",
        )
        ai = EvidenceItem.from_ai_knowledge("The prime minister is Example Old.")

        ranked = rank_evidence(
            [local, web, ai],
            "Who is the current prime minister?",
            today=date(2026, 6, 21),
        )
        reconciliation = reconcile_dates(ranked, "Who is the current prime minister?")
        resolution = resolve_evidence_conflicts(
            "Who is the current prime minister?",
            ranked,
            local_answer="Example Old [S1]",
            web_answer="Example New [W1]",
            ai_knowledge_answer="Example Old",
            reconciliation=reconciliation,
        )

        self.assertEqual(ranked[0].citation(), "[W1]")
        self.assertTrue(reconciliation.local_is_older_than_web)
        self.assertEqual(resolution.winner, EvidenceSourceType.WEB)
        self.assertTrue(resolution.should_disclose_conflict)
        self.assertIn("Newer reliable web evidence wins", resolution.evidence_note)

    def test_private_local_facts_override_web_and_ai(self) -> None:
        local = EvidenceItem(
            source_type=EvidenceSourceType.LOCAL_CHUNK,
            title="internal_plan.pdf",
            content="Project Alpha launch owner is Maya.",
            semantic_relevance_score=0.82,
            citation_label="S1",
        )
        web = EvidenceItem(
            source_type=EvidenceSourceType.WEB,
            title="Company profile",
            content="Project Alpha is owned by a public team.",
            url="https://example.com/company",
            semantic_relevance_score=0.9,
            citation_label="W1",
        )

        ranked = rank_evidence([local, web], "What does my uploaded document say about Alpha?")
        resolution = resolve_evidence_conflicts(
            "What does my uploaded document say about Alpha?",
            ranked,
            local_answer="Maya owns Project Alpha [S1]",
            web_answer="A public team owns Project Alpha [W1]",
        )

        self.assertEqual(ranked[0].citation(), "[S1]")
        self.assertEqual(resolution.winner, EvidenceSourceType.LOCAL_CHUNK)
        self.assertIn("Local files win", resolution.evidence_note)

    def test_verified_generation_payload_excludes_ai_for_current_public_facts(self) -> None:
        web = EvidenceItem(
            source_type=EvidenceSourceType.WEB,
            title="Official current result",
            content="The 2026 official result is final.",
            url="https://public.lu/result",
            document_date="2026-06-10",
            semantic_relevance_score=0.78,
            citation_label="W1",
        )
        ai = EvidenceItem.from_ai_knowledge("The old result may be different.")

        verified = verified_evidence_for_generation(
            "What is the latest official result?",
            [web, ai],
        )
        payload = build_final_answer_payload(
            "What is the latest official result?",
            [web, ai],
        )

        self.assertEqual([item.source_type for item in verified], [EvidenceSourceType.WEB])
        self.assertEqual(payload.evidence_badge, "Current web verified")
        self.assertEqual(payload.citations, ["[W1]"])
        self.assertIn("Use only the verified evidence", payload.generator_instructions)


if __name__ == "__main__":
    unittest.main()
