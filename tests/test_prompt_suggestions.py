from __future__ import annotations

import unittest

from verilume.core.document_index import IndexedDocument
from verilume.core.prompt_suggestions import (
    MAX_SUGGESTIONS,
    classify_document_type,
    generate_suggested_prompts,
)
from verilume.settings import AppSettings


def _document(
    filename: str,
    title: str,
    document_type: str,
    summary: str = "",
    keywords: list[str] | None = None,
    *,
    pages: int = 12,
    chunks: int = 20,
) -> IndexedDocument:
    return IndexedDocument(
        document_id=f"/docs/{filename}",
        filename=filename,
        title=title,
        page_count=pages,
        chunk_count=chunks,
        document_type=document_type,
        summary=summary,
        keywords=keywords or [],
    )


class PromptSuggestionTests(unittest.TestCase):
    def test_empty_library_returns_onboarding_prompts(self) -> None:
        suggestions = generate_suggested_prompts([], [], AppSettings())

        self.assertEqual(suggestions[0].title, "Upload documents")
        self.assertIn("Build your knowledge base", [item.title for item in suggestions])
        self.assertLessEqual(len(suggestions), MAX_SUGGESTIONS)

    def test_paper_suggestions_use_title_not_filename(self) -> None:
        documents = [
            _document(
                "Climate Change Impacts.pdf",
                "Climate Change Impacts",
                "scientific_paper",
                "A paper about climate impacts, methodology, findings, and references.",
                ["climate", "impacts"],
            )
        ]

        suggestions = generate_suggested_prompts(documents, [], AppSettings())
        rendered = "\n".join(f"{item.title} {item.prompt}" for item in suggestions)

        self.assertIn("Climate Change Impacts", rendered)
        self.assertNotIn(".pdf", rendered)
        self.assertTrue(any("Key findings" in item.title for item in suggestions))

    def test_core_generic_prompts_remain_available(self) -> None:
        documents = [
            _document(
                "Climate Change Impacts.pdf",
                "Climate Change Impacts",
                "scientific_paper",
                "A paper about climate impacts.",
            )
        ]

        titles = [
            item.title
            for item in generate_suggested_prompts(documents, [], AppSettings())
        ]

        self.assertIn("Summarise uploaded documents", titles)
        self.assertIn("List indexed documents", titles)
        self.assertIn("Compare local and web evidence", titles)

    def test_deleted_document_prompt_disappears_when_removed_from_index(self) -> None:
        documents = [
            _document("Bayesian Statistics.pdf", "Bayesian Statistics", "scientific_paper"),
            _document("Hydrology Thesis.pdf", "Hydrology Thesis", "thesis"),
        ]

        suggestions = generate_suggested_prompts(documents, [], AppSettings())
        rendered = "\n".join(f"{item.title} {item.prompt}" for item in suggestions)

        self.assertNotIn("Passport", rendered)
        self.assertNotIn("Passport.pdf", rendered)

    def test_presentation_and_certificate_templates_are_contextual(self) -> None:
        presentation = _document("strategy.pptx", "Company Strategy 2026", "document")
        certificate = _document("certificate.pdf", "Course Completion", "certificate")

        presentation_suggestions = generate_suggested_prompts(
            [presentation],
            [],
            AppSettings(),
        )
        presentation_titles = [item.title for item in presentation_suggestions]
        certificate_titles = [
            item.title
            for item in generate_suggested_prompts([certificate], [], AppSettings())
        ]

        self.assertEqual(classify_document_type(presentation), "presentation")
        self.assertIn("Create speaker notes", presentation_titles)
        self.assertTrue(
            any(
                item.title == "Create speaker notes"
                and item.prompt == "Create speaker notes from Company Strategy 2026"
                for item in presentation_suggestions
            )
        )
        self.assertIn("Extract important fields", certificate_titles)

    def test_large_library_prefers_collection_prompts(self) -> None:
        documents = [
            _document(
                f"paper-{index}.pdf",
                f"Research Paper {index}",
                "scientific_paper",
                pages=10 + index,
                chunks=20 + index,
            )
            for index in range(25)
        ]

        suggestions = generate_suggested_prompts(documents, [], AppSettings())
        rendered = "\n".join(item.title for item in suggestions)

        self.assertLessEqual(len(suggestions), MAX_SUGGESTIONS)
        self.assertIn("Create literature review", rendered)
        self.assertIn("Find common research themes", rendered)
        self.assertNotIn("Research Paper 0", rendered)

    def test_recent_history_boosts_formula_suggestions(self) -> None:
        document = _document(
            "bayes.pdf",
            "Bayesian Methods",
            "scientific_paper",
            "Paper with equations and posterior inference.",
        )

        suggestions = generate_suggested_prompts(
            [document],
            ["Can you explain the equations?"],
            AppSettings(),
        )

        self.assertTrue(any(item.category == "formula" for item in suggestions))


if __name__ == "__main__":
    unittest.main()
