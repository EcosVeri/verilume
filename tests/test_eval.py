from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from verilume.core.eval import GoldQuestion, evaluate_retrieval, load_gold_questions

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "eval_corpus"


def _src(document: str, page: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(document=document, page=page)


class GoldQuestionLoadingTests(unittest.TestCase):
    def test_loads_bundled_gold_set(self) -> None:
        gold = load_gold_questions(FIXTURE_DIR / "gold_questions.json")
        self.assertGreaterEqual(len(gold), 6)
        self.assertTrue(any(not q.should_find for q in gold))


class RetrievalScoringTests(unittest.TestCase):
    def test_hit_at_k_ranks_first_relevant_document(self) -> None:
        gold = [GoldQuestion(question="q", expected_documents=["solar_efficiency.txt"])]

        # Relevant doc is at rank 2: misses hit@1 but makes hit@3 and hit@5.
        def search(_query: str, k: int):
            return [_src("wind_turbine.txt"), _src("solar_efficiency.txt")][:k]

        report = evaluate_retrieval(search, gold)
        self.assertEqual(report.metrics.hit_at_1, 0.0)
        self.assertEqual(report.metrics.hit_at_3, 1.0)
        self.assertEqual(report.metrics.hit_at_5, 1.0)

    def test_not_found_question_scores_when_nothing_returned(self) -> None:
        gold = [GoldQuestion(question="q", expected_documents=[], should_find=False)]
        report = evaluate_retrieval(lambda _q, _k: [], gold)
        self.assertEqual(report.metrics.not_found_accuracy, 1.0)

    def test_not_found_question_fails_when_results_returned(self) -> None:
        gold = [GoldQuestion(question="q", expected_documents=[], should_find=False)]
        report = evaluate_retrieval(lambda _q, _k: [_src("solar_efficiency.txt")], gold)
        self.assertEqual(report.metrics.not_found_accuracy, 0.0)

    def test_page_accuracy_requires_matching_document_and_page(self) -> None:
        gold = [
            GoldQuestion(
                question="q",
                expected_documents=["solar_efficiency.txt"],
                expected_pages=[3],
            )
        ]
        good = evaluate_retrieval(lambda _q, _k: [_src("solar_efficiency.txt", 3)], gold)
        bad = evaluate_retrieval(lambda _q, _k: [_src("solar_efficiency.txt", 9)], gold)
        self.assertEqual(good.metrics.page_accuracy, 1.0)
        self.assertEqual(bad.metrics.page_accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
