from __future__ import annotations

import unittest

from verilume.core.agentic_planner import (
    ANSWER_MODEL,
    CALCULATE,
    EXTRACT_TABLE,
    SEARCH_LOCAL,
    SEARCH_WEB,
    SUMMARIZE_DOCUMENTS,
    AgenticPlanner,
)
from verilume.core.query_interpreter import InterpretedQuery
from verilume.settings import AppSettings


class AgenticPlannerTests(unittest.TestCase):
    def _plan(
        self,
        question: str,
        *,
        intent: str = "general",
        use_web: bool = False,
        settings: AppSettings | None = None,
    ):
        interpretation = InterpretedQuery(
            original_question=question,
            resolved_question=question,
            intent=intent,
            use_local=True,
            use_web=use_web,
            use_model_knowledge=True,
            search_queries=[question],
        )
        return AgenticPlanner().plan(
            question,
            interpretation,
            settings or AppSettings(enable_web_search=True, web_search_provider="duckduckgo"),
        )

    def test_stable_fact_uses_local_model_and_web_when_enabled(self) -> None:
        plan = self._plan("What is regression analysis?")

        self.assertIn(SEARCH_LOCAL, plan.actions)
        self.assertIn(ANSWER_MODEL, plan.actions)
        self.assertIn(SEARCH_WEB, plan.actions)
        self.assertEqual(plan.question_type, "definition")

    def test_current_fact_uses_local_and_web_without_model_evidence(self) -> None:
        plan = self._plan("Who is the prime minister of Norway?")

        self.assertIn(SEARCH_LOCAL, plan.actions)
        self.assertIn(SEARCH_WEB, plan.actions)
        self.assertNotIn(ANSWER_MODEL, plan.actions)
        self.assertEqual(plan.question_type, "current_dynamic_fact")

    def test_local_summary_adds_summarize_documents(self) -> None:
        plan = self._plan(
            "Summarise the documents in the database.",
            intent="local_document",
            settings=AppSettings(enable_web_search=False),
        )

        self.assertIn(SEARCH_LOCAL, plan.actions)
        self.assertIn(SUMMARIZE_DOCUMENTS, plan.actions)
        self.assertNotIn(SEARCH_WEB, plan.actions)

    def test_table_question_adds_extract_and_calculate(self) -> None:
        plan = self._plan(
            "What is the average price in this CSV?",
            settings=AppSettings(enable_web_search=False),
        )

        self.assertIn(SEARCH_LOCAL, plan.actions)
        self.assertIn(EXTRACT_TABLE, plan.actions)
        self.assertIn(CALCULATE, plan.actions)
        self.assertEqual(plan.question_type, "table_calculation")

    def test_explicit_web_keeps_local_and_model_for_stable_question(self) -> None:
        plan = self._plan("Search the web for Christophe Ley.", use_web=True)

        self.assertIn(SEARCH_LOCAL, plan.actions)
        self.assertIn(SEARCH_WEB, plan.actions)
        self.assertIn(ANSWER_MODEL, plan.actions)

    def test_local_only_mode_blocks_model_and_web_actions(self) -> None:
        plan = self._plan(
            "What is regression analysis?",
            settings=AppSettings(search_mode="local only", enable_web_search=True),
        )

        self.assertEqual(plan.actions, [SEARCH_LOCAL])

    def test_web_only_mode_uses_web_action(self) -> None:
        plan = self._plan(
            "Search this online.",
            settings=AppSettings(search_mode="web only", enable_web_search=True, web_search_provider="duckduckgo"),
        )

        self.assertEqual(plan.actions, [SEARCH_WEB])


if __name__ == "__main__":
    unittest.main()
