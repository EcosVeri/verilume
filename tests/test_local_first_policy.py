from __future__ import annotations

import unittest

from verilume.core.generation import LOCAL_UNKNOWN
from verilume.core.schemas import LocalSource, WebSource
from verilume.rag import VerilumeRAG
from verilume.settings import AppSettings


LOCAL_SOURCE = LocalSource(
    label="S1",
    document="doc.pdf",
    page=1,
    chunk_id="chunk-1",
    text="Local evidence snippet.",
    score=0.91,
)


class FakeRetriever:
    def __init__(self, sources):
        self.sources = list(sources)
        self.calls: list[tuple[tuple, dict]] = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return list(self.sources)


class FakeWebSearch:
    def __init__(self, sources):
        self.sources = list(sources)
        self.queries: list[str] = []

    def search(self, query):
        self.queries.append(query)
        return list(self.sources)


class BrokenWebSearch:
    def search(self, query):
        raise RuntimeError("Search provider unavailable")


class FakeGenerator:
    def __init__(self, local_answer, model_answer="Model answer", web_answer="Web answer [W1]"):
        self.local_answer = local_answer
        self.model_answer = model_answer
        self.web_answer = web_answer
        self.local_calls: list[tuple] = []
        self.model_calls: list[tuple] = []
        self.final_calls: list[dict] = []

    def rewrite_query(self, question, history):
        return question

    def answer_local(self, question, history, local_sources):
        self.local_calls.append((question, history, local_sources))
        return self.local_answer

    def answer_model_knowledge(self, question, history):
        self.model_calls.append((question, history))
        return self.model_answer

    def answer_with_web(self, **kwargs):
        return self.web_answer

    def answer_final(self, **kwargs):
        self.final_calls.append(kwargs)
        return self.web_answer


class LocalFirstWorkflowTests(unittest.TestCase):
    def _make_rag(
        self,
        *,
        local_answer,
        model_answer="Model answer",
        web_answer="Web answer [W1]",
        local_sources=None,
        web_sources=None,
        web_enabled=True,
    ):
        rag = VerilumeRAG(
            AppSettings(
                hf_token="token",
                tavily_api_key="key",
                enable_web_search=web_enabled,
                semantic_cache_enabled=False,
            )
        )
        rag.retriever = FakeRetriever([LOCAL_SOURCE] if local_sources is None else local_sources)
        rag.generator = FakeGenerator(local_answer, model_answer, web_answer)
        rag.web_search = FakeWebSearch(
            [
                WebSource(
                    label="W1",
                    title="One",
                    url="https://example.com/1",
                    content="one",
                )
            ]
            if web_sources is None
            else web_sources
        )
        rag._search_duckduckgo_fallback = lambda query: []
        rag._search_duckduckgo_fallback_queries = lambda queries: []
        return rag

    def test_stable_question_combines_model_and_web_when_web_enabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Econometrics applies statistical methods to economic data.",
            web_answer="Econometrics applies statistical methods to economic data and combines statistics and economics [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Econometrics overview",
                    url="https://example.edu/econometrics",
                    content="Econometrics combines statistics and economics.",
                )
            ],
        )

        result = rag.ask("What is econometrics?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertTrue(result.used_web)
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertEqual(result.diagnostics.get("evidence_winner"), "hybrid")
        self.assertIn("Econometrics", result.answer)

    def test_stable_question_keeps_local_evidence_but_still_checks_web_when_enabled(self) -> None:
        local_source = LocalSource(
            label="S1",
            document="econometrics_note.pdf",
            page=1,
            chunk_id="econometrics-note",
            text="Econometrics applies statistical methods to economic data.",
            score=0.95,
        )
        rag = self._make_rag(
            local_answer="Econometrics applies statistical methods to economic data [S1].",
            model_answer="Econometrics applies statistical methods to economic data.",
            web_answer=(
                "Econometrics applies statistical methods to economic data, "
                "with local evidence and web validation [S1] [W1]."
            ),
            local_sources=[local_source],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Econometrics overview",
                    url="https://example.edu/econometrics",
                    content="Econometrics combines statistics and economics.",
                )
            ],
        )

        result = rag.ask("What is econometrics?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertTrue(result.used_web)
        self.assertEqual(result.diagnostics["web_reason"], "local_weighted_hybrid")
        self.assertTrue(result.diagnostics["used_local"])
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertTrue(result.diagnostics["used_web"])
        self.assertEqual(result.diagnostics["evidence_winner"], "hybrid")
        self.assertIn("Econometrics", result.answer)

    def test_explicit_web_request_uses_web_after_local_and_model(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Econometrics applies statistical methods to economic data.",
            web_answer="Econometrics applies statistical methods to economic data and combines statistics and economics [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Econometrics overview",
                    url="https://example.edu/econometrics",
                    content="Econometrics combines statistics and economics.",
                )
            ],
        )

        result = rag.ask("Search the web about econometrics")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertTrue(result.used_web)
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertIn("[W1]", result.answer)

    def test_current_government_question_uses_web_but_checks_local_first(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Outdated answer.",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="FRIEDEN Luc - The Luxembourg Government",
                    url="https://gouvernement.lu/en/gouvernement/luc-frieden.html",
                    content="Luc Frieden is the Prime Minister of Luxembourg.",
                )
            ],
        )

        result = rag.ask("Who is the current prime minister of Luxembourg?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertFalse(result.diagnostics["parallel_model_with_web"])
        self.assertTrue(result.diagnostics["model_skipped_for_current_web"])
        self.assertTrue(result.used_web)
        self.assertIn("Luc Frieden", result.answer)

    def test_web_disabled_allows_model_for_stable_question(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=(
                "Bayesian inference updates prior beliefs with observed evidence "
                "to produce a posterior distribution."
            ),
            local_sources=[],
            web_enabled=False,
        )

        result = rag.ask("What is Bayesian inference?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "model-only")
        self.assertIn("posterior", result.answer.lower())

    def test_web_disabled_rejects_model_as_current_fact(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Sam Altman is the current CEO of OpenAI.",
            local_sources=[],
            web_enabled=False,
        )

        result = rag.ask("Who is the current CEO of OpenAI?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "low")
        self.assertNotIn("Sam Altman", result.answer)
        self.assertIn("AI knowledge is not reliable enough for current facts", result.answer)

    def test_dynamic_fact_without_current_word_uses_web_not_model_primary(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Europe has about 740 million people.",
            web_answer="The population of Europe is reported by the web evidence [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Europe population data",
                    url="https://example.org/europe-population",
                    content="The population of Europe is reported by this current dataset.",
                )
            ],
        )

        result = rag.ask("What is the population of Europe?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertTrue(result.used_web)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertTrue(result.diagnostics["model_skipped_for_current_web"])
        self.assertEqual(result.diagnostics["fact_type"], "dynamic_fact")
        self.assertIn("[W1]", result.answer)

    def test_explicit_web_failure_falls_back_to_local_answer(self) -> None:
        rag = self._make_rag(local_answer="Local answer [S1]")
        rag.web_search = BrokenWebSearch()

        result = rag.ask("Search the web for doc")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "local-grounded")
        self.assertIn("Local answer [S1]", result.answer)
        self.assertIn("Web update:", result.answer)


if __name__ == "__main__":
    unittest.main()
