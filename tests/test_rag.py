from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from verilume.core.generation import (
    LOCAL_UNKNOWN,
    MODEL_SELECTION_WARNING,
    MODEL_UNKNOWN,
    GenerationError,
)
from verilume.core.schemas import ChatMessage, LocalSource, WebSource
from verilume.core.search_planner import ResolvedSearchPlan
from verilume.rag import (
    LOCAL_FILE_NOT_FOUND,
    GenerationStopped,
    VerilumeRAG,
    _current_public_fact_evidence,
    _dedupe_web_queries,
    _merge_web_sources,
    _rank_web_sources,
    _should_rewrite_query,
    _verify_answer_against_evidence,
)
from verilume.settings import AppSettings


LOCAL_SOURCE = LocalSource(
    label="S1",
    document="doc.pdf",
    page=2,
    chunk_id="chunk-1",
    text="Local answer text.",
    score=0.91,
)


class FakeRetriever:
    def __init__(self, sources):
        self.sources = list(sources)
        self.calls: list[tuple[tuple, dict]] = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return list(self.sources)


class CorpusBrowsingRetriever(FakeRetriever):
    def __init__(self, sources, corpus_sources):
        super().__init__(sources)
        self.corpus_sources = list(corpus_sources)
        self.corpus_calls: list[dict] = []

    def sample_sources_by_document(self, **kwargs):
        self.corpus_calls.append(kwargs)
        return list(self.corpus_sources)


class SequentialLocalRetriever:
    def __init__(self, batches):
        self.batches = [list(batch) for batch in batches]
        self.calls: list[tuple[tuple, dict]] = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self.batches:
            return []
        return self.batches.pop(0)


class QueryAwareRetriever:
    def __init__(self, query_to_sources):
        self.query_to_sources = query_to_sources
        self.calls: list[tuple[tuple, dict]] = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        query = args[0] if args else ""
        return list(self.query_to_sources(query))


class FakeWebSearch:
    def __init__(self, sources):
        self.sources = list(sources)
        self.queries: list[str] = []

    def search(self, query):
        self.queries.append(query)
        return list(self.sources)


class SequentialWebSearch:
    def __init__(self, batches):
        self.batches = [list(batch) for batch in batches]
        self.queries: list[str] = []

    def search(self, query):
        self.queries.append(query)
        if not self.batches:
            return []
        return self.batches.pop(0)


class QueryAwareWebSearch:
    max_results = 5

    def __init__(self, query_to_sources):
        self.query_to_sources = query_to_sources
        self.queries: list[str] = []

    def search(self, query):
        self.queries.append(query)
        return list(self.query_to_sources(query))


class BrokenWebSearch:
    def search(self, query):
        raise RuntimeError("Search provider unavailable")


class SlowWebSearch(FakeWebSearch):
    def __init__(self, sources, delay_seconds: float = 0.15):
        super().__init__(sources)
        self.delay_seconds = delay_seconds

    def search(self, query):
        time.sleep(self.delay_seconds)
        return super().search(query)


class FakeGenerator:
    def __init__(self, local_answer, model_answer="Model answer", web_answer="Web answer [W1]"):
        self.local_answer = local_answer
        self.model_answer = model_answer
        self.web_answer = web_answer
        self.local_calls: list[tuple] = []
        self.model_calls: list[tuple] = []
        self.final_calls: list[dict] = []
        self.rewrite_calls: list[tuple] = []

    def rewrite_query(self, question, history):
        self.rewrite_calls.append((question, history))
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


class SlowModelGenerator(FakeGenerator):
    def __init__(
        self,
        local_answer,
        model_answer="Model answer",
        web_answer="Web answer [W1]",
        delay_seconds: float = 0.15,
    ):
        super().__init__(local_answer, model_answer, web_answer)
        self.delay_seconds = delay_seconds

    def answer_model_knowledge(self, question, history):
        time.sleep(self.delay_seconds)
        return super().answer_model_knowledge(question, history)


class CapacityErrorGenerator(FakeGenerator):
    def __init__(self, web_answer="Web answer [W1]"):
        super().__init__(local_answer="", web_answer=web_answer)

    def answer_local(self, question, history, local_sources):
        raise GenerationError(MODEL_SELECTION_WARNING)


class ModelOnlyCapacityErrorGenerator(FakeGenerator):
    def __init__(self):
        super().__init__(local_answer=LOCAL_UNKNOWN, model_answer=MODEL_UNKNOWN)

    def answer_model_knowledge(self, question, history):
        raise GenerationError(MODEL_SELECTION_WARNING)


class FinalSynthesisErrorGenerator(FakeGenerator):
    def answer_final(self, **kwargs):
        self.final_calls.append(kwargs)
        raise GenerationError("provider timed out")


class StructuredFinalGenerator(FakeGenerator):
    def __init__(self, local_answer=LOCAL_UNKNOWN, model_answer=MODEL_UNKNOWN):
        super().__init__(
            local_answer=local_answer,
            model_answer=model_answer,
            web_answer="Structured answer [W1]",
        )
        self.chat_messages: list[list[dict[str, str]]] = []

    def chat(self, messages):
        self.chat_messages.append(messages)
        return "Econometrics uses statistical methods to analyse economic data [W1]."


class RAGRoutingTests(unittest.TestCase):
    def _make_rag(
        self,
        *,
        local_answer,
        model_answer="Model answer",
        web_answer="Web answer [W1]",
        local_sources=None,
        web_sources=None,
    ):
        rag = VerilumeRAG(AppSettings(hf_token="token", tavily_api_key="key"))
        rag.retriever = FakeRetriever([LOCAL_SOURCE] if local_sources is None else local_sources)
        rag.generator = FakeGenerator(local_answer, model_answer, web_answer)
        rag.web_search = FakeWebSearch(
            [
                WebSource(label="W1", title="One", url="https://example.com/1", content="one"),
                WebSource(label="W2", title="Two", url="https://example.com/2", content="two"),
            ]
            if web_sources is None
            else web_sources
        )
        rag._search_duckduckgo_fallback = lambda query: []
        rag._search_duckduckgo_fallback_queries = lambda queries: []
        return rag

    def test_local_answer_short_circuits(self) -> None:
        rag = self._make_rag(local_answer="Local answer [S1]")
        result = rag.ask("What is in the file?")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertEqual([source.label for source in result.local_sources], ["S1"])
        self.assertEqual(result.web_sources, [])
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertIn(result.diagnostics["answer_verification_status"], {"verified", "partial"})
        self.assertEqual(rag.web_search.queries, [])

    def test_local_answer_can_report_hybrid_evidence_when_model_corroborates(self) -> None:
        local_source = LocalSource(
            label="S1",
            document="econometrics_note.pdf",
            page=1,
            chunk_id="econometrics-note",
            text="Econometrics applies statistical methods to economic data. [S1]",
            score=0.95,
        )
        rag = self._make_rag(
            local_answer="Econometrics applies statistical methods to economic data. [S1]",
            model_answer="Econometrics applies statistical methods to economic data.",
            web_answer="Econometrics applies statistical methods to economic data, as stated in the local note [S1].",
            local_sources=[local_source],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("What is econometrics?")

        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "local-grounded")
        self.assertTrue(result.diagnostics["used_local"])
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertEqual(result.diagnostics["evidence_winner"], "hybrid")
        self.assertEqual(len(rag.generator.final_calls), 1)
        self.assertEqual(result.diagnostics["answer_verification_status"], "verified")

    def test_repeated_question_uses_response_cache(self) -> None:
        rag = self._make_rag(local_answer="Local answer [S1]")

        first = rag.ask("What is in the file?")
        second = rag.ask("What is in the file?")

        self.assertNotIn("cache_hit", first.diagnostics)
        self.assertTrue(second.diagnostics["cache_hit"])
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(len(rag.retriever.calls), 1)
        self.assertEqual(len(rag.generator.local_calls), 1)

    def test_semantic_query_variation_uses_response_cache(self) -> None:
        rag = self._make_rag(local_answer="Cameroon has a total area answer [S1].")

        first = rag.ask("What is the size of Cameroon?")
        retrieval_calls_after_first = len(rag.retriever.calls)
        second = rag.ask("Total area of Cameroon")

        self.assertNotIn("cache_hit", first.diagnostics)
        self.assertTrue(second.diagnostics["cache_hit"])
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(len(rag.retriever.calls), retrieval_calls_after_first)

    def test_query_rewrite_skips_standalone_question_with_history(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Econometrics applies statistical methods to economic data.",
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask(
            "What is econometrics?",
            history=[ChatMessage(role="user", content="Tell me about Christophe Ley.")],
        )

        self.assertEqual(rag.generator.rewrite_calls, [])
        self.assertEqual(result.confidence, "model-only")

    def test_query_rewrite_runs_for_contextual_followup(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Econometrics assumptions include exogeneity.",
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask(
            "What about its assumptions?",
            history=[ChatMessage(role="user", content="Tell me about econometrics.")],
        )

        self.assertEqual(len(rag.generator.rewrite_calls), 1)
        self.assertEqual(result.confidence, "model-only")

    def test_query_rewrite_min_history_threshold_can_skip_short_context(self) -> None:
        self.assertFalse(
            _should_rewrite_query(
                "What about its assumptions?",
                [ChatMessage(role="user", content="Tell me about econometrics.")],
                min_history=2,
            )
        )

    def test_news_followup_resolves_pronoun_and_prior_uk_prime_minister_context(self) -> None:
        rag = self._make_rag(
            local_answer="This local answer should not be used [S1].",
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Reuters: UK Prime Minister Keir Starmer resigns",
                    url="https://www.reuters.com/world/uk/starmer-resigns",
                    content=(
                        "UK Prime Minister Keir Starmer resigned after pressure from his party "
                        "over the government's handling of the economy."
                    ),
                ),
                WebSource(
                    label="W2",
                    title="BBC News: Keir Starmer resignation",
                    url="https://www.bbc.com/news/uk-politics-starmer-resigns",
                    content="Keir Starmer resigned as UK prime minister after a cabinet split.",
                ),
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is the Prime Minister of the UK?"),
            ChatMessage(
                role="assistant",
                content="Confidence: High\n\nThe current prime minister of the United Kingdom is Keir Starmer [W1].",
            ),
            ChatMessage(role="user", content="Search news and tell me whether he has resigned."),
            ChatMessage(
                role="assistant",
                content="Confidence: High\n\nReuters reports that UK Prime Minister Keir Starmer resigned [W1].",
            ),
        ]

        result = rag.ask("Search Reuters and tell me why.", history=history)

        self.assertTrue(result.diagnostics["conversation_followup"])
        self.assertTrue(result.diagnostics["news_intent"])
        self.assertIn("UK Prime Minister Keir Starmer", result.diagnostics["resolved_query"])
        self.assertIn("Reuters", result.diagnostics["web_queries"][0])
        self.assertIn("Keir Starmer", result.diagnostics["web_queries"][0])
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertFalse(result.diagnostics["parallel_model_with_web"])
        self.assertTrue(result.diagnostics["model_skipped_for_current_web"])
        self.assertEqual(rag.generator.final_calls, [])
        self.assertTrue(result.used_web)
        self.assertIn("Keir Starmer resigned", result.answer)
        self.assertIn("[W1]", result.answer)

    def test_news_channels_followup_prefers_major_news_sources(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="AP News: UK prime minister resignation",
                    url="https://apnews.com/article/uk-prime-minister-resignation",
                    content="AP reports on the resignation of UK Prime Minister Keir Starmer.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is the Prime Minister of the UK?"),
            ChatMessage(role="assistant", content="Keir Starmer is the UK Prime Minister."),
        ]

        result = rag.ask("Search news channels and tell me whether he has resigned.", history=history)

        queries = " ".join(result.diagnostics["web_queries"])
        self.assertIn("Reuters", queries)
        self.assertIn("BBC News", queries)
        self.assertIn("Sky News", queries)
        self.assertIn("UK Prime Minister Keir Starmer", result.diagnostics["resolved_query"])
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)

    def test_country_memory_rewrites_government_role_followups(self) -> None:
        rag = self._make_rag(
            local_answer="Local files should not be used [S1].",
            model_answer=MODEL_UNKNOWN,
            web_answer="Judith Suminwa is the Prime Minister of the Democratic Republic of the Congo [W1].",
            web_sources=[
                WebSource(
                    label="W1",
                    title="DR Congo Prime Minister",
                    url="https://www.primature.cd/",
                    content="Judith Suminwa is Prime Minister of the Democratic Republic of the Congo.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of the RDC?"),
            ChatMessage(
                role="assistant",
                content="Félix Tshisekedi is president of the Democratic Republic of the Congo [W1].",
            ),
        ]

        result = rag.ask("Who is the prime minister?", history=history)

        self.assertTrue(result.diagnostics["conversation_followup"])
        self.assertEqual(result.diagnostics["conversation_country"], "Democratic Republic of the Congo")
        self.assertIn(
            "Prime Minister of the Democratic Republic of the Congo",
            result.diagnostics["resolved_query"],
        )
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertFalse(result.diagnostics["parallel_model_with_web"])
        self.assertTrue(result.diagnostics["model_skipped_for_current_web"])

    def test_country_memory_rewrites_age_at_presidency_followup(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Félix Tshisekedi age answer [W1].",
            web_sources=[
                WebSource(
                    label="W1",
                    title="Félix Tshisekedi biography",
                    url="https://example.com/felix",
                    content=(
                        "Félix Tshisekedi was born 13 June 1963. "
                        "He became President of the Democratic Republic of the Congo on 24 January 2019."
                    ),
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of the RDC?"),
            ChatMessage(
                role="assistant",
                content="Félix Tshisekedi is president of the Democratic Republic of the Congo [W1].",
            ),
        ]

        result = rag.ask("How old will he become the president?", history=history)

        self.assertTrue(result.diagnostics["conversation_followup"])
        self.assertIn(
            "How old was Félix Tshisekedi when he became President of the Democratic Republic of the Congo?",
            result.diagnostics["resolved_query"],
        )
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertTrue(result.diagnostics["parallel_model_with_web"])
        self.assertIn("55 years old", result.answer)
        self.assertIn("[W1]", result.answer)

    def test_role_followup_prefers_explicit_president_over_latest_prime_minister(self) -> None:
        rag = self._make_rag(
            local_answer="Local files should not be used [S1].",
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Félix Tshisekedi biography",
                    url="https://example.com/felix",
                    content=(
                        "Félix Tshisekedi became President of the Democratic Republic "
                        "of the Congo on 24 January 2019."
                    ),
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of RDC?"),
            ChatMessage(
                role="assistant",
                content="The current president of the Democratic Republic of the Congo is Félix Tshisekedi [W1].",
            ),
            ChatMessage(role="user", content="Who is the prime minister?"),
            ChatMessage(
                role="assistant",
                content="The current prime minister of the Democratic Republic of the Congo is Judith Suminwa [W2].",
            ),
        ]

        result = rag.ask("When did the president come into power?", history=history)

        self.assertIn(
            "When did Félix Tshisekedi become President of the Democratic Republic of the Congo?",
            result.diagnostics["resolved_query"],
        )
        self.assertNotIn("Judith Suminwa", result.diagnostics["resolved_query"])
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertIn("Félix Tshisekedi became President", result.answer)
        self.assertIn("24 January 2019", result.answer)
        self.assertIn("[W1]", result.answer)

    def test_direct_office_power_question_answers_date_and_person(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Félix Tshisekedi biography",
                    url="https://example.com/felix",
                    content=(
                        "Félix Tshisekedi is the President of the Democratic Republic of the Congo. "
                        "He took office on 24 January 2019 after the December 2018 election."
                    ),
                )
            ],
        )

        result = rag.ask("When did the president of RDC come into power?")

        self.assertTrue(result.used_web)
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertIn("Félix Tshisekedi became President of the Democratic Republic of the Congo", result.answer)
        self.assertIn("24 January 2019", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertFalse(result.answer.startswith("Confidence:"))

    def test_public_list_question_skips_irrelevant_local_chunks(self) -> None:
        rag = self._make_rag(
            local_answer="sas-certification-prep-guide select name label=Country [S1]",
            model_answer=MODEL_UNKNOWN,
            local_sources=[LOCAL_SOURCE],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Volcanic lakes around the world",
                    url="https://example.com/volcanic-lakes",
                    content=(
                        "Examples of volcanic lakes include Lake Toba in Indonesia, "
                        "Lake Taupo in New Zealand, Crater Lake in the United States, "
                        "Lake Nyos in Cameroon, and Lake Kivu in Africa."
                    ),
                )
            ],
        )

        result = rag.ask("Name the volcanic lakes in the world")

        self.assertEqual(result.diagnostics["search_plan_intent"], "public_knowledge")
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertNotIn("sas-certification", result.answer)
        self.assertIn("Lake Toba", result.answer)
        self.assertIn("[W1]", result.answer)

    def test_public_topic_followup_uses_previous_topic_for_web_search(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Volcanic lake eruption history",
                    url="https://example.com/volcanic-lake-eruptions",
                    content=(
                        "Volcanic lakes with historical volcanic activity include "
                        "Lake Toba, Lake Taupo, Crater Lake, Lake Kivu, and Lake Nyos."
                    ),
                )
            ],
        )
        first = rag.ask("Name the volcanic lakes in the world")
        history = [
            ChatMessage(role="user", content="Name the volcanic lakes in the world"),
            ChatMessage(role="assistant", content=first.answer),
        ]

        result = rag.ask(
            "Which ones have erupted in history?",
            history=history,
            conversation_state=first.conversation_state,
        )

        self.assertEqual(result.diagnostics["resolved_query"], "Which volcanic lakes have erupted in history?")
        self.assertEqual(result.diagnostics["search_plan_intent"], "public_knowledge")
        self.assertTrue(any("volcanic lakes" in query for query in result.diagnostics["web_queries"]))
        self.assertIn("Lake Toba", result.answer)

    def test_public_topic_head_followup_keeps_active_modifier(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Volcanic lake eruptions in recent history",
                    url="https://example.com/recent-volcanic-lake-eruptions",
                    content=(
                        "Volcanic lakes associated with eruptions in recent history include "
                        "Lake Taupo, Lake Toba, and crater lakes in active volcanic systems."
                    ),
                )
            ],
        )
        state = rag.ask("Name the volcanic lakes in the world").conversation_state

        result = rag.ask(
            "which lakes have erupted around the world not only in the usa in the last 50 years?",
            conversation_state=state,
        )

        self.assertIn("which volcanic lakes have erupted", result.diagnostics["resolved_query"])
        self.assertTrue(any("volcanic lakes" in query for query in result.diagnostics["web_queries"]))
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertIn("Lake Taupo", result.answer)

    def test_age_followup_keeps_explicit_role_phrase(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Félix Tshisekedi MONUC biography",
                    url="https://example.com/felix-monuc",
                    content=(
                        "Félix Tshisekedi was born 13 June 1963. "
                        "Félix Tshisekedi became the defense general of MONUC on 1 February 2005."
                    ),
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of the RDC?"),
            ChatMessage(
                role="assistant",
                content="Félix Tshisekedi is president of the Democratic Republic of the Congo [W1].",
            ),
        ]

        result = rag.ask("How old he become the defense general of MONUC?", history=history)

        self.assertIn(
            "How old was Félix Tshisekedi when he became the defense general of MONUC?",
            result.diagnostics["resolved_query"],
        )
        self.assertNotIn("President of the Democratic Republic of the Congo", result.diagnostics["resolved_query"])
        self.assertIn("41 years old", result.answer)
        self.assertIn("[W1]", result.answer)

    def test_cameroon_president_followup_uses_person_country_memory(self) -> None:
        president_sources = [
            WebSource(
                label="W1",
                title="President of Cameroon",
                url="https://example.cm/presidency",
                content=(
                    "Paul Biya, President of Cameroon, has served as president of Cameroon "
                    "since 6 November 1982."
                ),
            )
        ]
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=president_sources,
        )

        first = rag.ask("Who is the president of Cameroon?")

        self.assertIn("Paul Biya", first.answer)
        self.assertIn("[W1]", first.answer)

        history = [
            ChatMessage(role="user", content="Who is the president of Cameroon?"),
            ChatMessage(role="assistant", content=first.answer),
        ]
        second = rag.ask("When did he become the president?", history=history)

        self.assertTrue(second.diagnostics["conversation_followup"])
        self.assertEqual(second.diagnostics["conversation_country"], "Cameroon")
        self.assertIn("When did Paul Biya become President of Cameroon?", second.diagnostics["resolved_query"])
        self.assertNotIn("France", second.diagnostics["resolved_query"])
        self.assertFalse(second.diagnostics["local_retrieval_skipped"])
        self.assertIn("6 November 1982", second.answer)
        self.assertIn("[W1]", second.answer)

    def test_source_like_assistant_text_does_not_switch_country_memory(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Paul Biya biography",
                    url="https://example.cm/paul-biya",
                    content="Paul Biya has served as president of Cameroon since 6 November 1982.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is the president of Cameroon?"),
            ChatMessage(
                role="assistant",
                content=(
                    "President of Cameroon appears in the evidence. "
                    "Sources also mention President of France and Emmanuel Macron."
                ),
            ),
        ]

        result = rag.ask("When did he become the president?", history=history)

        self.assertEqual(result.diagnostics["conversation_country"], "Cameroon")
        self.assertIn("President of Cameroon", result.diagnostics["resolved_query"])
        self.assertNotIn("France", result.diagnostics["resolved_query"])

    def test_scientific_followup_resolves_it_to_active_topic_and_uses_plan(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model background answer.",
            web_sources=[
                WebSource(
                    label="W1",
                    title="Markov Chain Monte Carlo",
                    url="https://example.edu/mcmc",
                    content="Markov Chain Monte Carlo methods are associated with Metropolis and Ulam.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="What is Markov Chain Monte Carlo?"),
            ChatMessage(role="assistant", content="Markov Chain Monte Carlo is a sampling method."),
        ]

        result = rag.ask("Who invented it?", history=history)

        self.assertTrue(result.diagnostics["conversation_followup"])
        self.assertEqual(result.diagnostics["resolved_query"], "Who introduced Markov Chain Monte Carlo?")
        self.assertEqual(result.diagnostics["search_plan_intent"], "scientific_definition")
        self.assertIn("arXiv", result.diagnostics["search_plan_preferred_sources"])
        self.assertTrue(result.diagnostics["search_plan_need_local"])
        self.assertFalse(result.diagnostics["search_plan_need_web"])
        self.assertIn("Markov Chain Monte Carlo arxiv paper", result.diagnostics["web_queries"])

    def test_named_passport_issue_question_answers_from_local_document_evidence(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-issue",
            text=(
                "Passport REPUBLIC OF SAMPLE Name ALEX EXAMPLE "
                "Date de d6livrance/ Date of lssue 3t.03.2022 "
                "Date d'expiration / Date 0f explry 3r.03.2027 "
                "Place of issue EXAMPLETOWN"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        result = rag.ask("When was Alex Example issued a passport")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("31.03.2022", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])

    def test_passport_expiry_followup_answers_from_local_document_evidence(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-expiry",
            text=(
                "Passport REPUBLIC OF SAMPLE Name ALEX EXAMPLE "
                "Date de d6livrance/ Date of lssue 3t.03.2022 "
                "Date d'expiration / Date 0f explry 3r.03.2027 "
                "Place of issue EXAMPLETOWN"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        history = [
            ChatMessage(role="user", content="Is his passport in the local files?"),
            ChatMessage(
                role="assistant",
                content="Yes, Alex Example's passport is in the local files [S1].",
            ),
        ]

        result = rag.ask("When does his passport expire", history=history)

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("31.03.2027", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])

    def test_passport_issue_location_question_answers_from_local_document_evidence(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-issue-place",
            text=(
                "Passport REPUBLIC OF SAMPLE Name ALEX EXAMPLE "
                "Date de d6livrance/ Date of lssue 3t.03.2022 "
                "Date d'expiration / Date 0f explry 3r.03.2027 "
                "Lieu de d6llvrance/ Place ol issue BRUSSELS 13. Signature"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        result = rag.ask("Where was Alex Example's passport issued")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("Brussels", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])

    def test_passport_issue_location_followup_stays_local_to_active_document(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-issue-place-followup",
            text=(
                "Passport REPUBLIC OF SAMPLE Name ALEX EXAMPLE "
                "Date de d6livrance/ Date of lssue 3t.03.2022 "
                "Date d'expiration / Date 0f explry 3r.03.2027 "
                "Lieu de d6llvrance/ Place ol issue BRUSSELS 13. Signature"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        first = rag.ask("When was Alex Example issued a passport")
        history = [
            ChatMessage(role="user", content="When was Alex Example issued a passport"),
            ChatMessage(role="assistant", content=first.answer),
        ]

        result = rag.ask(
            "the location where it was issued",
            history=history,
            conversation_state=first.conversation_state,
        )

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("Brussels", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.model_calls, [])

    def test_named_birthplace_question_prefers_local_passport_evidence(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-birth-place",
            text=(
                "Passport REPUBLIC OF SAMPLE Name ALEX JORDAN SAMPLE "
                "Date of birth 01.01.2000 Sexe Sex 6. Lieu de naisbance/ Place of blrth M SAMPLETOWN "
                "7. Date de d6livrance/ Date of lssue 3t.03.2022"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        result = rag.ask("Where was Alex Jordan Sample born?")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("Sampletown", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])

    def test_birthplace_question_based_on_passport_answers_directly(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-birth-place-explicit",
            text=(
                "Passport REPUBLIC OF SAMPLE Name ALEX JORDAN SAMPLE "
                "Date of birth 01.01.2000 Sexe Sex 6. Lieu de naisbance/ Place of blrth M SAMPLETOWN "
                "7. Date de d6livrance/ Date of lssue 3t.03.2022"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        result = rag.ask("Where was Alex Jordan Sample born based on the passport")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("Sampletown", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])

    def test_origin_question_prefers_local_passport_evidence_before_model(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-origin",
            text=(
                "Passport REPUBLIC OF SAMPLE Name ALEX JORDAN SAMPLE "
                "Nationality NIGERIAN 4. Date of birth 01.01.2000 "
                "Place of birth SAMPLETOWN"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        result = rag.ask("Where is Alex from")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("Nigeria", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])

    def test_birthplace_question_uses_passport_when_only_one_name_token_matches(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-partial-name",
            text=(
                "Passport REPUBLIC OF SAMPLE Name MORGAN VALE "
                "Date of birth 01.01.2000 Place of birth SAMPLETOWN"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        result = rag.ask("Where was Robin Morgan born?")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("Sampletown", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertEqual(rag.generator.model_calls, [])

    def test_bare_reordered_name_can_use_passport_identity_evidence(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-reordered-name",
            text=(
                "Passport REPUBLIC OF SAMPLE Name MORGAN VALE "
                "Date of birth 01.01.2000 Place of birth SAMPLETOWN"
            ),
            score=0.94,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([passport_source])

        result = rag.ask("Vale Robin Morgan")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["sample_passport.pdf"])
        self.assertTrue(result.diagnostics["local_sufficient"])
        self.assertEqual(rag.generator.model_calls, [])

    def test_bare_person_query_resets_government_memory_and_uses_person_plan(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Alex Jordan profile",
                    url="https://example.edu/alex-jordan",
                    content="Alex Jordan has a university research profile.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is the Minister of Finance of France?"),
            ChatMessage(role="assistant", content="The finance minister of France is listed by the government."),
        ]

        result = rag.ask("Alex Jordan", history=history)

        self.assertFalse(result.diagnostics["conversation_followup"])
        self.assertEqual(result.diagnostics["conversation_country"], "")
        self.assertEqual(result.diagnostics["conversation_person"], "Alex Jordan")
        self.assertEqual(result.diagnostics["search_plan_intent"], "person")
        self.assertIn("ORCID", result.diagnostics["search_plan_preferred_sources"])
        self.assertIn("Alex Jordan GitHub", result.diagnostics["web_queries"])
        self.assertNotIn("France", result.diagnostics["resolved_query"])

    def test_government_statement_uses_official_source_plan(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="French government finance minister",
                    url="https://www.gouvernement.fr/",
                    content="The Minister of Finance of France is listed on the government website.",
                )
            ],
        )

        result = rag.ask("Minister of Finance France")

        self.assertEqual(result.diagnostics["conversation_country"], "France")
        self.assertEqual(result.diagnostics["search_plan_intent"], "government")
        self.assertTrue(result.diagnostics["search_plan_need_local"])
        self.assertIn("Government", result.diagnostics["search_plan_preferred_sources"])
        self.assertIn("Minister of Finance of France official government", result.diagnostics["web_queries"])

    def test_country_memory_rewrites_defence_minister_followup(self) -> None:
        rag = self._make_rag(
            local_answer="Local files should not be used [S1].",
            model_answer=MODEL_UNKNOWN,
            web_answer="The defence minister answer is in web evidence [W1].",
            web_sources=[
                WebSource(
                    label="W1",
                    title="DR Congo defence ministry",
                    url="https://example.gov.cd/defence",
                    content="The Minister of Defence of the Democratic Republic of the Congo is listed by the government.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of the RDC?"),
            ChatMessage(role="assistant", content="The RDC president is Félix Tshisekedi."),
        ]

        result = rag.ask("Who is the defence minister?", history=history)

        self.assertIn(
            "Minister of Defence of the Democratic Republic of the Congo",
            result.diagnostics["resolved_query"],
        )
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertFalse(result.diagnostics["parallel_model_with_web"])
        self.assertTrue(result.diagnostics["model_skipped_for_current_web"])

    def test_country_memory_expands_source_followup(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Reuters has reporting about the DR Congo government [W1].",
            web_sources=[
                WebSource(
                    label="W1",
                    title="Reuters DR Congo government",
                    url="https://www.reuters.com/world/africa/dr-congo-government",
                    content="Reuters reports on the Democratic Republic of the Congo government.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of the RDC?"),
            ChatMessage(role="assistant", content="Félix Tshisekedi is president of the Democratic Republic of the Congo."),
            ChatMessage(role="user", content="Who is the prime minister?"),
            ChatMessage(role="assistant", content="Judith Suminwa is Prime Minister of the Democratic Republic of the Congo."),
        ]

        result = rag.ask("Search Reuters", history=history)

        self.assertTrue(result.diagnostics["news_intent"])
        self.assertIn(
            "Democratic Republic of the Congo government",
            result.diagnostics["resolved_query"],
        )
        self.assertIn("Reuters Democratic Republic of the Congo government", result.diagnostics["web_queries"][0])
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertGreater(len(rag.retriever.calls), 0)

    def test_recent_country_switch_overrides_previous_country_memory(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="France defence minister answer [W1].",
            web_sources=[
                WebSource(
                    label="W1",
                    title="France defence minister",
                    url="https://www.gouvernement.fr/",
                    content="France defence minister information.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of the RDC?"),
            ChatMessage(role="assistant", content="The RDC president is Félix Tshisekedi."),
            ChatMessage(role="user", content="Now France"),
            ChatMessage(role="assistant", content="France politics is the active topic."),
        ]

        result = rag.ask("Who is the defence minister?", history=history)

        self.assertEqual(result.diagnostics["conversation_country"], "France")
        self.assertIn("Minister of Defence of France", result.diagnostics["resolved_query"])
        self.assertNotIn("Democratic Republic of the Congo", result.diagnostics["resolved_query"])

    def test_self_contained_acronym_question_does_not_inherit_country_memory(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="The ICC is the International Criminal Court.",
            web_answer="The ICC is the International Criminal Court [W1].",
            web_sources=[
                WebSource(
                    label="W1",
                    title="International Criminal Court",
                    url="https://www.icc-cpi.int/",
                    content="The International Criminal Court investigates and tries international crimes.",
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is president of the RDC?"),
            ChatMessage(role="assistant", content="Félix Tshisekedi is president of the Democratic Republic of the Congo."),
        ]

        result = rag.ask("What is the ICC?", history=history)

        self.assertFalse(result.diagnostics["conversation_followup"])
        self.assertEqual(result.diagnostics["resolved_query"], "What is the ICC?")
        self.assertEqual(result.diagnostics["conversation_country"], "Democratic Republic of the Congo")

    def test_lightweight_greeting_skips_retrieval_generation_and_web(self) -> None:
        rag = self._make_rag(local_answer="Local answer [S1]")
        result = rag.ask("hi there")

        self.assertEqual(result.confidence, "greeting")
        self.assertFalse(result.used_web)
        self.assertIn("Verilume", result.answer)
        self.assertEqual(result.diagnostics["pipeline"], "intent_router")
        self.assertEqual(rag.retriever.calls, [])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_conversation_identity_and_capability_routes_skip_rag(self) -> None:
        examples = [
            ("Thanks", "conversation", "welcome"),
            ("Who are you?", "identity", "local-first"),
            ("What can you do?", "capability", "search local PDFs"),
        ]
        for prompt, route, expected in examples:
            with self.subTest(prompt=prompt):
                rag = self._make_rag(local_answer="Local answer [S1]")
                result = rag.ask(prompt)

                self.assertEqual(result.confidence, route)
                self.assertFalse(result.used_web)
                self.assertIn(expected, result.answer)
                self.assertEqual(rag.retriever.calls, [])
                self.assertEqual(rag.generator.local_calls, [])
                self.assertEqual(rag.generator.model_calls, [])
                self.assertEqual(rag.web_search.queries, [])

    def test_local_file_question_searches_expanded_keywords_and_does_not_use_ai_or_web(
        self,
    ) -> None:
        rag = self._make_rag(local_answer="This should not be called.", local_sources=[])
        rag.retriever = SequentialLocalRetriever([[], []])

        result = rag.ask("Is Alex's language certificate in the local files?")

        self.assertEqual(result.answer, LOCAL_FILE_NOT_FOUND)
        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.used_web)
        self.assertEqual(result.local_sources, [])
        self.assertEqual(result.web_sources, [])
        self.assertEqual(len(rag.retriever.calls), 2)
        self.assertIn("language certificate", rag.retriever.calls[0][0][0].lower())
        self.assertIn("sproochentest", rag.retriever.calls[1][0][0].lower())
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_local_file_question_answers_from_expanded_local_hits(self) -> None:
        expanded_source = LocalSource(
            label="S1",
            document="language_certificate.pdf",
            page=1,
            chunk_id="language-certificate",
            text="Alex Sample Sproochentest language certificate result.",
            score=0.82,
        )
        rag = self._make_rag(local_answer="This should not be called.", local_sources=[])
        rag.retriever = SequentialLocalRetriever([[], [expanded_source]])

        result = rag.ask("Which document contains my language certificate?")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("language_certificate.pdf", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual(
            [source.document for source in result.local_sources], ["language_certificate.pdf"]
        )
        self.assertEqual(len(rag.retriever.calls), 2)
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_local_retrieval_runs_even_when_plan_prefers_model_or_web(self) -> None:
        class ModelFirstPlanner:
            def plan(self, interpretation):
                return ResolvedSearchPlan(
                    intent="public_knowledge",
                    need_local=False,
                    need_web=False,
                    need_model=True,
                )

        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Regression models relationships between variables.",
            local_sources=[],
        )
        rag.search_planner = ModelFirstPlanner()
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("What is regression?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertTrue(result.diagnostics["local_retrieval_forced_by_policy"])
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertIn("Regression", result.answer)

    def test_explicit_filename_summary_uses_local_evidence(self) -> None:
        diploma_source = LocalSource(
            label="S1",
            document="Damian_diploma.pdf",
            page=1,
            chunk_id="diploma-1",
            text="Damian Mingo Ndiwago completed the diploma in statistics with distinction.",
            score=0.91,
        )
        rag = self._make_rag(
            local_answer="The diploma says Damian completed statistics with distinction.",
            model_answer="Model answer that should not be used.",
            local_sources=[diploma_source],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("Summarise Damian_diploma.pdf")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("statistics", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["Damian_diploma.pdf"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.local_calls), 1)
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_examples_from_local_files_uses_local_evidence(self) -> None:
        example_source = LocalSource(
            label="S1",
            document="examples.pdf",
            page=2,
            chunk_id="examples-2",
            text="The local file gives examples of regression using housing prices and income.",
            score=0.88,
        )
        rag = self._make_rag(
            local_answer="One local example is regression using housing prices and income [S1].",
            model_answer="Model answer that should not be used.",
            local_sources=[example_source],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("Give examples from the local files")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("housing prices", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["examples.pdf"])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(len(rag.generator.local_calls), 1)
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_local_inventory_question_answers_from_document_stats(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "documents"
            chroma_dir = root / "chroma"
            manifest_path = root / "manifest.json"
            docs_dir.mkdir()
            diploma = docs_dir / "Damian_diploma.pdf"
            notes = docs_dir / "notes.txt"
            diploma.write_bytes(b"%PDF-1.4\n")
            notes.write_text("Local notes", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        str(diploma): {"pdf_pages": 3, "chunks": 3},
                        str(notes): {"pdf_pages": 0, "chunks": 1},
                    }
                ),
                encoding="utf-8",
            )

            rag = self._make_rag(
                local_answer=LOCAL_UNKNOWN,
                model_answer="Model answer that should not be used.",
                local_sources=[],
            )
            rag.settings = AppSettings(
                hf_token="token",
                enable_web_search=False,
                docs_dir=docs_dir,
                chroma_dir=chroma_dir,
                manifest_path=manifest_path,
            )

            result = rag.ask("how many local files/documents are there")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("2 uploaded documents", result.answer)
        self.assertIn("4 indexed chunks", result.answer)
        self.assertIn("3 PDF pages", result.answer)
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_local_file_fact_question_answers_from_local_evidence_without_model_or_web(self) -> None:
        certificate_source = LocalSource(
            label="S1",
            document="exam_payment_certificate_Sproochentest_2025.pdf",
            page=1,
            chunk_id="certificate",
            text=(
                "Luxembourg, le 30/01/2026. "
                "Alex Jordan Sample a regle le montant de 75 EUR en date du 12/10/2025 "
                "en vue de la passation de l'examen Sproochentest Letzebuergesch, "
                "session 2025-12-17-Me."
            ),
            score=0.96,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[certificate_source],
        )

        result = rag.ask("What is the exam date on the Sproochentest exam payment certificate?")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("2025-12-17", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.label for source in result.local_sources], ["S1"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_local_file_fact_question_anchors_to_explicit_filename_for_ocr_chunk(self) -> None:
        scanned_source = LocalSource(
            label="S1",
            document="scanned-smoke.pdf",
            page=1,
            chunk_id="scan-chunk",
            text="VERILUMEOCRTOKENAURORA-731",
            score=0.3098,
        )
        image_source = LocalSource(
            label="S2",
            document="ocr-smoke.png",
            page=1,
            chunk_id="image-chunk",
            text="VERILUMEOCRTOKENAURORA-731",
            score=0.1647,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Model answer that should not be used.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.retriever = FakeRetriever([scanned_source, image_source])

        result = rag.ask(
            "In the uploaded file scanned-smoke.pdf, what OCR token appears in the document?"
        )

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("AURORA-731", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["scanned-smoke.pdf"])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_local_gap_uses_model_before_web_when_web_is_enabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Econometrics applies statistical methods to economic data.",
            web_answer="Econometrics applies statistical methods to economic data and combines statistics and economics [W1]",
        )
        result = rag.ask("What is econometrics?")

        self.assertTrue(result.used_web)
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertEqual(result.diagnostics["evidence_winner"], "hybrid")
        self.assertIn("Econometrics", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertGreater(len(rag.web_search.queries), 0)

    def test_model_knowledge_and_web_search_run_in_parallel_after_local_gap(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Parallel model answer.",
            web_answer="Parallel web answer [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Parallel source",
                    url="https://gouvernement.lu/parallel",
                    content="Parallel source content.",
                )
            ],
        )
        rag.generator = SlowModelGenerator(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Parallel model answer.",
            web_answer="Parallel web answer [W1]",
            delay_seconds=0.15,
        )
        rag.settings = AppSettings(
            hf_token="token",
            tavily_api_key="key",
            web_search_max_results=1,
        )
        rag.web_search = SlowWebSearch(
            [
                WebSource(
                    label="W1",
                    title="Parallel source",
                    url="https://gouvernement.lu/parallel",
                    content="Parallel source content.",
                )
            ],
            delay_seconds=0.15,
        )

        started = time.perf_counter()
        result = rag.ask("Search the web about parallel evidence")
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.27)
        self.assertTrue(result.used_web)
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertEqual(result.diagnostics["web_count"], 1)

    def test_model_knowledge_fallback_handles_generic_questions_when_web_is_disabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Econometrics applies statistical methods to economic data.",
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        result = rag.ask("What is econometrics?")

        self.assertEqual(result.confidence, "model-only")
        self.assertFalse(result.used_web)
        self.assertIn("Econometrics", result.answer)
        self.assertIn("not externally verified", result.answer)
        self.assertEqual(result.web_sources, [])

    def test_model_answer_with_us_spelling_answers_uk_spelling_query_before_web(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=(
                "Optimization is the process of making a system, design, or decision "
                "as effective as possible."
            ),
            web_answer="Optimization is the process of making a system, design, or decision as effective as possible [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="optimization - Glossary",
                    url="https://example.gov/glossary/fiscal-optimization",
                    content="Optimization is the process of making a system, design, or decision as effective as possible.",
                )
            ],
        )

        result = rag.ask("what is optimisation")

        self.assertTrue(result.used_web)
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertGreater(len(rag.web_search.queries), 0)
        self.assertIn("Optimization", result.answer)
        self.assertEqual(result.local_sources, [])

    def test_generic_definition_filters_private_passport_local_source(self) -> None:
        passport_source = LocalSource(
            label="S1",
            document="sample_passport.pdf",
            page=1,
            chunk_id="passport-private",
            text=(
                "Passport REPUBLIC OF SAMPLE Name MORGAN VALE "
                "Date of birth 01.01.2000 Place of birth SAMPLETOWN Passport number SAMPLE123"
            ),
            score=0.99,
        )
        rag = self._make_rag(
            local_answer="The local passport text is the answer [S1].",
            model_answer="Optimization is the process of finding the best feasible option.",
            local_sources=[passport_source],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("define optimisation")

        self.assertEqual(result.confidence, "model-only")
        self.assertFalse(result.used_web)
        self.assertEqual(result.local_sources, [])
        self.assertEqual(rag.generator.local_calls, [])
        self.assertIn("Optimization", result.answer)
        self.assertNotIn("passport", result.answer.lower())
        self.assertNotIn("[S1]", result.answer)

    def test_local_and_model_gap_automatically_uses_web_when_enabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="I couldn't find any specific information about Florian Felice.",
            web_answer="Florian Felice is described in the web evidence [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Florian Felice profile",
                    url="https://example.com/florian-felice",
                    content="Florian Felice is described in this profile.",
                )
            ],
        )

        result = rag.ask("Florian Felice")

        self.assertTrue(result.used_web)
        self.assertEqual(result.diagnostics["web_reason"], "standard_static_hybrid")
        self.assertFalse(result.diagnostics["model_sufficient"])
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertGreater(len(rag.web_search.queries), 0)
        self.assertIn("[W1]", result.answer)

    def test_low_confidence_miss_is_not_reused_after_web_is_enabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[],
            web_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        first = rag.ask("Florian Felice")

        rag.settings = AppSettings(
            hf_token="token",
            tavily_api_key="key",
            enable_web_search=True,
        )
        rag.web_search = FakeWebSearch(
            [
                WebSource(
                    label="W1",
                    title="Florian Felice profile",
                    url="https://example.com/florian-felice",
                    content="Florian Felice is described in this profile.",
                )
            ]
        )
        second = rag.ask("Florian Felice")

        self.assertEqual(first.confidence, "low")
        self.assertNotIn("cache_hit", second.diagnostics)
        self.assertTrue(second.used_web)
        self.assertIn("[W1]", second.answer)

    def test_model_knowledge_answers_stable_identity_when_web_is_disabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Ada Lovelace was born on 10 December 1815.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("When was Ada Lovelace born?")

        self.assertEqual(result.confidence, "model-only")
        self.assertFalse(result.used_web)
        self.assertIn("Ada Lovelace", result.answer)
        self.assertIn("10 December 1815", result.answer)
        self.assertEqual(result.local_sources, [])
        self.assertEqual(result.web_sources, [])
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertEqual(rag.web_search.queries, [])

    def test_identity_fact_question_ignores_name_only_local_hits_before_model(self) -> None:
        name_only_source = LocalSource(
            label="S1",
            document="ada-notes.pdf",
            page=1,
            chunk_id="ada-notes",
            text="Ada Lovelace wrote notes on the Analytical Engine.",
            score=0.92,
            metadata={"query_overlap": 2, "fast_rerank_score": 0.91},
        )
        rag = self._make_rag(
            local_answer="The local file discusses Ada Lovelace notes [S1].",
            model_answer="Ada Lovelace was born on 10 December 1815.",
            local_sources=[name_only_source],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("When was Ada Lovelace born?")

        self.assertEqual(result.confidence, "model-only")
        self.assertFalse(result.used_web)
        self.assertEqual(result.local_sources, [])
        self.assertIn("10 December 1815", result.answer)
        self.assertNotIn("[S1]", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 1)

    def test_model_knowledge_answers_stable_geography_when_web_is_disabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Russia is the largest country in Europe by land area.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("The largest country in Europe")

        self.assertEqual(result.confidence, "model-only")
        self.assertFalse(result.used_web)
        self.assertIn("Russia", result.answer)
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertEqual(result.diagnostics["evidence_winner"], "model_knowledge")
        self.assertIn("model_knowledge", result.diagnostics["evidence_streams"])
        self.assertFalse(result.diagnostics["web_enabled"])

    def test_model_knowledge_answers_definitions_when_web_is_disabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=(
                "Bayesian inference updates prior beliefs with observed evidence "
                "to produce a posterior distribution."
            ),
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("What is Bayesian inference?")

        self.assertEqual(result.confidence, "model-only")
        self.assertIn("posterior", result.answer.lower())
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertEqual(result.diagnostics["evidence_winner"], "model_knowledge")

    def test_model_knowledge_answer_does_not_gain_local_citation_footer(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=(
                "Spectral analysis decomposes a signal into its constituent frequencies."
            ),
            local_sources=[LOCAL_SOURCE],
            web_sources=[],
        )

        result = rag.ask("What is spectral analysis?")

        self.assertEqual(result.confidence, "model-only")
        self.assertIn("Source: AI knowledge (not externally verified)", result.answer)
        self.assertNotIn("[S1]", result.answer)
        self.assertEqual(result.local_sources, [])
        self.assertFalse(result.used_web)
        self.assertFalse(result.diagnostics["used_local"])
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertEqual(result.diagnostics["evidence_winner"], "model_knowledge")

    def test_current_fact_without_web_does_not_fake_model_verification(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Sam Altman is the current CEO of OpenAI.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("Who is the current CEO of OpenAI?")

        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.used_web)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertFalse(result.diagnostics["model_knowledge_available"])
        self.assertFalse(result.diagnostics["used_model_knowledge"])
        self.assertTrue(result.diagnostics["model_skipped_for_current_web"])
        self.assertNotIn("Sam Altman", result.answer)
        self.assertIn("AI knowledge is not reliable enough for current facts", result.answer)

    def test_all_available_evidence_streams_are_reported_when_web_is_enabled(self) -> None:
        rag = self._make_rag(
            local_answer="Local econometrics note [S1].",
            model_answer="Model econometrics background.",
            web_answer="Local econometrics note with model econometrics background and a web econometrics source [S1] [W1].",
            local_sources=[LOCAL_SOURCE],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Econometrics overview",
                    url="https://example.edu/econometrics",
                    content="Econometrics uses statistics to study economic data.",
                )
            ],
        )

        result = rag.ask("Search the web about econometrics")

        self.assertTrue(result.used_web)
        self.assertTrue(result.diagnostics["used_local"])
        self.assertTrue(result.diagnostics["used_model_knowledge"])
        self.assertTrue(result.diagnostics["used_web"])
        self.assertIn("local", result.diagnostics["evidence_streams"])
        self.assertIn("model_knowledge", result.diagnostics["evidence_streams"])
        self.assertIn("web", result.diagnostics["evidence_streams"])
        self.assertEqual(result.diagnostics["evidence_winner"], "hybrid")

    def test_web_fallback_filters_to_used_web_citations(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Web answer [W1]",
            web_sources=[
                WebSource(
                    label="W1",
                    title="Dylan Jordan",
                    url="https://example.com/wrong",
                    content="Wrong person",
                ),
                WebSource(
                    label="W2",
                    title="Alex Jordan profile",
                    url="https://example.com/alex-jordan",
                    content="Alex Jordan is mentioned here.",
                ),
            ],
        )
        result = rag.ask("Who is Alex Jordan?")

        self.assertIn(result.confidence, {"medium", "high"})
        self.assertTrue(result.used_web)
        self.assertEqual([source.label for source in result.web_sources], ["W1"])

    def test_explicit_web_request_forces_web_stage(self) -> None:
        rag = self._make_rag(local_answer="Local answer [S1]", web_answer="Forced web [W1]")
        result = rag.ask("Search the web for this person")

        self.assertTrue(result.used_web)
        self.assertIn("[W1]", result.answer)

    def test_explicit_web_request_uses_clean_provider_query(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Web answer [W1]",
        )

        result = rag.ask("Search the web about Luxembourg")

        self.assertTrue(result.used_web)
        self.assertEqual(rag.web_search.queries[0], "Luxembourg")

    def test_web_search_fans_out_when_first_query_is_empty(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Luc Frieden is the prime minister of Luxembourg [W1]",
            web_sources=[],
        )
        rag.web_search = SequentialWebSearch(
            [
                [],
                [
                    WebSource(
                        label="W1",
                        title="Prime Minister of Luxembourg",
                        url="https://example.com/pm",
                        content="Luc Frieden is Prime Minister of Luxembourg.",
                    )
                ],
            ]
        )

        result = rag.ask("Who is the current prime minister of Luxembourg?")

        self.assertTrue(result.used_web)
        self.assertIn("Luc Frieden", result.answer)
        self.assertGreaterEqual(len(rag.web_search.queries), 2)
        self.assertTrue(any("official government" in query for query in rag.web_search.queries))

    def test_aggressive_web_fallback_expands_thin_queries(self) -> None:
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, model_answer=MODEL_UNKNOWN, web_sources=[])
        rag.web_search = QueryAwareWebSearch(
            lambda query: [
                WebSource(
                    label="W1",
                    title="Reliable source",
                    url="https://example.com/reliable",
                    content="Obscure topic reliable evidence.",
                    score=0.8,
                )
            ]
            if "reliable source" in query
            else []
        )

        sources = rag._search_web_sources(["obscure topic"], question="obscure topic")

        self.assertEqual(len(sources), 1)
        self.assertTrue(any("reliable source" in query for query in rag.web_search.queries))

    def test_size_query_uses_normalized_web_fallback_variants(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Africa has a total area of about 30.37 million square kilometres [W1].",
            local_sources=[],
            web_sources=[],
        )
        rag.web_search = QueryAwareWebSearch(
            lambda query: [
                WebSource(
                    label="W1",
                    title="Africa area reference",
                    url="https://example.org/africa-area",
                    content="Africa has a total area of about 30.37 million square kilometres.",
                    score=0.82,
                )
            ]
            if "area africa" in query.lower() or "africa area" in query.lower()
            else []
        )

        result = rag.ask("Size of Africa")

        self.assertTrue(result.used_web)
        self.assertIn("30.37 million", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertTrue(any("area africa" in query.lower() for query in rag.web_search.queries))

    def test_smallest_country_in_europe_skips_irrelevant_local_files(self) -> None:
        rag = self._make_rag(
            local_answer="Bayesian local paper should not be used [S1].",
            model_answer=MODEL_UNKNOWN,
            web_answer="Vatican City is the smallest country in Europe [W1].",
            web_sources=[
                WebSource(
                    label="W1",
                    title="Vatican City smallest country",
                    url="https://example.com/vatican",
                    content="Vatican City is the smallest country in Europe by area.",
                )
            ],
        )

        result = rag.ask("The smallest country in Europe")

        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertEqual(result.diagnostics["search_plan_intent"], "public_knowledge")
        self.assertIn("Vatican City", result.answer)
        self.assertNotIn("Bayesian", result.answer)
        self.assertTrue(result.used_web)

    def test_president_of_smallest_country_uses_head_of_state_search(self) -> None:
        rag = self._make_rag(
            local_answer="Irrelevant local answer [S1].",
            model_answer=MODEL_UNKNOWN,
            web_answer=(
                "Vatican City does not have a president; its head of state is the Pope [W1]."
            ),
            web_sources=[
                WebSource(
                    label="W1",
                    title="Vatican City head of state",
                    url="https://example.com/vatican-head",
                    content=(
                        "Vatican City does not have a president. "
                        "The Pope is head of state of Vatican City."
                    ),
                )
            ],
        )

        result = rag.ask("The president of the smallest country in Europe")

        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertEqual(result.diagnostics["search_plan_intent"], "public_knowledge")
        self.assertIn("does not have a president", result.answer)
        self.assertTrue(any("head of state" in query for query in result.diagnostics["web_queries"]))

    def test_answer_verification_reports_source_support(self) -> None:
        source = WebSource(
            label="W1",
            title="Econometrics source",
            url="https://example.com/econometrics",
            content="Econometrics applies statistical methods to economic data.",
            score=0.9,
        )

        verified = _verify_answer_against_evidence(
            "Econometrics applies statistical methods to economic data [W1].",
            [],
            [source],
            "What is econometrics?",
            AppSettings(),
        )
        unsupported = _verify_answer_against_evidence(
            "Econometrics is a type of medieval poetry [W1].",
            [],
            [source],
            "What is econometrics?",
            AppSettings(),
        )

        self.assertEqual(verified["status"], "verified")
        self.assertEqual(unsupported["status"], "unsupported")

    def test_current_secretary_of_state_skips_polluting_local_state_chunks(self) -> None:
        state_column_source = LocalSource(
            label="S1",
            document="state-column.csv",
            page=None,
            chunk_id="state-column",
            text="State = NY. State column values include NY, CA, and TX.",
            score=0.99,
        )
        rag = self._make_rag(
            local_answer="The local file says State = NY [S1].",
            model_answer="Older model answer.",
            local_sources=[state_column_source],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Mississippi Secretary of State Michael Watson",
                    url="https://www.sos.ms.gov/home",
                    content="Michael Watson is Mississippi's Secretary of State.",
                    score=1.0,
                ),
                WebSource(
                    label="W2",
                    title="Secretary of State Marco Rubio - U.S. Department of State",
                    url="https://www.state.gov/secretary-of-state/",
                    content="Marco Rubio was sworn in as the 72nd Secretary of State.",
                    score=0.2,
                )
            ],
        )

        result = rag.ask("Who is the secretary of state?")

        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertEqual(result.local_sources, [])
        self.assertTrue(result.used_web)
        self.assertIn("Marco Rubio", result.answer)
        self.assertNotIn("Michael Watson", result.answer)
        self.assertNotIn("State = NY", result.answer)

    def test_weak_primary_web_results_are_augmented_with_duckduckgo_fallback(self) -> None:
        noisy_sources = [
            WebSource(
                label=f"W{index}",
                title=f"Social source {index}",
                url=f"https://www.facebook.com/example-{index}",
                content="Loose mention of Luxembourg.",
            )
            for index in range(1, 6)
        ]
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Luc Frieden is the prime minister of Luxembourg [W1]",
            web_sources=noisy_sources,
        )
        rag._search_duckduckgo_fallback = lambda query: [
            WebSource(
                label="W1",
                title="FRIEDEN Luc - The Luxembourg Government",
                url="https://gouvernement.lu/en/gouvernement/luc-frieden.html",
                content="Luc Frieden is Prime Minister of Luxembourg.",
            )
        ]

        result = rag.ask("who is the current prime minister of Luxembourg")

        self.assertEqual(
            result.web_sources[0].url,
            "https://gouvernement.lu/en/gouvernement/luc-frieden.html",
        )
        self.assertIn("[W1]", result.answer)

    def test_empty_primary_web_results_are_augmented_with_duckduckgo_fallback(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Alex Sample profile [W1]",
            local_sources=[],
            web_sources=[],
        )
        rag._search_duckduckgo_fallback_queries = lambda queries: [
            WebSource(
                label="W1",
                title="Alex Sample profile",
                url="https://example.com/alex-sample",
                content="Alex Sample is listed on this profile.",
            )
        ]

        result = rag.ask("Alex Sample")

        self.assertEqual([source.label for source in result.web_sources], ["W1"])
        self.assertTrue(result.used_web)
        self.assertIn("[W1]", result.answer)

    def test_lowercase_name_statement_uses_web_and_tolerates_close_spelling(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Sophia Loizidou is a researcher at the University of Luxembourg [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Sophia Loizidou - University of Luxembourg",
                    url="https://www.uni.lu/example/sophia-loizidou",
                    content="Sophia Loizidou is a doctoral researcher at the University of Luxembourg.",
                )
            ],
        )

        result = rag.ask("sofia loizidou")

        self.assertTrue(result.used_web)
        self.assertTrue(result.diagnostics["requires_web_validation"])
        self.assertEqual([source.label for source in result.web_sources], ["W1"])
        self.assertIn("Sophia Loizidou", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 1)

    def test_identity_lookup_rejects_local_certificate_name_match(self) -> None:
        local_certificate = LocalSource(
            label="S1",
            document="exam_payment_certificate_Sproochentest_2025.pdf",
            page=1,
            chunk_id="certificate",
            text=(
                "Luxembourg ATTESTATION ALEX JORDAN SAMPLE paid for the "
                "Sproochentest language exam certificate."
            ),
            score=0.95,
        )
        rag = self._make_rag(
            local_answer="Alex Jordan Sample appears in a certificate [S1].",
            model_answer=MODEL_UNKNOWN,
            local_sources=[local_certificate],
            web_sources=[
                WebSource(
                    label="W1",
                    title="ORBilu: Profile of ALEX JORDAN SAMPLE",
                    url="https://orbilu.uni.lu/profile?uid=50039094",
                    content="Profile of ALEX JORDAN SAMPLE at the University of Luxembourg.",
                )
            ],
        )

        result = rag.ask("Alex Jordan Sample")

        self.assertEqual(result.local_sources, [])
        self.assertTrue(result.used_web)
        self.assertIn("ORBilu", result.answer)
        self.assertNotIn("certificate", result.answer.lower())

    def test_identity_lookup_rejects_incidental_local_mentions(self) -> None:
        incidental_local = LocalSource(
            label="S1",
            document="econometrics_course_notes.pdf",
            page=12,
            chunk_id="incidental-person-mention",
            text=(
                "Research methods course notes. Acknowledgements mention Florian Felice "
                "as someone who attended a seminar, but this page is not a profile or CV."
            ),
            score=0.96,
        )
        rag = self._make_rag(
            local_answer="Florian Felice is mentioned in course notes [S1].",
            model_answer=MODEL_UNKNOWN,
            local_sources=[incidental_local],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Florian Felice - University of Luxembourg",
                    url="https://www.uni.lu/example/florian-felice",
                    content=(
                        "Florian Felice is a doctoral researcher at the University of Luxembourg."
                    ),
                )
            ],
        )

        result = rag.ask("Florian Felice")

        self.assertEqual(result.local_sources, [])
        self.assertTrue(result.used_web)
        self.assertEqual([source.label for source in result.web_sources], ["W1"])
        self.assertIn("Florian Felice", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertNotIn("[S1]", result.answer)
        self.assertNotIn("course notes", result.answer.lower())

    def test_reversed_name_identity_lookup_can_fall_back_to_strong_local_profile_evidence(self) -> None:
        local_profile = LocalSource(
            label="S1",
            document="alex_jordan_cv.pdf",
            page=1,
            chunk_id="cv-profile",
            text=(
                "Alex Jordan Sample. University of Luxembourg doctoral researcher in "
                "computational statistics. Contact details and profile summary."
            ),
            score=0.97,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[local_profile],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Alex Jordan profile",
                    url="https://example.com/alex-jordan",
                    content="Alex Jordan has a university profile.",
                )
            ],
        )

        result = rag.ask("Jordan Alex")

        self.assertFalse(result.used_web)
        self.assertEqual([source.label for source in result.local_sources], ["S1"])
        self.assertIn("Alex Jordan", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 1)
        self.assertEqual(result.diagnostics["answer_verification_status"], "verified")

    def test_identity_lookup_filters_same_name_web_outlier_by_context_cluster(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Christophe LEY - FSTM",
                    url="https://www.uni.lu/fstm-en/people/christophe-ley",
                    content=(
                        "Christophe Ley is Associate Professor in Mathematics at the University "
                        "of Luxembourg with research in statistics."
                    ),
                ),
                WebSource(
                    label="W2",
                    title="Christophe Ley – The Conversation",
                    url="https://theconversation.com/profiles/christophe-ley-2436551",
                    content=(
                        "Christophe Ley is Associate Professor in Mathematics at the University "
                        "of Luxembourg and writes about statistics."
                    ),
                ),
                WebSource(
                    label="W3",
                    title="Christophe Ley",
                    url="https://en.wikipedia.org/wiki/Christophe_Ley",
                    content="Christophe Ley is a Luxembourgish statistician and author.",
                ),
                WebSource(
                    label="W4",
                    title="Christophe Ley | Department of Recreation, Parks & Tourism",
                    url="https://rpt.sfsu.edu/christophe-ley",
                    content=(
                        "With more than 20 years of experience, Ley has worked with the San "
                        "Francisco CVB and tourism conferences."
                    ),
                ),
            ],
        )

        result = rag.ask("Christophe ley")

        self.assertTrue(result.used_web)
        self.assertIn("Luxembourgish statistician", result.answer)
        self.assertNotIn("Recreation, Parks & Tourism", result.answer)
        self.assertFalse(
            any("Recreation, Parks & Tourism" in source.title for source in result.web_sources)
        )

    def test_person_lookup_discards_local_chunk_about_different_named_person(self) -> None:
        contaminated = LocalSource(
            label="S1",
            document="people.pdf",
            page=1,
            chunk_id="gabriella-profile",
            text=(
                "Gabriella Vinco is a Doctoral Researcher at the University of Luxembourg. "
                "The same department also mentions Christophe Ley in an acknowledgements list."
            ),
            score=0.98,
        )
        correct = WebSource(
            label="W1",
            title="Christophe Ley - University of Luxembourg",
            url="https://www.uni.lu/fstm-en/people/christophe-ley",
            content="Christophe Ley is Associate Professor in Mathematics at the University of Luxembourg.",
        )
        rag = self._make_rag(
            local_answer="Gabriella Vinco is a Doctoral Researcher [S1].",
            model_answer=MODEL_UNKNOWN,
            local_sources=[contaminated],
            web_sources=[correct],
        )

        result = rag.ask("Christophe Ley")

        self.assertEqual(result.local_sources, [])
        self.assertTrue(result.used_web)
        self.assertEqual([source.label for source in result.web_sources], ["W1"])
        self.assertIn("Christophe Ley", result.answer)
        self.assertNotIn("Gabriella", result.answer)

    def test_person_lookup_discards_regression_document_even_when_name_is_incidental(self) -> None:
        regression = LocalSource(
            label="S1",
            document="E1_MultipleRegressionModel.pdf",
            page=8,
            chunk_id="regression-8",
            text=(
                "Multiple regression exercise. PRICE = beta0 + beta1M2 + beta2AGE + beta3DIS. "
                "Example prepared by Gabriella Vinco for class discussion."
            ),
            score=0.99,
        )
        profile = WebSource(
            label="W1",
            title="Gabriella Vinco - University profile",
            url="https://www.uni.lu/example/gabriella-vinco",
            content="Gabriella Vinco is a researcher at the University of Luxembourg.",
        )
        rag = self._make_rag(
            local_answer="The regression document says PRICE = beta0 + beta1M2 + beta2AGE + beta3DIS [S1].",
            model_answer=MODEL_UNKNOWN,
            local_sources=[regression],
            web_sources=[profile],
        )

        result = rag.ask("Gabriella Vinco")

        self.assertEqual(result.local_sources, [])
        self.assertTrue(result.used_web)
        self.assertIn("Gabriella Vinco", result.answer)
        self.assertNotIn("PRICE", result.answer)
        self.assertNotIn("regression", result.answer.lower())

    def test_entity_lookup_uses_fast_extractive_synthesis(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Sophia Loizidou - University of Luxembourg",
                    url="https://www.uni.lu/fstm-en/people/sophia-loizidou/",
                    content=(
                        "Sophia Loizidou is a doctoral researcher in statistics "
                        "at the University of Luxembourg."
                    ),
                )
            ],
        )
        rag.generator = StructuredFinalGenerator()

        result = rag.ask("sofia loizidou")

        self.assertIn("Confidence:", result.answer)
        self.assertIn("Sophia Loizidou", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertFalse(result.answer.startswith("Confidence:"))
        self.assertEqual(rag.generator.chat_messages, [])
        self.assertEqual(rag.generator.final_calls, [])

    def test_scientific_lookup_uses_expanded_local_queries(self) -> None:
        thesis_source = LocalSource(
            label="S1",
            document="alex-thesis.pdf",
            page=42,
            chunk_id="thesis-hremc",
            text=(
                "Replica Exchange Hamiltonian Monte Carlo combines Hamiltonian Monte Carlo "
                "with replica exchange for Bayesian parameter inference."
            ),
            score=0.76,
            metadata={"document_type": "thesis"},
        )
        rag = self._make_rag(
            local_answer="Replica Exchange Hamiltonian Monte Carlo combines HMC with replica exchange [S1].",
            local_sources=[],
            web_sources=[],
        )
        rag.retriever = QueryAwareRetriever(
            lambda query: [thesis_source] if "Monte Carlo" in query and not query.startswith("What is") else []
        )

        result = rag.ask("What is Replica Exchange Hamiltonian Montecarlo?")

        self.assertTrue(
            any("Replica Exchange Hamiltonian Monte Carlo" in call[0][0] for call in rag.retriever.calls)
        )
        self.assertEqual([source.document for source in result.local_sources], ["alex-thesis.pdf"])
        self.assertFalse(result.used_web)
        self.assertIn("[S1]", result.answer)

    def test_scientific_web_answer_is_structured_from_evidence(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Replica Exchange Hamiltonian Monte Carlo paper",
                    url="https://example.edu/hremc",
                    content=(
                        "Replica Exchange Hamiltonian Monte Carlo combines Hamiltonian "
                        "Monte Carlo with replica exchange sampling. It improves exploration "
                        "of multimodal posterior distributions and supports marginal likelihood "
                        "estimation. The method is applied to Bayesian hydrological models."
                    ),
                    score=0.9,
                )
            ],
        )
        rag.generator = StructuredFinalGenerator()

        result = rag.ask("What is Replica Exchange Hamiltonian Monte Carlo?")

        self.assertIn("Confidence:", result.answer)
        self.assertIn("Key points:", result.answer)
        self.assertIn("Sources:", result.answer)
        self.assertIn("combines Hamiltonian Monte Carlo", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertFalse(result.answer.startswith("Confidence:"))
        self.assertEqual(rag.generator.chat_messages, [])

    def test_public_entity_lookup_prefers_clear_local_answer(self) -> None:
        local_source = LocalSource(
            label="S1",
            document="ley.pdf",
            page=1,
            chunk_id="chunk-ley",
            text="Christophe Ley local profile.",
            score=0.91,
        )
        rag = self._make_rag(
            local_answer="Local profile [S1]",
            model_answer="Model background",
            web_answer="Local profile [S1] with current web validation [W1]",
            local_sources=[local_source],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Christophe Ley profile",
                    url="https://example.com/ley",
                    content="Christophe Ley current profile.",
                )
            ],
        )

        result = rag.ask("Christophe Ley")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertEqual([source.label for source in result.local_sources], ["S1"])
        self.assertEqual(result.web_sources, [])

    def test_current_question_uses_current_information_label_from_web_evidence(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Older model answer says someone else.",
            web_answer="Luc Frieden is the prime minister of Luxembourg [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="FRIEDEN Luc - The Luxembourg Government",
                    url="https://gouvernement.lu/en/gouvernement/luc-frieden.html",
                    content="Luc Frieden is Prime Minister of Luxembourg.",
                )
            ],
        )

        result = rag.ask("Who is the current prime minister of Luxembourg?")

        self.assertEqual(result.confidence, "current-information")
        self.assertTrue(result.used_web)
        self.assertIn("Luc Frieden", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertFalse(result.diagnostics["parallel_model_with_web"])
        self.assertTrue(result.diagnostics["model_skipped_for_current_web"])
        self.assertFalse(result.diagnostics["used_model_knowledge"])

    def test_generic_country_president_query_continues_to_web_after_local_miss(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="President Bola Ahmed Tinubu",
                    url="https://statehouse.gov.ng/president-bola-ahmed-tinubu/",
                    content="Bola Ahmed Tinubu is the President of Nigeria.",
                )
            ],
        )

        result = rag.ask("Who is the president of Nigeria?")

        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertEqual(result.diagnostics["search_plan_intent"], "government")
        self.assertEqual(result.diagnostics["search_plan"]["country"], "Nigeria")
        self.assertTrue(result.used_web)
        self.assertEqual(result.confidence, "current-information")
        self.assertIn("Bola Ahmed Tinubu", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertNotIn("could not verify current information", result.answer.lower())

    def test_generic_country_president_followup_uses_remembered_holder(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Bola Ahmed Tinubu biography",
                    url="https://statehouse.gov.ng/president-bola-ahmed-tinubu/",
                    content=(
                        "Bola Ahmed Tinubu became President of Nigeria on 29 May 2023."
                    ),
                )
            ],
        )
        history = [
            ChatMessage(role="user", content="Who is the president of Nigeria?"),
            ChatMessage(
                role="assistant",
                content="The current president of Nigeria is Bola Ahmed Tinubu [W1].",
            ),
        ]

        result = rag.ask("When did he become president?", history=history)

        self.assertTrue(result.diagnostics["conversation_followup"])
        self.assertEqual(
            result.diagnostics["resolved_query"],
            "When did Bola Ahmed Tinubu become President of Nigeria?",
        )
        self.assertTrue(result.used_web)
        self.assertIn("29 May 2023", result.answer)
        self.assertIn("[W1]", result.answer)

    def test_current_country_president_strips_honorific_prefixes_from_web_evidence(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="National Government | Kenya Embassy Ankara Türkiye",
                    url="https://kenyaembassy.org.tr/pages/15/National-Government",
                    content=(
                        "THE GOVERNMENT OF KENYA THE THREE ARMS OF GOVERNMENT. "
                        "Office of the President HIS EXCELLENCY HON. WILLIAM SAMOEI RUTO, "
                        "PhD., C.G.H. President of the Republic of Kenya and Commander-In-Chief "
                        "of the Defence Forces."
                    ),
                ),
                WebSource(
                    label="W2",
                    title="President of Kenya - Wikipedia",
                    url="https://en.wikipedia.org/wiki/President_of_Kenya",
                    content="William Samoei Ruto is the president of Kenya.",
                ),
            ],
        )

        result = rag.ask("Who is the current president of Kenya?")

        self.assertEqual(result.confidence, "current-information")
        self.assertTrue(result.used_web)
        self.assertIn("William Samoei Ruto", result.answer)
        self.assertNotIn("His Excellency [W1]", result.answer)
        self.assertNotIn("His Excellency", result.answer)

    def test_prime_minister_query_recognizes_norway_country(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="The prime minister of the United States is not relevant [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Prime Minister - Government.no",
                    url="https://www.regjeringen.no/en/the-government/prime-minister/",
                    content="Jonas Gahr Støre is Prime Minister of Norway.",
                )
            ],
        )

        result = rag.ask("Who is the prime minister of Norway?")

        self.assertEqual(result.diagnostics["conversation_country"], "Norway")
        self.assertEqual(result.diagnostics["search_plan_intent"], "government")
        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertIn("Norway", result.diagnostics["web_queries"][0])
        self.assertNotIn("United States", result.answer)
        self.assertEqual(result.answer, "The current prime minister of Norway is Jonas Gahr Støre [W1].")

    def test_usa_current_president_uses_official_web_evidence_over_stale_model(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="The current president of the United States is Joe Biden.",
            web_answer="The current president of the United States is Joe Biden [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="President Donald J. Trump - The White House",
                    url="https://www.whitehouse.gov/administration/donald-j-trump/",
                    content=(
                        "President Donald J. Trump is the 45th and 47th President of "
                        "the United States."
                    ),
                ),
                WebSource(
                    label="W2",
                    title="President of the United States",
                    url="https://www.usa.gov/presidents",
                    content="The president of the United States is the head of state.",
                ),
            ],
        )

        result = rag.ask("Who is the president of the USA?")

        self.assertEqual(result.confidence, "current-information")
        self.assertTrue(result.used_web)
        self.assertIn("Donald J. Trump", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertNotIn("Joe Biden", result.answer)
        self.assertTrue(result.diagnostics["current_role_override"])

    def test_current_public_fact_evidence_rejects_generic_directory_and_trims_headline_suffix(self) -> None:
        web_sources = [
            WebSource(
                label="W1",
                title="Fact Sheets - The White House",
                url="https://www.whitehouse.gov/fact-sheets/",
                content="President Donald J. Trump Updates Tariffs after the latest review.",
            ),
            WebSource(
                label="W2",
                title="Find and contact elected officials - USAGov",
                url="https://www.usa.gov/elected-officials",
                content="Find the names and current activities of state and territorial legislators.",
            ),
            WebSource(
                label="W3",
                title="Donald Trump | Breaking News & Latest Updates | AP News",
                url="https://apnews.com/hub/donald-trump",
                content="Stay informed and read the latest breaking news and updates on Donald Trump.",
            ),
        ]

        evidence = _current_public_fact_evidence("Who is the current president of the USA?", web_sources)

        self.assertEqual(
            [(source.label, candidate) for source, candidate in evidence],
            [("W1", "Donald J. Trump")],
        )

    def test_dedupe_web_queries_extracts_query_from_stringified_dict(self) -> None:
        queries = [
            "{'query': 'current president of the USA', 'source': 'local', 'local_search': True}",
            "{'query': 'current president of the USA', 'source': 'model knowledge'}",
            "current president of the USA official government",
        ]

        deduped = _dedupe_web_queries(queries)

        self.assertEqual(
            deduped,
            [
                "current president of the USA",
                "current president of the USA official government",
            ],
        )

    def test_usa_king_question_corrects_role_and_uses_president_evidence(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="The current king of the USA is King Charles III.",
            web_answer="The current king of the USA is King Charles III [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="King Charles III arrives in US for state visit",
                    url="https://example-news.test/king-charles-us-visit",
                    content="King Charles III visited the United States for a state visit.",
                    score=10.0,
                ),
                WebSource(
                    label="W2",
                    title="President Donald J. Trump - The White House",
                    url="https://www.whitehouse.gov/administration/donald-j-trump/",
                    content=(
                        "President Donald J. Trump is the 45th and 47th President of "
                        "the United States."
                    ),
                    score=0.1,
                ),
            ],
        )

        result = rag.ask("Who is the king of the USA?")

        self.assertEqual(result.confidence, "current-information")
        self.assertTrue(result.used_web)
        self.assertIn("does not have a king or queen", result.answer)
        self.assertIn("Donald J. Trump", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertNotIn("King Charles", result.answer)
        self.assertEqual(
            result.web_sources[0].url,
            "https://www.whitehouse.gov/administration/donald-j-trump/",
        )
        self.assertTrue(result.diagnostics["current_role_override"])
        self.assertIn("current president of the United States", rag.web_search.queries[0])

    def test_current_uk_prime_minister_discards_outdated_rishi_source(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="The prime minister of the UK is Rishi Sunak.",
            web_answer="The current prime minister of the UK is Rishi Sunak [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="The Rt Hon Rishi Sunak MP - GOV.UK",
                    url="https://www.gov.uk/government/people/rishi-sunak",
                    content="Rishi Sunak was Prime Minister between 25 October 2022 and 5 July 2024.",
                    score=20.0,
                ),
                WebSource(
                    label="W2",
                    title="Prime Minister - GOV.UK",
                    url="https://www.gov.uk/government/ministers/prime-minister",
                    content=(
                        "Current role holder The Rt Hon Sir Keir Starmer KCB KC MP. "
                        "Sir Keir Starmer became Prime Minister on 5 July 2024."
                    ),
                    score=0.1,
                ),
                WebSource(
                    label="W3",
                    title="Prime Minister of the United Kingdom - Wikipedia",
                    url="https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom",
                    content=(
                        "Incumbent Keir Starmer since 5 July 2024. The prime minister "
                        "of the United Kingdom is the head of government."
                    ),
                    score=0.1,
                ),
            ],
        )

        result = rag.ask("The prime minister of UK")

        self.assertEqual(result.confidence, "current-information")
        self.assertTrue(result.used_web)
        self.assertIn("Keir Starmer", result.answer)
        self.assertNotIn("Rishi Sunak", result.answer)
        self.assertNotIn("Evidence conflict detected", result.answer)
        self.assertTrue(result.diagnostics["current_role_override"])
        self.assertFalse(result.diagnostics["evidence_conflict"])
        self.assertEqual(
            result.web_sources[0].url,
            "https://www.gov.uk/government/ministers/prime-minister",
        )
        self.assertIn("GOV.UK Prime Minister", rag.web_search.queries[0])

    def test_current_netherlands_king_uses_royal_house_evidence(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="The best web evidence I found points to these sources.",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="King Willem-Alexander | Royal House of the Netherlands",
                    url="https://www.royal-house.nl/members-royal-house/king-willem-alexander",
                    content=(
                        "King Willem-Alexander has been King of the Netherlands "
                        "since 30 April 2013."
                    ),
                ),
                WebSource(
                    label="W2",
                    title="Queen Maxima of the Netherlands",
                    url="https://example.com/queen-maxima",
                    content="Queen Maxima is queen consort of the Netherlands.",
                ),
            ],
        )

        result = rag.ask("Who is the king of Netherlands?")

        self.assertEqual(result.confidence, "current-information")
        self.assertTrue(result.used_web)
        self.assertIn("Willem-Alexander", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertNotIn("Maxima", result.answer)
        self.assertTrue(result.diagnostics["current_role_override"])

    def test_office_holder_extraction_trims_noisy_headline_before_followup(self) -> None:
        web_sources = [
            WebSource(
                label="W1",
                title="President Paul BIYA Receives US AFRICOM Deputy Commander",
                url="https://example.cm/presidency",
                content=(
                    "President Paul BIYA Receives US AFRICOM Deputy Commander at Unity Palace. "
                    "Paul Biya has served as President of Cameroon since 6 November 1982."
                ),
            )
        ]
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Paul Biya evidence [W1].",
            local_sources=[],
            web_sources=web_sources,
        )

        first = rag.ask("Who is the president of Cameroon?")

        self.assertIn("Paul Biya", first.answer)
        self.assertNotIn("Receives", first.answer)
        self.assertNotIn("AFRICOM", first.answer)

        history = [
            ChatMessage(role="user", content="Who is the president of Cameroon?"),
            ChatMessage(role="assistant", content=first.answer),
        ]
        second = rag.ask("When did he become president?", history=history)

        self.assertIn(
            "When did Paul Biya become President of Cameroon?",
            second.diagnostics["resolved_query"],
        )
        self.assertNotIn("Receives", second.diagnostics["resolved_query"])
        self.assertNotIn("AFRICOM", second.diagnostics["resolved_query"])
        self.assertIn("6 November 1982", second.answer)
        self.assertIn("[W1]", second.answer)

    def test_current_role_validation_requires_official_or_independent_agreement(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="A random blog says Alex Example is prime minister [W1].",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Unofficial politics blog",
                    url="https://personal-blog.example/current-pm",
                    content="Alex Example is the current prime minister of the United Kingdom.",
                )
            ],
        )

        result = rag.ask("Who is the current prime minister of the UK?")

        self.assertNotIn("Alex Example is the current prime minister", result.answer)
        self.assertNotIn("current_role_override", result.diagnostics)

    def test_current_question_does_not_fall_back_to_ai_when_web_fails(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Older model answer says someone else.",
        )
        rag.web_search = BrokenWebSearch()

        result = rag.ask("Who is the current prime minister of Luxembourg?")

        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.used_web)
        self.assertNotIn("Older model answer", result.answer)
        self.assertIn("AI knowledge is not reliable enough for current facts", result.answer)

    def test_web_disabled_blocks_current_public_fact_from_ai_final_truth(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="The current prime minister is an outdated model answer.",
            local_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("Who is the current prime minister of Luxembourg?")

        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.used_web)
        self.assertIn("AI knowledge is not reliable enough for current facts", result.answer)

    def test_current_public_fact_skips_stale_local_file_evidence(self) -> None:
        local_source = LocalSource(
            label="S1",
            document="old-prime-minister.pdf",
            page=1,
            chunk_id="old-pm",
            text="The 2024 local file says the prime minister is Example Old.",
            score=0.91,
            metadata={"document_date": "2024-01-01"},
        )
        rag = self._make_rag(
            local_answer="The local file says Example Old [S1].",
            model_answer=MODEL_UNKNOWN,
            web_answer="Luc Frieden is the prime minister of Luxembourg [W1].",
            local_sources=[local_source],
            web_sources=[
                WebSource(
                    label="W1",
                    title="FRIEDEN Luc - The Luxembourg Government",
                    url="https://gouvernement.lu/en/gouvernement/luc-frieden.html",
                    content="Luc Frieden is the Prime Minister of Luxembourg.",
                    published_date="2026-06-01",
                )
            ],
        )

        result = rag.ask("Who is the current prime minister of Luxembourg?")

        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertEqual(result.local_sources, [])
        self.assertIn("Luc Frieden", result.answer)
        self.assertNotIn("Example Old", result.answer)

    def test_public_definition_query_skips_unrelated_local_sources(self) -> None:
        sas_source = LocalSource(
            label="S1",
            document="sas-certification-prep-guide.pdf",
            page=839,
            chunk_id="sas-839",
            text="%INCLUDE statement macro parameters and SAS programming directives.",
            score=0.95,
            metadata={"query_overlap": 2, "fast_rerank_score": 0.91},
        )
        rag = self._make_rag(
            local_answer="SAS answer [S1]",
            model_answer="Toxicology is the study of adverse effects of chemicals.",
            web_answer="Toxicology studies harmful effects of chemicals. REACH is an EU chemicals regulation [W1].",
            local_sources=[sas_source],
            web_sources=[
                WebSource(
                    label="W1",
                    title="REACH Regulation - ECHA",
                    url="https://echa.europa.eu/regulations/reach/legislation",
                    content=(
                        "REACH is a European Union regulation concerning registration, "
                        "evaluation, authorisation and restriction of chemicals."
                    ),
                ),
                WebSource(
                    label="W2",
                    title="Toxicology definition",
                    url="https://www.niehs.nih.gov/health/topics/science/toxicology",
                    content="Toxicology is a field of science that studies harmful effects of substances.",
                ),
            ],
        )

        result = rag.ask("What is toxicology? What is EU REACH directive")

        self.assertFalse(result.diagnostics["local_retrieval_skipped"])
        self.assertEqual(result.local_sources, [])
        self.assertGreater(len(rag.retriever.calls), 0)
        self.assertGreater(len(rag.generator.local_calls), 0)
        self.assertTrue(result.used_web)
        self.assertIn("Toxicology", result.answer)
        self.assertIn("REACH", result.answer)
        self.assertNotIn("SAS", result.answer)

    def test_identity_lookup_filters_unrelated_local_and_web_sources(self) -> None:
        unrelated_local = LocalSource(
            label="S1",
            document="horizon.pdf",
            page=1,
            chunk_id="chunk-horizon",
            text="Horizon 2020 evaluator and research activity details for another person.",
            score=0.93,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="This should not be used [W1]",
            local_sources=[unrelated_local],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Dylan Jordan",
                    url="https://example.com/dylan",
                    content="Dylan Jordan is a different person.",
                )
            ],
        )

        result = rag.ask("Alex Jordan Sample")

        self.assertEqual(result.local_sources, [])
        self.assertEqual(result.web_sources, [])
        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "low")

    def test_mixed_case_identity_lookup_filters_unrelated_local_sources(self) -> None:
        unrelated_local = LocalSource(
            label="S1",
            document="course_overview.pdf",
            page=4,
            chunk_id="chunk-course",
            text="This course overview discusses econometric modelling and environmental economics.",
            score=0.93,
        )
        rag = self._make_rag(
            local_answer="This unrelated local answer should not be used [S1]",
            model_answer=MODEL_UNKNOWN,
            local_sources=[unrelated_local],
            web_sources=[],
        )

        result = rag.ask("Christophe ley")

        self.assertEqual(result.local_sources, [])
        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "low")
        self.assertNotIn("unrelated local answer", result.answer)

    def test_local_file_question_filters_irrelevant_primary_hits_after_expansion(self) -> None:
        irrelevant = LocalSource(
            label="S1",
            document="sas_certification.pdf",
            page=10,
            chunk_id="sas-cert",
            text="SAS certification preparation content about formats and libraries.",
            score=0.56,
        )
        relevant = LocalSource(
            label="S2",
            document="language_certificate.pdf",
            page=1,
            chunk_id="language-cert",
            text="Alex Sample Sproochentest language certificate exam result.",
            score=0.64,
        )
        rag = self._make_rag(local_answer="This should not be called.", local_sources=[])
        rag.retriever = SequentialLocalRetriever([[irrelevant], [relevant]])

        result = rag.ask("Which document contains my language certificate?")

        self.assertEqual([source.document for source in result.local_sources], ["language_certificate.pdf"])
        self.assertIn("language_certificate.pdf", result.answer)
        self.assertNotIn("sas_certification.pdf", result.answer)

    def test_local_corpus_summary_browses_each_indexed_document(self) -> None:
        passport = LocalSource(
            label="S1",
            document="DamianMingopassport .pdf",
            page=1,
            chunk_id="passport-1",
            text="Passport details with name, nationality, date of birth and place of issue.",
            score=1.0,
        )
        diploma = LocalSource(
            label="S2",
            document="Damian_diploma.pdf",
            page=1,
            chunk_id="diploma-1",
            text="This letter confirms that Damian Mingo Ndiwago defended his doctoral thesis on 26 September 2024.",
            score=1.0,
        )
        regression = LocalSource(
            label="S3",
            document="E1_MultipleRegressionModel.pdf",
            page=1,
            chunk_id="regression-1",
            text="Applied Time Series Analysis and Forecasting exercise about a multiple regression model using Stata.",
            score=1.0,
        )
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, local_sources=[])
        rag.retriever = CorpusBrowsingRetriever([], [passport, diploma, regression])

        result = rag.ask("summarise the documents")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertEqual(
            [source.document for source in result.local_sources],
            ["DamianMingopassport .pdf", "Damian_diploma.pdf", "E1_MultipleRegressionModel.pdf"],
        )
        self.assertIn("DamianMingopassport .pdf", result.answer)
        self.assertIn("Damian_diploma.pdf", result.answer)
        self.assertIn("E1_MultipleRegressionModel.pdf", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertIn("[S2]", result.answer)
        self.assertIn("[S3]", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertTrue(rag.retriever.corpus_calls)

    def test_content_of_files_browses_local_corpus(self) -> None:
        local_source = LocalSource(
            label="S1",
            document="course_notes.pdf",
            page=3,
            chunk_id="course-3",
            text="These course notes explain regression coefficients and fitted values.",
            score=1.0,
        )
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, local_sources=[])
        rag.retriever = CorpusBrowsingRetriever([], [local_source])

        result = rag.ask("what is the content of the files")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertEqual([source.document for source in result.local_sources], ["course_notes.pdf"])
        self.assertIn("course_notes.pdf", result.answer)
        self.assertIn("[S1]", result.answer)

    def test_docs_in_database_summary_browses_local_corpus(self) -> None:
        local_source = LocalSource(
            label="S1",
            document="database_notes.pdf",
            page=1,
            chunk_id="database-notes-1",
            text="These indexed notes summarize local database content about regression models.",
            score=1.0,
        )
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, local_sources=[])
        rag.retriever = CorpusBrowsingRetriever([], [local_source])

        result = rag.ask("summarise the docs in the database")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertEqual([source.document for source in result.local_sources], ["database_notes.pdf"])
        self.assertIn("database_notes.pdf", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 0)
        self.assertEqual(result.diagnostics["evidence_policy"], "local_only")

    def test_spaced_data_base_inventory_uses_local_stats(self) -> None:
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, local_sources=[])

        result = rag.ask("how many files are in the data base")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("local Verilume library", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 0)

    def test_person_lookup_can_answer_from_verified_local_profile_when_model_refuses(self) -> None:
        local_source = LocalSource(
            label="S1",
            document="profile.pdf",
            page=1,
            chunk_id="damian-profile-1",
            text="Damian Mingo Ndiwago is a doctoral researcher at the University of Luxembourg.",
            score=0.88,
        )
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            local_sources=[local_source],
            web_sources=[],
        )
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)

        result = rag.ask("Who is Damian Mingo Ndiwago?")

        self.assertFalse(result.used_web)
        self.assertEqual([source.document for source in result.local_sources], ["profile.pdf"])
        self.assertIn("Damian Mingo Ndiwago", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertNotIn("could not answer", result.answer.lower())

    def test_phd_defense_date_answers_from_local_diploma(self) -> None:
        thesis_source = LocalSource(
            label="S1",
            document="Damian_diploma.pdf",
            page=1,
            chunk_id="diploma-defense",
            text="This letter is to confirm that Mr Damian MINGO NDIWAGO has defended his doctoral thesis on 26 September 2024.",
            score=0.93,
        )
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, model_answer=MODEL_UNKNOWN, local_sources=[thesis_source])

        result = rag.ask("when did Damian Mingo defend his phd")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertEqual([source.document for source in result.local_sources], ["Damian_diploma.pdf"])
        self.assertIn("26 September 2024", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual(len(rag.generator.model_calls), 0)

    def test_identity_lookup_rejects_noisy_comment_search_results(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Noisy answer [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Large Reasoning Models | Pavel Kordik",
                    url="https://www.linkedin.com/posts/example-large-reasoning-models",
                    content=(
                        "Image 3 Image 4 Comments Like Comment Share Copy Link. "
                        "Alex Sample commented on this unrelated post."
                    ),
                )
            ],
        )

        result = rag.ask("Alex Sample")

        self.assertEqual(result.web_sources, [])
        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "low")

    def test_uncited_web_sources_get_a_cited_fallback_answer(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="I found a source but will not cite it.",
            web_sources=[
                WebSource(
                    label="W1",
                    title="Bernard Fonlon",
                    url="https://example.com/fonlon",
                    content="Bernard Fonlon profile.",
                )
            ],
        )

        result = rag.ask("Bernard Fonlon")

        self.assertEqual([source.label for source in result.web_sources], ["W1"])
        self.assertIn(result.confidence, {"medium", "high"})
        self.assertIn("[W1]", result.answer)
        self.assertNotIn("will not guess", result.answer.lower())

    def test_web_response_keeps_five_best_sources_visible(self) -> None:
        web_sources = [
            WebSource(
                label=f"W{index}",
                title=f"Source {index}",
                url=f"https://example.com/{index}",
                content=f"Relevant source {index}",
            )
            for index in range(1, 7)
        ]
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Answer from the strongest source [W1]",
            web_sources=web_sources,
        )

        result = rag.ask("Search the web about Luxembourg")

        self.assertEqual(
            [source.label for source in result.web_sources], ["W1", "W2", "W3", "W4", "W5"]
        )

    def test_web_source_ranking_prefers_official_sources_over_noisy_results(self) -> None:
        ranked = _rank_web_sources(
            [
                WebSource(
                    label="W1",
                    title="Video about Luxembourg Prime Minister",
                    url="https://www.youtube.com/watch?v=abc",
                    content="Luxembourg Prime Minister Luc Frieden discusses Europe.",
                    score=0.99,
                ),
                WebSource(
                    label="W2",
                    title="FRIEDEN Luc - The Luxembourg Government",
                    url="https://gouvernement.lu/en/gouvernement/luc-frieden.html",
                    content="Luc Frieden is the Prime Minister of Luxembourg.",
                    score=0.2,
                ),
            ],
            ["who is the current prime minister of Luxembourg official government"],
        )

        self.assertEqual(ranked[0].url, "https://gouvernement.lu/en/gouvernement/luc-frieden.html")
        self.assertEqual([source.label for source in ranked], ["W1", "W2"])

    def test_web_source_ranking_drops_noisy_sources_when_enough_alternatives_exist(self) -> None:
        sources = [
            WebSource(
                label=f"W{index}",
                title=f"Official source {index}",
                url=f"https://gouvernement.lu/en/source-{index}",
                content="Prime Minister Luxembourg evidence.",
            )
            for index in range(1, 6)
        ]
        sources.append(
            WebSource(
                label="W6",
                title="Social result",
                url="https://www.facebook.com/example",
                content="Prime Minister Luxembourg social result.",
                score=100,
            )
        )

        ranked = _rank_web_sources(sources, ["prime minister Luxembourg"])

        self.assertNotIn("facebook.com", [source.url for source in ranked[:5]])

    def test_stop_request_raises_generation_stopped(self) -> None:
        rag = self._make_rag(local_answer="Local answer [S1]")

        with self.assertRaises(GenerationStopped):
            rag.ask("Who is this?", should_stop=lambda: True)

    def test_model_capacity_error_uses_web_when_web_is_configured(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN, web_answer="Web answer after model issue [W1]"
        )
        rag.generator = CapacityErrorGenerator(web_answer="Web answer after model issue [W1]")

        result = rag.ask("What does the file say?")

        self.assertEqual(result.confidence, "web-assisted")
        self.assertIn("Web answer after model issue [W1]", result.answer)
        self.assertTrue(result.used_web)

    def test_model_capacity_error_during_model_fallback_uses_web_when_enabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            web_answer="Alex Sample web answer [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Alex Sample profile",
                    url="https://example.com/alex-sample",
                    content="Alex Sample profile.",
                )
            ],
        )
        rag.generator = ModelOnlyCapacityErrorGenerator()

        result = rag.ask("Alex Sample")

        self.assertIn(result.confidence, {"medium", "high"})
        self.assertTrue(result.used_web)
        self.assertIn("[W1]", result.answer)
        self.assertNotEqual(result.confidence, "model-selection-warning")

    def test_model_capacity_error_returns_warning_when_web_is_unavailable(self) -> None:
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN)
        rag.settings = AppSettings(hf_token="token", enable_web_search=False)
        rag.generator = CapacityErrorGenerator()

        result = rag.ask("What does the file say?")

        self.assertEqual(result.confidence, "model-selection-warning")
        self.assertIn("Select another model", result.answer)
        self.assertFalse(result.used_web)

    def test_final_synthesis_error_returns_extractive_answer_with_confidence(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Official econometrics source",
                    url="https://example.edu/econometrics",
                    content=(
                        "Econometrics uses statistical methods to analyse economic data. "
                        "It is commonly used for forecasting and causal analysis."
                    ),
                    score=0.82,
                )
            ],
        )
        rag.generator = FinalSynthesisErrorGenerator(local_answer=LOCAL_UNKNOWN, model_answer=MODEL_UNKNOWN)

        result = rag.ask("Search the web about econometrics")

        self.assertIn("Confidence:", result.answer)
        self.assertIn("Econometrics uses statistical methods", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertNotIn("answer synthesis failed", result.answer.lower())
        self.assertFalse(result.answer.startswith("Confidence:"))
        self.assertIn(result.confidence, {"high", "medium"})

    def test_final_synthesis_uses_verified_evidence_payload(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Official econometrics source",
                    url="https://example.edu/econometrics",
                    content="Econometrics uses statistical methods to analyse economic data.",
                    score=0.82,
                )
            ],
        )
        rag.generator = StructuredFinalGenerator()

        result = rag.ask("Search the web about econometrics")

        self.assertIn("Confidence:", result.answer)
        self.assertIn("[W1]", result.answer)
        self.assertTrue(result.answer.startswith("Econometrics uses statistical methods"))
        self.assertEqual(len(rag.generator.chat_messages), 1)
        user_prompt = rag.generator.chat_messages[0][1]["content"]
        self.assertIn("Use only the verified evidence", user_prompt)
        self.assertIn("[W1] kind=web", user_prompt)

    def test_web_search_failure_falls_back_to_model_answer(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Luxembourg is a small European country.",
        )
        rag.web_search = BrokenWebSearch()

        result = rag.ask("Search the web about Luxembourg")

        self.assertEqual(result.confidence, "model-only")
        self.assertIn("Luxembourg is a small European country.", result.answer)
        self.assertIn("Web update: Tavily search could not complete", result.answer)
        self.assertFalse(result.used_web)

    def test_web_search_failure_keeps_local_answer_when_available(self) -> None:
        rag = self._make_rag(local_answer="Local answer [S1]")
        rag.web_search = BrokenWebSearch()

        result = rag.ask("Search the web for doc")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertEqual([source.label for source in result.local_sources], ["S1"])
        self.assertIn("Local answer [S1]", result.answer)
        self.assertIn("Web update: Tavily search could not complete", result.answer)
        self.assertFalse(result.used_web)

    def test_web_search_failure_without_local_or_model_answer_is_low_confidence(self) -> None:
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, model_answer=MODEL_UNKNOWN)
        rag.web_search = BrokenWebSearch()

        result = rag.ask("Search the web about Luxembourg")

        self.assertEqual(result.confidence, "low")
        self.assertIn("Tavily web search could not complete", result.answer)
        self.assertIn("What you can try:", result.answer)
        self.assertIn("Search forms to try:", result.answer)
        self.assertFalse(result.used_web)

    def test_web_search_fanout_deduplicates_equivalent_queries(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_sources=[
                WebSource(
                    label="W1",
                    title="Luxembourg government",
                    url="https://example.gov/luxembourg",
                    content="Luxembourg government source.",
                )
            ],
        )

        sources = rag._search_web_sources([" Luxembourg ", "luxembourg", "LUXEMBOURG"], question="Luxembourg")

        self.assertEqual(len(sources), 1)
        self.assertEqual(rag.web_search.queries, ["Luxembourg"])

    def test_merge_web_sources_uses_canonical_urls_and_keeps_best_source(self) -> None:
        weaker = WebSource(
            label="W1",
            title="Short source",
            url="https://www.example.com/story/?utm_source=news",
            content="Thin.",
            score=0.1,
        )
        stronger = WebSource(
            label="W2",
            title="Official source",
            url="http://example.com/story",
            content="A fuller official source with better evidence.",
            score=0.9,
        )

        merged = _merge_web_sources([weaker], [stronger], limit=5)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].title, "Official source")
        self.assertEqual(merged[0].label, "W1")
