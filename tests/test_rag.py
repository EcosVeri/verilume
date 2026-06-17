from __future__ import annotations

import unittest

from verilume.core.generation import (
    LOCAL_UNKNOWN,
    MODEL_SELECTION_WARNING,
    MODEL_UNKNOWN,
    GenerationError,
)
from verilume.core.schemas import LocalSource, WebSource
from verilume.rag import LOCAL_FILE_NOT_FOUND, VerilumeRAG, GenerationStopped, _rank_web_sources
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


class SequentialLocalRetriever:
    def __init__(self, batches):
        self.batches = [list(batch) for batch in batches]
        self.calls: list[tuple[tuple, dict]] = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self.batches:
            return []
        return self.batches.pop(0)


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

    def test_local_file_question_searches_expanded_keywords_and_does_not_use_ai_or_web(self) -> None:
        rag = self._make_rag(local_answer="This should not be called.", local_sources=[])
        rag.retriever = SequentialLocalRetriever([[], []])

        result = rag.ask("Is Damian's language certificate in the local files?")

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
            text="Damian Ndiwago Sproochentest language certificate result.",
            score=0.82,
        )
        rag = self._make_rag(local_answer="This should not be called.", local_sources=[])
        rag.retriever = SequentialLocalRetriever([[], [expanded_source]])

        result = rag.ask("Which document contains my language certificate?")

        self.assertEqual(result.confidence, "local-grounded")
        self.assertFalse(result.used_web)
        self.assertIn("language_certificate.pdf", result.answer)
        self.assertIn("[S1]", result.answer)
        self.assertEqual([source.document for source in result.local_sources], ["language_certificate.pdf"])
        self.assertEqual(len(rag.retriever.calls), 2)
        self.assertEqual(rag.generator.local_calls, [])
        self.assertEqual(rag.generator.model_calls, [])
        self.assertEqual(rag.web_search.queries, [])

    def test_local_gap_uses_model_and_web_when_web_is_enabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer="Econometrics applies statistical methods to economic data.",
            web_answer="Econometrics combines statistics and economics [W1]",
        )
        result = rag.ask("What is econometrics?")

        self.assertEqual(result.confidence, "web-assisted")
        self.assertTrue(result.used_web)
        self.assertIn("[W1]", result.answer)
        self.assertEqual(rag.generator.final_calls[0]["model_answer"], "Econometrics applies statistical methods to economic data.")
        self.assertEqual([source.label for source in result.web_sources], ["W1", "W2"])

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

    def test_web_fallback_filters_to_used_web_citations(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            model_answer=MODEL_UNKNOWN,
            web_answer="Web answer [W1]",
            web_sources=[
                WebSource(label="W1", title="Dylan Mingo", url="https://example.com/wrong", content="Wrong person"),
                WebSource(
                    label="W2",
                    title="Damian Mingo profile",
                    url="https://example.com/damian",
                    content="Damian Mingo is mentioned here.",
                ),
            ],
        )
        result = rag.ask("Who is Damian Mingo?")

        self.assertEqual(result.confidence, "web-assisted")
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
            web_answer="Damian Ndiwago profile [W1]",
            local_sources=[],
            web_sources=[],
        )
        rag._search_duckduckgo_fallback_queries = lambda queries: [
            WebSource(
                label="W1",
                title="Damian Ndiwago profile",
                url="https://example.com/damian-ndiwago",
                content="Damian Ndiwago is listed on this profile.",
            )
        ]

        result = rag.ask("Damian Ndiwago")

        self.assertEqual([source.label for source in result.web_sources], ["W1"])
        self.assertTrue(result.used_web)
        self.assertIn("[W1]", result.answer)

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
        self.assertEqual(rag.generator.final_calls[0]["model_answer"], "Older model answer says someone else.")

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
        self.assertIn("Evidence conflict detected", result.answer)
        self.assertIn("Keir Starmer", result.answer)
        self.assertNotIn("Rishi Sunak", result.answer)
        self.assertTrue(result.diagnostics["current_role_override"])
        self.assertTrue(result.diagnostics["evidence_conflict"])
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
                    title="Dylan Mingo",
                    url="https://example.com/dylan",
                    content="Dylan Mingo is a different person.",
                )
            ],
        )

        result = rag.ask("Damian Mingo Ndiwago")

        self.assertEqual(result.local_sources, [])
        self.assertEqual(result.web_sources, [])
        self.assertFalse(result.used_web)
        self.assertEqual(result.confidence, "low")

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
                        "Damian Ndiwago commented on this unrelated post."
                    ),
                )
            ],
        )

        result = rag.ask("Damian Ndiwago")

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
        self.assertEqual(result.confidence, "web-assisted")
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

        self.assertEqual([source.label for source in result.web_sources], ["W1", "W2", "W3", "W4", "W5"])

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
        rag = self._make_rag(local_answer=LOCAL_UNKNOWN, web_answer="Web answer after model issue [W1]")
        rag.generator = CapacityErrorGenerator(web_answer="Web answer after model issue [W1]")

        result = rag.ask("What does the file say?")

        self.assertEqual(result.confidence, "web-assisted")
        self.assertIn("Web answer after model issue [W1]", result.answer)
        self.assertTrue(result.used_web)

    def test_model_capacity_error_during_model_fallback_uses_web_when_enabled(self) -> None:
        rag = self._make_rag(
            local_answer=LOCAL_UNKNOWN,
            web_answer="Damian Ndiwago web answer [W1]",
            local_sources=[],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Damian Ndiwago profile",
                    url="https://example.com/damian",
                    content="Damian Ndiwago profile.",
                )
            ],
        )
        rag.generator = ModelOnlyCapacityErrorGenerator()

        result = rag.ask("Damian Ndiwago")

        self.assertEqual(result.confidence, "web-assisted")
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
        self.assertFalse(result.used_web)
