from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory

from verilume.core.evidence import EvidencePolicy
from verilume.core.schemas import ChatMessage, LocalSource, RAGResponse, WebSource
from verilume.core.semantic_cache import (
    SemanticCache,
    document_fingerprint,
    semantic_cache_ttl_seconds,
)
from verilume.rag import VerilumeRAG
from verilume.settings import AppSettings


class SemanticCacheTests(unittest.TestCase):
    def _settings(self, tmp: str) -> AppSettings:
        return AppSettings(
            docs_dir=f"{tmp}/docs",
            chroma_dir=f"{tmp}/chroma",
            manifest_path=f"{tmp}/manifest.json",
            semantic_cache_path=f"{tmp}/semantic_cache.json",
            hf_token="token",
            tavily_api_key="key",
        )

    def _response(self, answer: str = "Regression models relationships [S1].") -> RAGResponse:
        return RAGResponse(
            answer=answer,
            local_sources=[
                LocalSource(
                    label="S1",
                    document="regression.pdf",
                    page=1,
                    chunk_id="chunk-1",
                    text="Regression models relationships between variables.",
                    score=0.92,
                )
            ],
            web_sources=[
                WebSource(
                    label="W1",
                    title="Regression overview",
                    url="https://example.edu/regression",
                    content="Regression analysis models relationships between variables.",
                    score=0.81,
                )
            ],
            used_web=True,
            confidence="high",
            diagnostics={
                "model_answer": "Regression is a statistical modelling method.",
                "ranked_evidence": [{"label": "S1", "final_score": 0.94}],
            },
        )

    def test_semantic_cache_reuses_close_stable_query(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            cache = SemanticCache(settings.semantic_cache_path)
            fingerprint = document_fingerprint(settings)

            cache.store(
                "What is regression analysis?",
                self._response(),
                policy=EvidencePolicy.LOCAL_MODEL_WEB,
                document_fingerprint=fingerprint,
                web_enabled=True,
                generation_backend=settings.generation_backend,
                model_name=settings.active_generation_model(),
                web_provider=settings.web_search_provider,
            )

            cached = cache.lookup(
                "Explain regression analysis",
                policy=EvidencePolicy.LOCAL_MODEL_WEB,
                document_fingerprint=fingerprint,
                web_enabled=True,
                generation_backend=settings.generation_backend,
                model_name=settings.active_generation_model(),
                web_provider=settings.web_search_provider,
            )

            self.assertIsNotNone(cached)
            response = cached.to_rag_response() if cached else None
            self.assertEqual(response.answer, "Regression models relationships [S1].")
            self.assertTrue(response.diagnostics["semantic_cache_hit"])
            self.assertEqual(response.local_sources[0].document, "regression.pdf")

    def test_semantic_cache_misses_when_web_or_model_context_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            cache = SemanticCache(settings.semantic_cache_path)
            fingerprint = document_fingerprint(settings)

            cache.store(
                "Who is Florian Felice?",
                self._response("Florian Felice is a researcher [W1]."),
                policy=EvidencePolicy.LOCAL_MODEL_WEB,
                document_fingerprint=fingerprint,
                web_enabled=True,
                generation_backend=settings.generation_backend,
                model_name=settings.active_generation_model(),
                web_provider=settings.web_search_provider,
            )

            self.assertIsNone(
                cache.lookup(
                    "Who is Florian Felice?",
                    policy=EvidencePolicy.LOCAL_MODEL_WEB,
                    document_fingerprint=fingerprint,
                    web_enabled=False,
                    generation_backend=settings.generation_backend,
                    model_name=settings.active_generation_model(),
                    web_provider=settings.web_search_provider,
                )
            )
            self.assertIsNone(
                cache.lookup(
                    "Who is Florian Felice?",
                    policy=EvidencePolicy.LOCAL_MODEL_WEB,
                    document_fingerprint=fingerprint,
                    web_enabled=True,
                    generation_backend=settings.generation_backend,
                    model_name="different-model",
                    web_provider=settings.web_search_provider,
                )
            )

    def test_current_fact_cache_expires_by_ttl(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            cache = SemanticCache(settings.semantic_cache_path)
            fingerprint = document_fingerprint(settings)

            cache.store(
                "Who is the current CEO of OpenAI?",
                self._response("A current answer [W1]."),
                policy=EvidencePolicy.LOCAL_PLUS_WEB,
                document_fingerprint=fingerprint,
                web_enabled=True,
                generation_backend=settings.generation_backend,
                model_name=settings.active_generation_model(),
                web_provider=settings.web_search_provider,
            )

            payload = json.loads(settings.semantic_cache_path.read_text(encoding="utf-8"))
            payload["entries"][0]["timestamp"] = (
                datetime.now(timezone.utc) - timedelta(hours=3)
            ).isoformat()
            settings.semantic_cache_path.write_text(json.dumps(payload), encoding="utf-8")

            cached = cache.lookup(
                "Who is the current CEO of OpenAI?",
                policy=EvidencePolicy.LOCAL_PLUS_WEB,
                document_fingerprint=fingerprint,
                web_enabled=True,
                generation_backend=settings.generation_backend,
                model_name=settings.active_generation_model(),
                web_provider=settings.web_search_provider,
            )

            self.assertIsNotNone(cached)
            self.assertFalse(
                cached.is_fresh(
                    datetime.now(timezone.utc),
                    ttl_seconds=60,
                    current_document_fingerprint=fingerprint,
                )
            )

    def test_local_document_cache_invalidates_when_fingerprint_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            settings.docs_dir.mkdir(parents=True)
            settings.manifest_path.write_text('{"files": ["one.pdf"]}', encoding="utf-8")
            first_fingerprint = document_fingerprint(settings)
            cache = SemanticCache(settings.semantic_cache_path)

            cache.store(
                "Summarise the uploaded documents",
                self._response("The uploaded document covers regression [S1]."),
                policy=EvidencePolicy.LOCAL_ONLY,
                document_fingerprint=first_fingerprint,
                web_enabled=True,
                generation_backend=settings.generation_backend,
                model_name=settings.active_generation_model(),
                web_provider=settings.web_search_provider,
            )

            settings.manifest_path.write_text('{"files": ["one.pdf", "two.pdf"]}', encoding="utf-8")
            changed_fingerprint = document_fingerprint(settings)

            self.assertNotEqual(first_fingerprint, changed_fingerprint)
            self.assertIsNone(
                cache.lookup(
                    "Summarize uploaded documents",
                    policy=EvidencePolicy.LOCAL_ONLY,
                    document_fingerprint=changed_fingerprint,
                    web_enabled=True,
                    generation_backend=settings.generation_backend,
                    model_name=settings.active_generation_model(),
                    web_provider=settings.web_search_provider,
                )
            )

    def test_rag_semantic_cache_hooks_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            rag = object.__new__(VerilumeRAG)
            rag.settings = settings
            rag.semantic_cache = SemanticCache(settings.semantic_cache_path)

            rag._store_semantic_cached_response(
                "What is regression analysis?",
                self._response(),
            )
            cached = rag._semantic_cached_response("Explain regression analysis")

            self.assertIsNotNone(cached)
            self.assertTrue(cached.diagnostics["semantic_cache_hit"])
            self.assertEqual(cached.answer, "Regression models relationships [S1].")

    def test_rag_ask_skips_semantic_cache_for_contextual_followup(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            cache = SemanticCache(settings.semantic_cache_path)
            fingerprint = document_fingerprint(settings)
            cache.store(
                "What is regression analysis?",
                self._response(),
                policy=EvidencePolicy.LOCAL_MODEL_WEB,
                document_fingerprint=fingerprint,
                web_enabled=True,
                generation_backend=settings.generation_backend,
                model_name=settings.active_generation_model(),
                web_provider=settings.web_search_provider,
            )

            fresh_response = self._response("Fresh contextual answer [S1].")
            rag = object.__new__(VerilumeRAG)
            rag.settings = settings
            rag.semantic_cache = cache
            rag._response_cache = {}
            rag._ask_uncached = lambda question, history, conversation_state, should_stop, on_stage, **kwargs: fresh_response

            result = VerilumeRAG.ask(
                rag,
                "What about it?",
                history=[ChatMessage(role="user", content="What is regression analysis?")],
            )

            self.assertEqual(result.answer, "Fresh contextual answer [S1].")
            self.assertNotIn("semantic_cache_hit", result.diagnostics)

    def test_dynamic_questions_use_short_semantic_cache_ttl(self) -> None:
        settings = AppSettings(
            semantic_cache_current_ttl_seconds=123,
            semantic_cache_stable_ttl_seconds=999,
        )
        from verilume.core.evidence import classify_question

        understanding = classify_question("What is the latest weather in Luxembourg today?")

        self.assertEqual(semantic_cache_ttl_seconds(understanding, settings), 123)


if __name__ == "__main__":
    unittest.main()
