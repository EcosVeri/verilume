from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from verilume.core.generation import HuggingFaceGenerator, OllamaGenerator, create_generator
from verilume.settings import AppSettings, save_user_config


class AppSettingsTests(unittest.TestCase):
    def test_settings_are_normalized(self) -> None:
        settings = AppSettings(
            docs_dir="~/tmp-docs",
            chroma_dir="~/tmp-chroma",
            manifest_path="~/tmp-manifest.json",
            chunk_size=50,
            chunk_overlap=500,
            max_workers=0,
            batch_size=0,
            hf_max_new_tokens=1,
            hf_temperature=-1,
            hf_timeout_seconds=0,
            web_search_provider="Brave Search",
            web_search_max_results=0,
            web_search_timeout_seconds=0,
            web_search_cache_ttl_seconds=-1,
            web_search_max_workers=0,
            tavily_max_results=0,
            tavily_timeout_seconds=0,
            retriever_k=0,
            max_history_turns=-2,
            query_rewrite_min_history=-3,
            query_rewrite_similarity_threshold=5,
            retrieval_score_threshold=4,
            rrf_constant=0,
            rrf_dense_weight=-1,
            rrf_lexical_weight=-1,
            rrf_semantic_boost=-1,
            rrf_score_scale=-1,
            rerank_mismatch_penalty=4,
            confidence_high_threshold=4,
            confidence_medium_threshold=-1,
            answer_style="academic",
        )

        self.assertIsInstance(settings.docs_dir, Path)
        self.assertGreaterEqual(settings.chunk_size, 100)
        self.assertLess(settings.chunk_overlap, settings.chunk_size)
        self.assertEqual(settings.max_workers, 1)
        self.assertEqual(settings.batch_size, 1)
        self.assertEqual(settings.hf_max_new_tokens, 32)
        self.assertEqual(settings.hf_temperature, 0.0)
        self.assertEqual(settings.hf_timeout_seconds, 5.0)
        self.assertEqual(settings.web_search_provider, "brave")
        self.assertEqual(settings.web_search_max_results, 1)
        self.assertEqual(settings.web_search_timeout_seconds, 5.0)
        self.assertEqual(settings.web_search_cache_ttl_seconds, 0.0)
        self.assertEqual(settings.web_search_max_workers, 1)
        self.assertEqual(settings.tavily_max_results, 1)
        self.assertEqual(settings.tavily_timeout_seconds, 5.0)
        self.assertEqual(settings.retriever_k, 1)
        self.assertEqual(settings.max_history_turns, 0)
        self.assertEqual(settings.query_rewrite_min_history, 0)
        self.assertEqual(settings.query_rewrite_similarity_threshold, 1.0)
        self.assertEqual(settings.retrieval_score_threshold, 1.0)
        self.assertEqual(settings.rrf_constant, 1)
        self.assertEqual(settings.rrf_dense_weight, 0.0)
        self.assertEqual(settings.rrf_lexical_weight, 0.0)
        self.assertEqual(settings.rrf_semantic_boost, 0.0)
        self.assertEqual(settings.rrf_score_scale, 0.0)
        self.assertEqual(settings.rerank_mismatch_penalty, 1.0)
        self.assertEqual(settings.confidence_high_threshold, 1.0)
        self.assertEqual(settings.confidence_medium_threshold, 0.0)
        self.assertEqual(settings.answer_style, "Research")

    def test_web_provider_secrets_are_masked(self) -> None:
        settings = AppSettings(
            web_search_provider="google_cse",
            google_cse_api_key="google-secret",
            google_cse_id="engine-id",
            brave_api_key="brave-secret",
            exa_api_key="exa-secret",
            serpapi_api_key="serp-secret",
            bing_api_key="bing-secret",
            custom_web_search_api_key="custom-secret",
        )

        values = settings.public_dict()

        self.assertTrue(settings.web_search_ready())
        self.assertEqual(settings.web_search_provider_label(), "Google CSE")
        self.assertEqual(values["google_cse_api_key"], "***")
        self.assertEqual(values["brave_api_key"], "***")
        self.assertEqual(values["exa_api_key"], "***")
        self.assertEqual(values["serpapi_api_key"], "***")
        self.assertEqual(values["bing_api_key"], "***")
        self.assertEqual(values["custom_web_search_api_key"], "***")

    def test_save_user_config_writes_local_env_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.env"
            saved_path = save_user_config(
                AppSettings(
                    hf_token="hf-secret",
                    hf_llm_model="Qwen/Qwen2.5-7B-Instruct",
                    web_search_provider="brave",
                    brave_api_key="brave-secret",
                    answer_style="short",
                ),
                path=path,
            )

            content = saved_path.read_text(encoding="utf-8")

        self.assertIn('HF_TOKEN="hf-secret"', content)
        self.assertIn('HF_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"', content)
        self.assertIn('WEB_SEARCH_PROVIDER="brave"', content)
        self.assertIn('BRAVE_API_KEY="brave-secret"', content)
        self.assertIn('ANSWER_STYLE="Short"', content)

    def test_hugging_face_backend_is_selected(self) -> None:
        settings = AppSettings(
            generation_backend="huggingface",
            hf_token="token",
            hf_llm_model="Qwen/Qwen2.5-7B-Instruct",
        )

        generator = create_generator(settings)

        self.assertIsInstance(generator, HuggingFaceGenerator)
        self.assertEqual(settings.active_generation_model(), "Qwen/Qwen2.5-7B-Instruct")

    def test_ollama_backend_is_selected(self) -> None:
        settings = AppSettings(
            generation_backend="ollama",
            ollama_model="llama3.2:3b",
        )

        generator = create_generator(settings)

        self.assertIsInstance(generator, OllamaGenerator)
        self.assertEqual(settings.active_generation_model(), "llama3.2:3b")
