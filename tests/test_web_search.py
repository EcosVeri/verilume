from __future__ import annotations

import unittest
from unittest.mock import patch

from verilume.core.web_search import BraveSearch, DuckDuckGoSearch, TavilySearch, create_web_search
from verilume.settings import AppSettings


class FakeResponse:
    def __init__(self, payload: dict | None = None, text: str = "") -> None:
        self.text = text
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload or {}


class FakeTavilyClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"results": []}


class WebSearchTests(unittest.TestCase):
    def test_factory_uses_selected_provider(self) -> None:
        settings = AppSettings(web_search_provider="brave", brave_api_key="key")

        service = create_web_search(settings)

        self.assertIsInstance(service, BraveSearch)
        self.assertTrue(service.is_configured)

    def test_brave_results_are_normalized(self) -> None:
        payload = {
            "web": {
                "results": [
                    {
                        "title": "Example result",
                        "url": "https://example.com",
                        "description": "A search snippet from 2026.",
                        "page_age": "2026-06-16",
                    }
                ]
            }
        }

        with patch("verilume.core.web_search.requests.get", return_value=FakeResponse(payload)):
            sources = BraveSearch("key").search("example query")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].label, "W1")
        self.assertEqual(sources[0].title, "Example result")
        self.assertEqual(sources[0].url, "https://example.com")
        self.assertEqual(sources[0].published_date, "2026-06-16")

    def test_duckduckgo_is_ready_without_key(self) -> None:
        settings = AppSettings(web_search_provider="duckduckgo")

        service = create_web_search(settings)

        self.assertTrue(settings.web_search_ready())
        self.assertTrue(service.is_configured)

    def test_duckduckgo_falls_back_to_html_results(self) -> None:
        html = """
        <html>
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">
            Example Duck Result
          </a>
          <a class="result__snippet">Snippet from DuckDuckGo result.</a>
        </html>
        """

        with patch(
            "verilume.core.web_search.requests.get",
            side_effect=[
                FakeResponse({"AbstractText": "", "Results": [], "RelatedTopics": []}),
                FakeResponse(text=html),
            ],
        ):
            sources = DuckDuckGoSearch().search("Bernard Fonlon")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].label, "W1")
        self.assertEqual(sources[0].title, "Example Duck Result")
        self.assertEqual(sources[0].url, "https://example.com")
        self.assertEqual(sources[0].content, "Snippet from DuckDuckGo result.")

    def test_duckduckgo_adds_html_results_when_instant_answer_is_thin(self) -> None:
        html = """
        <html>
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgovernment.example%2Fpm">
            Official Prime Minister
          </a>
          <a class="result__snippet">Luc Frieden is prime minister.</a>
        </html>
        """
        payload = {
            "Heading": "Luxembourg",
            "AbstractURL": "https://example.com/luxembourg",
            "AbstractText": "Luxembourg is a country.",
            "Results": [],
            "RelatedTopics": [],
        }

        with patch(
            "verilume.core.web_search.requests.get",
            side_effect=[
                FakeResponse(payload),
                FakeResponse(text=html),
            ],
        ):
            sources = DuckDuckGoSearch(max_results=5).search("current prime minister Luxembourg")

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].url, "https://example.com/luxembourg")
        self.assertEqual(sources[1].url, "https://government.example/pm")

    def test_tavily_search_does_not_force_exact_match(self) -> None:
        service = TavilySearch("key")
        fake_client = FakeTavilyClient()
        service._client = fake_client

        sources = service.search("Luxembourg")

        self.assertEqual(sources, [])
        self.assertGreaterEqual(len(fake_client.calls), 1)
        self.assertFalse(fake_client.calls[0]["exact_match"])
