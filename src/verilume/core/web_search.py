"""Optional web search providers."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from tavily import TavilyClient

from verilume.core.schemas import WebSource
from verilume.settings import AppSettings


DATE_PATTERN = re.compile(
    r"\b(?:20\d{2}|19\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{4})\b",
    re.IGNORECASE,
)


class WebSearchService:
    provider_name = "Web search"

    def __init__(self, max_results: int = 5, timeout_seconds: float = 20.0) -> None:
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return True

    def search(self, query: str) -> list[WebSource]:
        raise NotImplementedError


class TavilySearch(WebSearchService):
    provider_name = "Tavily"

    def __init__(self, api_key: str, max_results: int = 5, timeout_seconds: float = 20.0) -> None:
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key.strip()
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self):
        if self._client is None:
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []

        try:
            payload = self._client_search(query, time_range="year")
            results = payload.get("results", [])
            if not results:
                payload = self._client_search(query, time_range=None)
                results = payload.get("results", [])
        except Exception:
            results = self._search_with_requests(query)

        return _sources_from_items(results, self.max_results)

    def _client_search(self, query: str, time_range: str | None) -> dict:
        return self.client.search(
            query=query,
            max_results=self.max_results,
            search_depth="advanced",
            time_range=time_range,
            include_answer=False,
            include_raw_content="text",
            exact_match=False,
            timeout=self.timeout_seconds,
        )

    def _search_with_requests(self, query: str) -> list[dict]:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": "text",
                "max_results": self.max_results,
                "time_range": "year",
                "exact_match": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json().get("results", [])


class DuckDuckGoSearch(WebSearchService):
    provider_name = "DuckDuckGo"

    def search(self, query: str) -> list[WebSource]:
        if not query.strip():
            return []

        items: list[dict] = []
        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "no_redirect": 1,
                    "skip_disambig": 1,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            items = _duckduckgo_items(response.json())
        except Exception:
            items = []

        if len(items) < self.max_results:
            try:
                response = requests.get(
                    "https://html.duckduckgo.com/html/",
                    headers={"User-Agent": "Verilume/0.1 (+https://github.com/verilume)"},
                    params={"q": query},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                items.extend(_duckduckgo_html_items(response.text))
            except Exception:
                pass

        return _sources_from_items(_dedupe_result_items(items), self.max_results)


class BraveSearch(WebSearchService):
    provider_name = "Brave Search"

    def __init__(self, api_key: str, max_results: int = 5, timeout_seconds: float = 20.0) -> None:
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            params={"q": query, "count": self.max_results},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _sources_from_items(response.json().get("web", {}).get("results", []), self.max_results)


class ExaSearch(WebSearchService):
    provider_name = "Exa"

    def __init__(self, api_key: str, max_results: int = 5, timeout_seconds: float = 20.0) -> None:
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []
        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            },
            json={
                "query": query,
                "numResults": self.max_results,
                "contents": {"text": True},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _sources_from_items(response.json().get("results", []), self.max_results)


class SerpAPISearch(WebSearchService):
    provider_name = "SerpAPI"

    def __init__(self, api_key: str, max_results: int = 5, timeout_seconds: float = 20.0) -> None:
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": self.max_results,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _sources_from_items(response.json().get("organic_results", []), self.max_results)


class BingSearch(WebSearchService):
    provider_name = "Bing Search API"

    def __init__(self, api_key: str, max_results: int = 5, timeout_seconds: float = 20.0) -> None:
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            params={"q": query, "count": self.max_results, "responseFilter": "Webpages"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _sources_from_items(response.json().get("webPages", {}).get("value", []), self.max_results)


class GoogleCSESearch(WebSearchService):
    provider_name = "Google CSE"

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key.strip()
        self.search_engine_id = search_engine_id.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.search_engine_id)

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(self.max_results, 10),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _sources_from_items(response.json().get("items", []), self.max_results)


class CustomJsonSearch(WebSearchService):
    provider_name = "Custom provider"

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        endpoint: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self.provider_name = provider_name.strip() or self.provider_name
        self.api_key = api_key.strip()
        self.endpoint = endpoint.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []

        endpoint = self.endpoint
        params: dict[str, Any] = {}
        headers = {"Accept": "application/json"}
        if "{query}" in endpoint:
            endpoint = endpoint.replace("{query}", quote_plus(query))
        else:
            params["q"] = query

        if "{api_key}" in endpoint:
            endpoint = endpoint.replace("{api_key}", quote_plus(self.api_key))
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(
            endpoint,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _sources_from_items(_generic_result_items(response.json()), self.max_results)


def create_web_search(settings: AppSettings) -> WebSearchService:
    provider = settings.web_search_provider
    kwargs = {
        "max_results": settings.web_search_max_results,
        "timeout_seconds": settings.web_search_timeout_seconds,
    }
    if provider == "duckduckgo":
        return DuckDuckGoSearch(**kwargs)
    if provider == "brave":
        return BraveSearch(settings.brave_api_key, **kwargs)
    if provider == "exa":
        return ExaSearch(settings.exa_api_key, **kwargs)
    if provider == "serpapi":
        return SerpAPISearch(settings.serpapi_api_key, **kwargs)
    if provider == "bing":
        return BingSearch(settings.bing_api_key, **kwargs)
    if provider == "google_cse":
        return GoogleCSESearch(settings.google_cse_api_key, settings.google_cse_id, **kwargs)
    if provider == "custom":
        return CustomJsonSearch(
            settings.custom_web_search_provider,
            settings.custom_web_search_api_key,
            settings.custom_web_search_endpoint,
            **kwargs,
        )
    return TavilySearch(settings.tavily_api_key, **kwargs)


def _sources_from_items(items: list[dict], max_results: int) -> list[WebSource]:
    web_sources: list[WebSource] = []
    for item in items:
        if len(web_sources) >= max_results:
            break
        url = _first_text(item, "url", "link", "href")
        if not url:
            continue
        title = _first_text(item, "title", "name", "heading") or url
        content = _first_text(
            item,
            "content",
            "snippet",
            "description",
            "text",
            "body",
            "abstract",
            "raw_content",
        )
        web_sources.append(
            WebSource(
                label=f"W{len(web_sources) + 1}",
                title=title,
                url=url,
                content=content,
                score=_score(item),
                published_date=_published_date(item),
                metadata={"visible_dates": _visible_dates(item)},
            )
        )
    return web_sources


def _dedupe_result_items(items: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in items:
        url = _first_text(item, "url", "link", "href")
        key = _normalize_url_key(url)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _normalize_url_key(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    scheme = parsed.scheme.lower() or "https"
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{scheme}://{host}{path}"


def _duckduckgo_items(payload: dict) -> list[dict]:
    items: list[dict] = []
    abstract_url = str(payload.get("AbstractURL") or "").strip()
    abstract_text = str(payload.get("AbstractText") or "").strip()
    if abstract_url and abstract_text:
        items.append(
            {
                "title": payload.get("Heading") or abstract_url,
                "url": abstract_url,
                "content": abstract_text,
            }
        )

    for item in _flatten_related_topics(payload.get("Results", [])):
        items.append(item)
    for item in _flatten_related_topics(payload.get("RelatedTopics", [])):
        items.append(item)
    return items


def _duckduckgo_html_items(html: str) -> list[dict]:
    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)
    return parser.items


def _flatten_related_topics(values: list[dict]) -> list[dict]:
    items: list[dict] = []
    for value in values:
        if "Topics" in value:
            items.extend(_flatten_related_topics(value.get("Topics") or []))
            continue
        url = str(value.get("FirstURL") or "").strip()
        text = str(value.get("Text") or "").strip()
        if url and text:
            items.append({"title": text.split(" - ", 1)[0], "url": url, "content": text})
    return items


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict] = []
        self._current_link: dict[str, str] | None = None
        self._current_snippet: list[str] | None = None
        self._last_link: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = values.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._current_link = {"title": "", "url": _duckduckgo_result_url(values.get("href"))}
            return
        if "result__snippet" in classes and self._last_link is not None:
            self._current_snippet = []

    def handle_data(self, data: str) -> None:
        if self._current_link is not None:
            self._current_link["title"] += data
        elif self._current_snippet is not None:
            self._current_snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_link is not None:
            item = {
                "title": " ".join(self._current_link["title"].split()),
                "url": self._current_link["url"],
                "content": "",
            }
            if item["title"] and item["url"]:
                self.items.append(item)
                self._last_link = item
            self._current_link = None
            return

        if self._current_snippet is not None and tag in {"a", "div", "span"}:
            snippet = " ".join(" ".join(self._current_snippet).split())
            if snippet and self._last_link is not None:
                self._last_link["content"] = snippet
            self._current_snippet = None


def _duckduckgo_result_url(url: str | None) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    if value.startswith("//"):
        return f"https:{value}"
    return value


def _generic_result_items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for path in (
        ("results",),
        ("organic_results",),
        ("items",),
        ("web", "results"),
        ("webPages", "value"),
    ):
        value = _nested_value(payload, path)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _nested_value(payload: dict, path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_text(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return str(value).strip()
    return ""


def _score(item: dict) -> float | None:
    value = item.get("score") or item.get("position")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _published_date(item: dict) -> str | None:
    for key in (
        "published_date",
        "publishedDate",
        "published",
        "date",
        "dateLastCrawled",
        "page_age",
    ):
        value = item.get(key)
        if value:
            return str(value)
    return None


def _visible_dates(item: dict) -> list[str]:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in (
            "title",
            "name",
            "content",
            "snippet",
            "description",
            "text",
            "raw_content",
        )
    )
    values: list[str] = []
    for match in DATE_PATTERN.finditer(haystack):
        value = match.group(0).strip()
        if value not in values:
            values.append(value)
    return values[:5]
