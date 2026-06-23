"""Optional web search providers for Verilume."""

from __future__ import annotations

import copy
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Type
from urllib.parse import parse_qs, parse_qsl, quote_plus, urlencode, unquote, urlparse

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

TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "msclkid",
    "oly_enc_id",
    "ref",
    "ref_src",
    "spm",
    "utm",
}

PRIORITY_SOURCE_PROFILES: dict[str, tuple[str, ...]] = {
    "person": (
        "linkedin.com",
        "orcid.org",
        "scholar.google.",
        "researchgate.net",
        "github.com",
        ".edu",
        "ac.",
        ".gov",
    ),
    "news": (
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "ft.com",
        "financialtimes.com",
        "theguardian.com",
        "euronews.com",
        "aljazeera.com",
    ),
    "laws": (
        "eur-lex.europa.eu",
        "echa.europa.eu",
        "ec.europa.eu",
        "europa.eu",
        ".gov",
        "gov.",
    ),
    "science": (
        "arxiv.org",
        "doi.org",
        "semanticscholar.org",
        "pubmed.ncbi.nlm.nih.gov",
        "nature.com",
        "science.org",
        "ieee.org",
        "acm.org",
    ),
    "government": (
        ".gov",
        "gov.",
        ".mil",
        "europa.eu",
        "un.org",
        "government",
        "gouvernement.lu",
        "public.lu",
    ),
}


class WebSearchService:
    """Base class for all Verilume web search providers."""

    provider_name = "Web search"

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        search = cls.__dict__.get("search")
        if search is None or getattr(search, "_verilume_cached", False):
            return

        def cached_search(self, query: str) -> list[WebSource]:
            cached = self._cached_search_result(query)
            if cached is not None:
                return cached
            sources = search(self, query)
            self._store_search_result(query, sources)
            return sources

        cached_search._verilume_cached = True
        cls.search = cached_search

    def __init__(
        self,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
        cache_ttl_seconds: float = 600.0,
    ) -> None:
        self.max_results = max(1, int(max_results))
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._search_cache: dict[str, tuple[float, list[WebSource]]] = {}

    @property
    def is_configured(self) -> bool:
        return True

    def search(self, query: str) -> list[WebSource]:
        raise NotImplementedError

    def _cached_search_result(self, query: str) -> list[WebSource] | None:
        if self.cache_ttl_seconds <= 0:
            return None
        key = _search_cache_key(query, self.max_results)
        cached = self._search_cache.get(key)
        if cached is None:
            return None
        expires_at, sources = cached
        if time.monotonic() >= expires_at:
            self._search_cache.pop(key, None)
            return None
        return copy.deepcopy(sources)

    def _store_search_result(self, query: str, sources: list[WebSource]) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        key = _search_cache_key(query, self.max_results)
        self._search_cache[key] = (
            time.monotonic() + self.cache_ttl_seconds,
            copy.deepcopy(sources),
        )


WEB_SEARCH_PROVIDERS: dict[str, Type[WebSearchService]] = {}


def register_web_search_provider(
    name: str,
    provider_class: Type[WebSearchService],
) -> None:
    """
    Register a custom web search provider.

    A provider must inherit from WebSearchService and implement:

        search(self, query: str) -> list[WebSource]

    Example:

        register_web_search_provider("my_search", MySearchProvider)
    """
    key = _provider_key(name)

    if not key:
        raise ValueError("Provider name cannot be empty.")

    if not issubclass(provider_class, WebSearchService):
        raise TypeError("provider_class must inherit from WebSearchService.")

    WEB_SEARCH_PROVIDERS[key] = provider_class


def available_web_search_providers() -> list[str]:
    """Return registered external/custom provider names."""
    return sorted(WEB_SEARCH_PROVIDERS)


def classify_query_domain(query: str) -> str:
    normalized = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if not normalized:
        return "general"
    if any(
        marker in normalized
        for marker in (
            "government",
            "official",
            "minister",
            "president",
            "king",
            "queen",
            "grand duke",
        )
    ):
        return "government"
    if any(marker in normalized for marker in ("law", "regulation", "legal", "directive", "statute")):
        return "laws"
    if any(marker in normalized for marker in ("paper", "study", "research", "journal", "scientific", "doi", "arxiv")):
        return "science"
    if any(marker in normalized for marker in ("news", "recent", "latest", "breaking", "resign")):
        return "news"
    if any(marker in normalized for marker in ("who is", "biography", "profile", "researcher", "professor", "ceo", "founder")):
        return "person"
    return "general"


def boost_priority_sources(
    sources: list[WebSource],
    domain: str,
    *,
    boost: float = 1.3,
) -> list[WebSource]:
    priority_domains = PRIORITY_SOURCE_PROFILES.get(domain, ())
    if not priority_domains:
        return list(sources)
    boosted = list(sources)
    for source in boosted:
        url_lower = (source.url or "").lower()
        if any(priority in url_lower for priority in priority_domains):
            current_score = float(source.score or 0.5)
            source.score = min(1.0, current_score * boost)
            source.metadata = dict(source.metadata or {})
            source.metadata["priority_source"] = True
            source.metadata["priority_domain"] = domain
    return sorted(boosted, key=lambda source: float(source.score or 0.0), reverse=True)


class TavilySearch(WebSearchService):
    provider_name = "Tavily"

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
        self.api_key = api_key.strip()
        self._client: TavilyClient | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self) -> TavilyClient:
        if self._client is None:
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def search(self, query: str) -> list[WebSource]:
        if not self.is_configured or not query.strip():
            return []

        try:
            payload = self._client_search(query, time_range=None)
            results = payload.get("results", [])

        except Exception:
            results = self._search_with_requests(query)

        return _sources_from_items(results, self.max_results)

    def _client_search(
        self,
        query: str,
        time_range: str | None,
    ) -> dict[str, Any]:
        return self.client.search(
            query=query,
            max_results=self.max_results,
            search_depth="basic",
            time_range=time_range,
            include_answer=False,
            include_raw_content=False,
            exact_match=False,
            timeout=self.timeout_seconds,
        )

    def _search_with_requests(self, query: str) -> list[dict[str, Any]]:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "max_results": self.max_results,
                "exact_match": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", [])


class DuckDuckGoSearch(WebSearchService):
    provider_name = "DuckDuckGo"

    def search(self, query: str) -> list[WebSource]:
        if not query.strip():
            return []

        items: list[dict[str, Any]] = []

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
                    headers={
                        "User-Agent": ("Verilume/0.1 (+https://github.com/verilume/verilume)")
                    },
                    params={"q": query},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                items.extend(_duckduckgo_html_items(response.text))
            except Exception:
                pass

        return _sources_from_items(
            _dedupe_result_items(items),
            self.max_results,
        )


class BraveSearch(WebSearchService):
    provider_name = "Brave Search"

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
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
            params={
                "q": query,
                "count": self.max_results,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        items = payload.get("web", {}).get("results", [])
        return _sources_from_items(items, self.max_results)


class ExaSearch(WebSearchService):
    provider_name = "Exa"

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
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

        payload = response.json()
        return _sources_from_items(payload.get("results", []), self.max_results)


class SerpAPISearch(WebSearchService):
    provider_name = "SerpAPI"

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
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

        payload = response.json()
        return _sources_from_items(
            payload.get("organic_results", []),
            self.max_results,
        )


class BingSearch(WebSearchService):
    provider_name = "Bing Search API"

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
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
            params={
                "q": query,
                "count": self.max_results,
                "responseFilter": "Webpages",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        items = payload.get("webPages", {}).get("value", [])
        return _sources_from_items(items, self.max_results)


class GoogleCSESearch(WebSearchService):
    provider_name = "Google CSE"

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
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

        payload = response.json()
        return _sources_from_items(payload.get("items", []), self.max_results)


class CustomJsonSearch(WebSearchService):
    """
    Generic custom JSON search provider.

    Supported endpoint patterns:

    1. Query parameter style:
       https://example.com/search
       Verilume sends ?q=<query>

    2. Template style:
       https://example.com/search?q={query}&key={api_key}

    The response should contain one of:
       results
       organic_results
       items
       web.results
       webPages.value
    """

    provider_name = "Custom provider"

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        endpoint: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
    ) -> None:
        super().__init__(
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
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

        return _sources_from_items(
            _generic_result_items(response.json()),
            self.max_results,
        )


@dataclass(frozen=True, slots=True)
class WebSearchProviderMeta:
    name: str
    provider_class: Type[WebSearchService]
    factory: Callable[[AppSettings, dict[str, Any]], WebSearchService]
    requires_api_key: bool = True
    config_fields: tuple[str, ...] = ("api_key",)


def _provider_kwargs(settings: AppSettings) -> dict[str, Any]:
    return {
        "max_results": settings.web_search_max_results,
        "timeout_seconds": settings.web_search_timeout_seconds,
    }


BUILTIN_WEB_SEARCH_PROVIDERS: dict[str, WebSearchProviderMeta] = {
    "tavily": WebSearchProviderMeta(
        "Tavily",
        TavilySearch,
        lambda settings, kwargs: TavilySearch(settings.tavily_api_key, **kwargs),
        config_fields=("tavily_api_key",),
    ),
    "duckduckgo": WebSearchProviderMeta(
        "DuckDuckGo",
        DuckDuckGoSearch,
        lambda _settings, kwargs: DuckDuckGoSearch(**kwargs),
        requires_api_key=False,
        config_fields=(),
    ),
    "brave": WebSearchProviderMeta(
        "Brave Search",
        BraveSearch,
        lambda settings, kwargs: BraveSearch(settings.brave_api_key, **kwargs),
        config_fields=("brave_api_key",),
    ),
    "exa": WebSearchProviderMeta(
        "Exa",
        ExaSearch,
        lambda settings, kwargs: ExaSearch(settings.exa_api_key, **kwargs),
        config_fields=("exa_api_key",),
    ),
    "serpapi": WebSearchProviderMeta(
        "SerpAPI",
        SerpAPISearch,
        lambda settings, kwargs: SerpAPISearch(settings.serpapi_api_key, **kwargs),
        config_fields=("serpapi_api_key",),
    ),
    "bing": WebSearchProviderMeta(
        "Bing Search API",
        BingSearch,
        lambda settings, kwargs: BingSearch(settings.bing_api_key, **kwargs),
        config_fields=("bing_api_key",),
    ),
    "google_cse": WebSearchProviderMeta(
        "Google CSE",
        GoogleCSESearch,
        lambda settings, kwargs: GoogleCSESearch(
            settings.google_cse_api_key,
            settings.google_cse_id,
            **kwargs,
        ),
        config_fields=("google_cse_api_key", "google_cse_id"),
    ),
    "custom": WebSearchProviderMeta(
        "Custom provider",
        CustomJsonSearch,
        lambda settings, kwargs: CustomJsonSearch(
            settings.custom_web_search_provider,
            settings.custom_web_search_api_key,
            settings.custom_web_search_endpoint,
            **kwargs,
        ),
        requires_api_key=False,
        config_fields=("custom_web_search_provider", "custom_web_search_endpoint"),
    ),
}


def create_web_search(settings: AppSettings) -> WebSearchService:
    """
    Create the selected web search provider.

    Built-in providers:
    - tavily
    - duckduckgo
    - brave
    - exa
    - serpapi
    - bing
    - google_cse
    - custom

    External providers:
    - register with register_web_search_provider("name", ProviderClass)
    - set WEB_SEARCH_PROVIDER=name
    """
    provider = _provider_key(settings.web_search_provider)
    kwargs = _provider_kwargs(settings)

    meta = BUILTIN_WEB_SEARCH_PROVIDERS.get(provider)
    if meta is not None:
        return _configure_search_cache(
            meta.factory(settings, kwargs),
            settings.web_search_cache_ttl_seconds,
        )

    provider_class = WEB_SEARCH_PROVIDERS.get(provider)

    if provider_class is not None:
        return _configure_search_cache(
            provider_class(**kwargs),
            settings.web_search_cache_ttl_seconds,
        )

    return _configure_search_cache(
        BUILTIN_WEB_SEARCH_PROVIDERS["tavily"].factory(settings, kwargs),
        settings.web_search_cache_ttl_seconds,
    )


def _configure_search_cache(service: WebSearchService, ttl_seconds: float) -> WebSearchService:
    service.cache_ttl_seconds = max(0.0, float(ttl_seconds))
    service._search_cache.clear()
    return service


def _provider_key(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _search_cache_key(query: str, max_results: int) -> str:
    normalized_query = re.sub(r"\s+", " ", (query or "").strip().lower())
    return f"{max(1, int(max_results))}:{normalized_query}"


def _sources_from_items(
    items: list[dict[str, Any]],
    max_results: int,
) -> list[WebSource]:
    web_sources: list[WebSource] = []

    for item in items:
        if len(web_sources) >= max_results:
            break

        url = _first_text(item, "url", "link", "href")

        if not url:
            continue

        title = (
            _first_text(
                item,
                "title",
                "name",
                "heading",
            )
            or url
        )

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


def _dedupe_result_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        url = _first_text(item, "url", "link", "href")
        key = _normalize_url_key(url)

        if not key or key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return deduped


def normalize_web_url_key(url: str) -> str:
    """Return a stable URL key for cross-provider source deduplication."""
    value = (url or "").strip()

    if not value:
        return ""

    parsed = urlparse(value)
    if not parsed.netloc and parsed.path:
        parsed = urlparse(f"https://{value}")

    host = parsed.netloc.lower().removeprefix("www.")
    host = re.sub(r":(?:80|443)$", "", host)
    if not host:
        return ""

    path = re.sub(r"/+", "/", parsed.path or "/")
    path = re.sub(r"/(?:index|default)\.(?:html?|aspx?)$", "/", path, flags=re.IGNORECASE)
    path = path.rstrip("/") or "/"

    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in TRACKING_QUERY_PARAMS:
            continue
        query_items.append((lowered, value.strip()))
    query = urlencode(sorted(query_items))
    return f"{host}{path}{f'?{query}' if query else ''}"


def _normalize_url_key(url: str) -> str:
    return normalize_web_url_key(url)


def _duckduckgo_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

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


def _duckduckgo_html_items(html: str) -> list[dict[str, Any]]:
    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)
    return parser.items


def _flatten_related_topics(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for value in values:
        if "Topics" in value:
            items.extend(_flatten_related_topics(value.get("Topics") or []))
            continue

        url = str(value.get("FirstURL") or "").strip()
        text = str(value.get("Text") or "").strip()

        if url and text:
            items.append(
                {
                    "title": text.split(" - ", 1)[0],
                    "url": url,
                    "content": text,
                }
            )

    return items


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict[str, str]] = []
        self._current_link: dict[str, str] | None = None
        self._current_snippet: list[str] | None = None
        self._last_link: dict[str, str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attrs)
        classes = values.get("class", "")

        if tag == "a" and "result__a" in classes:
            self._current_link = {
                "title": "",
                "url": _duckduckgo_result_url(values.get("href")),
            }
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


def _generic_result_items(payload: Any) -> list[dict[str, Any]]:
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
        ("data",),
        ("documents",),
    ):
        value = _nested_value(payload, path)

        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload

    for key in path:
        if not isinstance(value, dict):
            return None

        value = value.get(key)

    return value


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def _score(item: dict[str, Any]) -> float | None:
    value = item.get("score")

    if value is None:
        value = item.get("position")

    if value is None:
        value = item.get("rank")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _published_date(item: dict[str, Any]) -> str | None:
    for key in (
        "published_date",
        "publishedDate",
        "published",
        "date",
        "dateLastCrawled",
        "page_age",
        "created_at",
        "updated_at",
    ):
        value = item.get(key)

        if value:
            return str(value)

    return None


def _visible_dates(item: dict[str, Any]) -> list[str]:
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
            "body",
        )
    )

    values: list[str] = []

    for match in DATE_PATTERN.finditer(haystack):
        value = match.group(0).strip()

        if value not in values:
            values.append(value)

    return values[:5]
