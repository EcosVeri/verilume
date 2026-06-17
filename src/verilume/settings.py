"""Application settings loaded from environment variables and UI overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


HOME = Path.home()
DATA_HOME = HOME / ".verilume"
USER_CONFIG_PATH = DATA_HOME / "config.env"

WEB_SEARCH_PROVIDER_LABELS = {
    "tavily": "Tavily",
    "duckduckgo": "DuckDuckGo",
    "brave": "Brave Search",
    "exa": "Exa",
    "serpapi": "SerpAPI",
    "bing": "Bing Search API",
    "google_cse": "Google CSE",
    "custom": "Custom provider",
}

WEB_SEARCH_PROVIDER_ALIASES = {
    "tavily": "tavily",
    "duckduckgo": "duckduckgo",
    "duck_duck_go": "duckduckgo",
    "ddg": "duckduckgo",
    "brave": "brave",
    "brave_search": "brave",
    "exa": "exa",
    "serpapi": "serpapi",
    "serp_api": "serpapi",
    "bing": "bing",
    "bing_search": "bing",
    "bing_search_api": "bing",
    "google": "google_cse",
    "google_cse": "google_cse",
    "google_custom_search": "google_cse",
    "custom": "custom",
    "custom_provider": "custom",
}

WEB_SEARCH_API_KEY_FIELDS = {
    "tavily": "tavily_api_key",
    "brave": "brave_api_key",
    "exa": "exa_api_key",
    "serpapi": "serpapi_api_key",
    "bing": "bing_api_key",
    "google_cse": "google_cse_api_key",
    "custom": "custom_web_search_api_key",
}

ANSWER_STYLE_CHOICES = ("Short", "Standard", "Detailed", "Research")
ANSWER_STYLE_ALIASES = {
    "short": "Short",
    "brief": "Short",
    "concise": "Short",
    "standard": "Standard",
    "normal": "Standard",
    "default": "Standard",
    "detailed": "Detailed",
    "detail": "Detailed",
    "long": "Detailed",
    "research": "Research",
    "research_grade": "Research",
    "academic": "Research",
}


def _load_dotenv() -> None:
    load_dotenv()
    load_dotenv(USER_CONFIG_PATH, override=True)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser()


def normalize_web_search_provider(provider: str) -> str:
    key = provider.strip().lower().replace("-", "_").replace(" ", "_")
    return WEB_SEARCH_PROVIDER_ALIASES.get(key, "custom" if key else "tavily")


def normalize_answer_style(style: str) -> str:
    key = style.strip().lower().replace("-", "_").replace(" ", "_")
    return ANSWER_STYLE_ALIASES.get(key, "Standard")


def save_user_config(settings: "AppSettings", path: Path = USER_CONFIG_PATH) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = _saved_config_values(settings)
    lines = [
        "# Verilume local configuration",
        "# This file can contain API keys. Do not commit it.",
    ]
    lines.extend(f"{key}={_env_value(value)}" for key, value in values.items())
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


def _saved_config_values(settings: "AppSettings") -> dict[str, str | int | float | bool]:
    return {
        "HF_TOKEN": settings.hf_token,
        "HF_LLM_MODEL": settings.hf_llm_model,
        "HF_PROVIDER": settings.hf_provider,
        "WEB_SEARCH_PROVIDER": settings.web_search_provider,
        "TAVILY_API_KEY": settings.tavily_api_key,
        "BRAVE_API_KEY": settings.brave_api_key,
        "EXA_API_KEY": settings.exa_api_key,
        "SERPAPI_API_KEY": settings.serpapi_api_key,
        "BING_API_KEY": settings.bing_api_key,
        "GOOGLE_CSE_API_KEY": settings.google_cse_api_key,
        "GOOGLE_CSE_ID": settings.google_cse_id,
        "CUSTOM_WEB_SEARCH_PROVIDER": settings.custom_web_search_provider,
        "CUSTOM_WEB_SEARCH_API_KEY": settings.custom_web_search_api_key,
        "CUSTOM_WEB_SEARCH_ENDPOINT": settings.custom_web_search_endpoint,
        "ENABLE_WEB_SEARCH": settings.enable_web_search,
        "WEB_SEARCH_MAX_RESULTS": settings.web_search_max_results,
        "WEB_SEARCH_TIMEOUT_SECONDS": settings.web_search_timeout_seconds,
        "SHOW_LOCAL_SOURCES": settings.show_local_sources,
        "ANSWER_STYLE": settings.answer_style,
        "RETRIEVER_K": settings.retriever_k,
        "RETRIEVAL_SCORE_THRESHOLD": settings.retrieval_score_threshold,
    }


def _env_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass(frozen=True, slots=True)
class AppSettings:
    app_title: str = "Verilume"
    app_icon: str = "\U0001f4da\U0001f50e"
    app_password: str = ""
    max_chat_messages: int = 30
    show_local_sources: bool = True
    answer_style: str = "Standard"

    docs_dir: Path = DATA_HOME / "documents"
    chroma_dir: Path = DATA_HOME / "chroma_db"
    collection_name: str = "verilume_docs"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "cpu"

    chunk_size: int = 1000
    chunk_overlap: int = 120
    reset_db: bool = False
    max_workers: int = 8
    batch_size: int = 128
    manifest_path: Path = DATA_HOME / "ingestion_manifest.json"

    hf_token: str = ""
    hf_llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_provider: str = "auto"
    hf_max_new_tokens: int = 700
    hf_temperature: float = 0.1
    hf_timeout_seconds: float = 90.0

    web_search_provider: str = "tavily"
    tavily_api_key: str = ""
    brave_api_key: str = ""
    exa_api_key: str = ""
    serpapi_api_key: str = ""
    bing_api_key: str = ""
    google_cse_api_key: str = ""
    google_cse_id: str = ""
    custom_web_search_provider: str = ""
    custom_web_search_api_key: str = ""
    custom_web_search_endpoint: str = ""
    enable_web_search: bool = True
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 20.0
    tavily_max_results: int = 5
    tavily_timeout_seconds: float = 20.0

    retriever_k: int = 5
    max_history_turns: int = 6
    enable_query_rewrite: bool = True
    retrieval_score_threshold: float = 0.35

    def __post_init__(self) -> None:
        object.__setattr__(self, "docs_dir", Path(self.docs_dir).expanduser())
        object.__setattr__(self, "chroma_dir", Path(self.chroma_dir).expanduser())
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).expanduser())
        object.__setattr__(self, "answer_style", normalize_answer_style(str(self.answer_style)))
        object.__setattr__(self, "chunk_size", max(100, int(self.chunk_size)))
        object.__setattr__(self, "chunk_overlap", max(0, min(int(self.chunk_overlap), self.chunk_size - 1)))
        object.__setattr__(self, "max_workers", max(1, int(self.max_workers)))
        object.__setattr__(self, "batch_size", max(1, int(self.batch_size)))
        object.__setattr__(self, "hf_max_new_tokens", max(32, int(self.hf_max_new_tokens)))
        object.__setattr__(self, "hf_temperature", max(0.0, float(self.hf_temperature)))
        object.__setattr__(self, "hf_timeout_seconds", max(5.0, float(self.hf_timeout_seconds)))
        object.__setattr__(
            self,
            "web_search_provider",
            normalize_web_search_provider(str(self.web_search_provider)),
        )
        object.__setattr__(self, "web_search_max_results", max(1, int(self.web_search_max_results)))
        object.__setattr__(
            self,
            "web_search_timeout_seconds",
            max(5.0, float(self.web_search_timeout_seconds)),
        )
        object.__setattr__(self, "tavily_max_results", max(1, int(self.tavily_max_results)))
        object.__setattr__(self, "tavily_timeout_seconds", max(5.0, float(self.tavily_timeout_seconds)))
        object.__setattr__(self, "retriever_k", max(1, int(self.retriever_k)))
        object.__setattr__(self, "max_history_turns", max(0, int(self.max_history_turns)))
        object.__setattr__(
            self,
            "retrieval_score_threshold",
            max(0.0, min(1.0, float(self.retrieval_score_threshold))),
        )

    @classmethod
    def from_env(cls) -> "AppSettings":
        _load_dotenv()
        defaults = cls()
        return cls(
            app_title=os.getenv("APP_TITLE", defaults.app_title),
            app_icon=os.getenv("APP_ICON", defaults.app_icon),
            app_password=os.getenv("APP_PASSWORD", defaults.app_password),
            max_chat_messages=_int("MAX_CHAT_MESSAGES", defaults.max_chat_messages),
            show_local_sources=_bool("SHOW_LOCAL_SOURCES", defaults.show_local_sources),
            answer_style=os.getenv("ANSWER_STYLE", defaults.answer_style),
            docs_dir=_path("DOCS_DIR", defaults.docs_dir),
            chroma_dir=_path("CHROMA_DIR", defaults.chroma_dir),
            collection_name=os.getenv("COLLECTION_NAME", defaults.collection_name),
            embed_model=os.getenv("EMBED_MODEL", defaults.embed_model),
            embed_device=os.getenv("EMBED_DEVICE", defaults.embed_device),
            chunk_size=_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            reset_db=_bool("RESET_DB", defaults.reset_db),
            max_workers=_int("MAX_WORKERS", defaults.max_workers),
            batch_size=_int("BATCH_SIZE", defaults.batch_size),
            manifest_path=_path("MANIFEST_PATH", defaults.manifest_path),
            hf_token=os.getenv("HF_TOKEN", defaults.hf_token),
            hf_llm_model=os.getenv("HF_LLM_MODEL", defaults.hf_llm_model),
            hf_provider=os.getenv("HF_PROVIDER", defaults.hf_provider),
            hf_max_new_tokens=_int("HF_MAX_NEW_TOKENS", defaults.hf_max_new_tokens),
            hf_temperature=_float("HF_TEMPERATURE", defaults.hf_temperature),
            hf_timeout_seconds=_float("HF_TIMEOUT_SECONDS", defaults.hf_timeout_seconds),
            web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", defaults.web_search_provider),
            tavily_api_key=os.getenv("TAVILY_API_KEY", defaults.tavily_api_key),
            brave_api_key=os.getenv("BRAVE_API_KEY", defaults.brave_api_key),
            exa_api_key=os.getenv("EXA_API_KEY", defaults.exa_api_key),
            serpapi_api_key=os.getenv("SERPAPI_API_KEY", defaults.serpapi_api_key),
            bing_api_key=os.getenv("BING_API_KEY", defaults.bing_api_key),
            google_cse_api_key=os.getenv("GOOGLE_CSE_API_KEY", defaults.google_cse_api_key),
            google_cse_id=os.getenv("GOOGLE_CSE_ID", defaults.google_cse_id),
            custom_web_search_provider=os.getenv(
                "CUSTOM_WEB_SEARCH_PROVIDER", defaults.custom_web_search_provider
            ),
            custom_web_search_api_key=os.getenv(
                "CUSTOM_WEB_SEARCH_API_KEY", defaults.custom_web_search_api_key
            ),
            custom_web_search_endpoint=os.getenv(
                "CUSTOM_WEB_SEARCH_ENDPOINT", defaults.custom_web_search_endpoint
            ),
            enable_web_search=_bool("ENABLE_WEB_SEARCH", defaults.enable_web_search),
            web_search_max_results=_int(
                "WEB_SEARCH_MAX_RESULTS",
                _int("TAVILY_MAX_RESULTS", defaults.web_search_max_results),
            ),
            web_search_timeout_seconds=_float(
                "WEB_SEARCH_TIMEOUT_SECONDS",
                _float("TAVILY_TIMEOUT_SECONDS", defaults.web_search_timeout_seconds),
            ),
            tavily_max_results=_int("TAVILY_MAX_RESULTS", defaults.tavily_max_results),
            tavily_timeout_seconds=_float("TAVILY_TIMEOUT_SECONDS", defaults.tavily_timeout_seconds),
            retriever_k=_int("RETRIEVER_K", defaults.retriever_k),
            max_history_turns=_int("MAX_HISTORY_TURNS", defaults.max_history_turns),
            enable_query_rewrite=_bool("ENABLE_QUERY_REWRITE", defaults.enable_query_rewrite),
            retrieval_score_threshold=_float(
                "RETRIEVAL_SCORE_THRESHOLD", defaults.retrieval_score_threshold
            ),
        )

    def with_overrides(self, **kwargs: Any) -> "AppSettings":
        clean = {key: value for key, value in kwargs.items() if value is not None}
        return replace(self, **clean)

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["hf_token"] = "***" if self.hf_token else ""
        for key in (
            "tavily_api_key",
            "brave_api_key",
            "exa_api_key",
            "serpapi_api_key",
            "bing_api_key",
            "google_cse_api_key",
            "custom_web_search_api_key",
        ):
            values[key] = "***" if values.get(key) else ""
        return values

    def web_search_provider_label(self) -> str:
        if self.web_search_provider == "custom" and self.custom_web_search_provider.strip():
            return self.custom_web_search_provider.strip()
        return WEB_SEARCH_PROVIDER_LABELS.get(self.web_search_provider, "Custom provider")

    def web_search_api_key(self) -> str:
        field = WEB_SEARCH_API_KEY_FIELDS.get(self.web_search_provider)
        if not field:
            return ""
        return str(getattr(self, field, "") or "").strip()

    def web_search_ready(self) -> bool:
        if not self.enable_web_search:
            return False
        provider = self.web_search_provider
        if provider == "duckduckgo":
            return True
        if provider == "google_cse":
            return bool(self.google_cse_api_key.strip() and self.google_cse_id.strip())
        if provider == "custom":
            return bool(self.custom_web_search_endpoint.strip())
        return bool(self.web_search_api_key())


DEFAULT_MODEL_CHOICES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "HuggingFaceH4/zephyr-7b-beta",
]


def ensure_app_dirs(settings: AppSettings) -> None:
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
