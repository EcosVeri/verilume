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


# ############################################################
# Generation backends
# ############################################################

GENERATION_BACKEND_LABELS = {
    "huggingface": "Hugging Face",
    "ollama": "Ollama",
}

GENERATION_BACKEND_ALIASES = {
    "hf": "huggingface",
    "hugging_face": "huggingface",
    "huggingface": "huggingface",
    "hugging-face": "huggingface",
    "ollama": "ollama",
    "local": "ollama",
    "local_ollama": "ollama",
}


DEFAULT_HF_MODEL_CHOICES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "HuggingFaceH4/zephyr-7b-beta",
]

DEFAULT_OLLAMA_MODEL_CHOICES = [
    "llama3.1:8b",
    "llama3.2:3b",
    "qwen2.5:7b",
    "mistral:7b",
    "gemma2:9b",
    "deepseek-r1:8b",
]


# ############################################################
# Web search providers
# ############################################################

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


# ############################################################
# Answer styles
# ############################################################

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


SEARCH_MODE_CHOICES = (
    "Auto",
    "Local Only",
    "Local + AI",
    "Local + AI + Web",
    "Web Only",
    "Research Mode",
)

SEARCH_MODE_ALIASES = {
    "auto": "Auto",
    "default": "Auto",
    "local": "Local Only",
    "local_only": "Local Only",
    "local files": "Local Only",
    "local_ai": "Local + AI",
    "local + ai": "Local + AI",
    "local_plus_ai": "Local + AI",
    "local_ai_web": "Local + AI + Web",
    "local + ai + web": "Local + AI + Web",
    "local_plus_ai_plus_web": "Local + AI + Web",
    "hybrid": "Local + AI + Web",
    "ai": "AI Only",
    "ai_only": "AI Only",
    "ai only": "AI Only",
    "model": "AI Only",
    "model_only": "AI Only",
    "web": "Web Only",
    "web_only": "Web Only",
    "web only": "Web Only",
    "research": "Research Mode",
    "research_mode": "Research Mode",
    "research mode": "Research Mode",
}

APPEARANCE_CHOICES = ("dark", "light")
APPEARANCE_ALIASES = {
    "dark": "dark",
    "night": "dark",
    "moon": "dark",
    "light": "light",
    "day": "light",
    "sun": "light",
}


@dataclass(frozen=True, slots=True)
class AnswerStyleProfile:
    name: str
    max_tokens: int
    temperature: float
    style_instruction: str
    verbosity: str


ANSWER_STYLE_PROFILES = {
    "Short": AnswerStyleProfile(
        name="Short",
        max_tokens=300,
        temperature=0.05,
        style_instruction="Be concise. Answer in 2-3 sentences unless citations need a short qualifier.",
        verbosity="brief",
    ),
    "Standard": AnswerStyleProfile(
        name="Standard",
        max_tokens=700,
        temperature=0.1,
        style_instruction="Provide a clear, complete answer with only the detail needed.",
        verbosity="standard",
    ),
    "Detailed": AnswerStyleProfile(
        name="Detailed",
        max_tokens=1200,
        temperature=0.15,
        style_instruction="Provide a comprehensive answer with useful context and examples.",
        verbosity="detailed",
    ),
    "Research": AnswerStyleProfile(
        name="Research",
        max_tokens=2000,
        temperature=0.2,
        style_instruction="Provide a careful research-style answer with thorough source comparison.",
        verbosity="detailed",
    ),
}


DEFAULT_EVIDENCE_OFFICIAL_DOMAINS = (
    ".gov",
    ".edu",
    "ac.",
    "apnews.com",
    "arxiv.org",
    "bbc.com",
    "bbc.co.uk",
    "doi.org",
    "europa.eu",
    "gouvernement.lu",
    "nature.com",
    "oecd.org",
    "public.lu",
    "reuters.com",
    "science.org",
    "who.int",
    "worldbank.org",
)


# ############################################################
# Environment helpers
# ############################################################


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


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)

    if value is None:
        return default

    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or default


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)

    if not value:
        return default

    return Path(value).expanduser()


def _env_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')

    return f'"{escaped}"'


# ############################################################
# Normalizers
# ############################################################


def normalize_generation_backend(backend: str) -> str:
    key = backend.strip().lower().replace("-", "_").replace(" ", "_")
    return GENERATION_BACKEND_ALIASES.get(key, "huggingface")


def normalize_hf_provider(provider: str) -> str:
    key = provider.strip().lower().replace("_", "-").replace(" ", "-")
    return key or "auto"


def normalize_web_search_provider(provider: str) -> str:
    key = provider.strip().lower().replace("-", "_").replace(" ", "_")
    return WEB_SEARCH_PROVIDER_ALIASES.get(
        key,
        "custom" if key else "tavily",
    )


def normalize_answer_style(style: str) -> str:
    key = style.strip().lower().replace("-", "_").replace(" ", "_")
    return ANSWER_STYLE_ALIASES.get(key, "Standard")


def normalize_search_mode(mode: str) -> str:
    raw = (mode or "").strip().lower()
    key = raw.replace("-", "_").replace(" ", "_")
    return SEARCH_MODE_ALIASES.get(key, SEARCH_MODE_ALIASES.get(raw, "Auto"))


def normalize_appearance(appearance: str) -> str:
    key = (appearance or "").strip().lower().replace("-", "_").replace(" ", "_")
    return APPEARANCE_ALIASES.get(key, "dark")


# ############################################################
# Save local user config
# ############################################################


def save_user_config(
    settings: "AppSettings",
    path: Path = USER_CONFIG_PATH,
) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    values = _saved_config_values(settings)

    lines = [
        "# Verilume local configuration",
        "# This file can contain API keys. Do not commit it.",
    ]

    lines.extend(f"{key}={_env_value(value)}" for key, value in values.items())

    path.write_text(
        "\n".join(lines).strip() + "\n",
        encoding="utf-8",
    )

    return path


def _saved_config_values(
    settings: "AppSettings",
) -> dict[str, str | int | float | bool]:
    return {
        # Generation backend
        "GENERATION_BACKEND": settings.generation_backend,
        # Hugging Face
        "HF_TOKEN": settings.hf_token,
        "HF_LLM_MODEL": settings.hf_llm_model,
        "HF_PROVIDER": settings.hf_provider,
        "HF_MAX_NEW_TOKENS": settings.hf_max_new_tokens,
        "HF_TEMPERATURE": settings.hf_temperature,
        "HF_TIMEOUT_SECONDS": settings.hf_timeout_seconds,
        # Ollama
        "OLLAMA_BASE_URL": settings.ollama_base_url,
        "OLLAMA_MODEL": settings.ollama_model,
        "OLLAMA_TEMPERATURE": settings.ollama_temperature,
        "OLLAMA_NUM_PREDICT": settings.ollama_num_predict,
        "OLLAMA_TIMEOUT_SECONDS": settings.ollama_timeout_seconds,
        # Web search
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
        "WEB_SEARCH_CACHE_TTL_SECONDS": settings.web_search_cache_ttl_seconds,
        "WEB_SEARCH_MAX_WORKERS": settings.web_search_max_workers,
        "ENABLE_AGGRESSIVE_WEB_FALLBACK": settings.enable_aggressive_web_fallback,
        "WEB_SEARCH_FALLBACK_MAX_RESULTS": settings.web_search_fallback_max_results,
        # Semantic cache
        "SEMANTIC_CACHE_ENABLED": settings.semantic_cache_enabled,
        "SEMANTIC_CACHE_PATH": settings.semantic_cache_path,
        "SEMANTIC_CACHE_STABLE_TTL_SECONDS": settings.semantic_cache_stable_ttl_seconds,
        "SEMANTIC_CACHE_CURRENT_TTL_SECONDS": settings.semantic_cache_current_ttl_seconds,
        "SEMANTIC_CACHE_ENTITY_TTL_SECONDS": settings.semantic_cache_entity_ttl_seconds,
        "SEMANTIC_CACHE_LOCAL_TTL_SECONDS": settings.semantic_cache_local_ttl_seconds,
        "TABLE_STORE_DIR": settings.table_store_dir,
        "KNOWLEDGE_GRAPH_PATH": settings.knowledge_graph_path,
        "ENABLE_GRAPHRAG": settings.enable_graphrag,
        "MULTIMODAL_STORE_PATH": settings.multimodal_store_path,
        "FORMULA_STORE_PATH": settings.formula_store_path,
        "OCR_BLOCK_STORE_PATH": settings.ocr_block_store_path,
        "STRUCTURED_DOCUMENT_STORE_PATH": settings.structured_document_store_path,
        "BENCHMARK_MODE": settings.benchmark_mode,
        # Retrieval and UI
        "VERILUME_APPEARANCE": settings.appearance,
        "SHOW_LOCAL_SOURCES": settings.show_local_sources,
        "ANSWER_STYLE": settings.answer_style,
        "SEARCH_MODE": settings.search_mode,
        "RETRIEVER_K": settings.retriever_k,
        "RETRIEVAL_SCORE_THRESHOLD": settings.retrieval_score_threshold,
        "ENABLE_QUERY_REWRITE": settings.enable_query_rewrite,
        "QUERY_REWRITE_MIN_HISTORY": settings.query_rewrite_min_history,
        "QUERY_REWRITE_SIMILARITY_THRESHOLD": settings.query_rewrite_similarity_threshold,
        "RETRIEVAL_MODE": settings.retrieval_mode,
        "RRF_CONSTANT": settings.rrf_constant,
        "RRF_DENSE_WEIGHT": settings.rrf_dense_weight,
        "RRF_LEXICAL_WEIGHT": settings.rrf_lexical_weight,
        "RRF_SEMANTIC_BOOST": settings.rrf_semantic_boost,
        "RRF_SCORE_SCALE": settings.rrf_score_scale,
        "ENABLE_RERANKER": settings.enable_reranker,
        "RERANKER_MODEL": settings.reranker_model,
        "RERANKER_DEVICE": settings.reranker_device,
        "RERANKER_TOP_K": settings.reranker_top_k,
        "RERANK_SEMANTIC_WEIGHT": settings.rerank_semantic_weight,
        "RERANK_LEXICAL_WEIGHT": settings.rerank_lexical_weight,
        "RERANK_PHRASE_BONUS_FULL": settings.rerank_phrase_bonus_full,
        "RERANK_PHRASE_BONUS_PARTIAL": settings.rerank_phrase_bonus_partial,
        "RERANK_MISMATCH_PENALTY": settings.rerank_mismatch_penalty,
        "RERANK_MISMATCH_THRESHOLD": settings.rerank_mismatch_threshold,
        "STRONG_LOCAL_SCORE_THRESHOLD": settings.strong_local_score_threshold,
        "STRONG_LOCAL_MIN_SOURCES": settings.strong_local_min_sources,
        "ANSWER_VERIFICATION_MODE": settings.answer_verification_mode,
        "ANSWER_VERIFICATION_MIN_OVERLAP": settings.answer_verification_min_overlap,
        "FORMULA_DETECTION_THRESHOLD": settings.formula_detection_threshold,
    }


# ############################################################
# Main settings object
# ############################################################


@dataclass(frozen=True, slots=True)
class AppSettings:
    # App
    app_title: str = "Verilume"
    app_icon: str = "📚🔎"
    app_password: str = ""
    max_chat_messages: int = 30
    show_local_sources: bool = True
    answer_style: str = "Standard"
    search_mode: str = "Auto"
    appearance: str = "dark"

    # Storage
    docs_dir: Path = DATA_HOME / "documents"
    chroma_dir: Path = DATA_HOME / "chroma_db"
    collection_name: str = "verilume_docs"
    semantic_cache_enabled: bool = True
    semantic_cache_path: Path = DATA_HOME / "semantic_cache.json"
    semantic_cache_stable_ttl_seconds: int = 604800
    semantic_cache_current_ttl_seconds: int = 3600
    semantic_cache_entity_ttl_seconds: int = 604800
    semantic_cache_local_ttl_seconds: int = 0
    table_store_dir: Path = DATA_HOME / "tables"
    knowledge_graph_path: Path = DATA_HOME / "knowledge_graph.sqlite"
    enable_graphrag: bool = True
    multimodal_store_path: Path = DATA_HOME / "multimodal.sqlite"
    formula_store_path: Path = DATA_HOME / "formulas.sqlite"
    ocr_block_store_path: Path = DATA_HOME / "ocr_blocks.sqlite"
    structured_document_store_path: Path = DATA_HOME / "structured_documents.sqlite"
    benchmark_mode: bool = False

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "cpu"
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = DATA_HOME / "embedding_cache"

    # Ingestion
    chunk_size: int = 1000
    chunk_overlap: int = 120
    chunk_strategy: str = "semantic"
    reset_db: bool = False
    max_workers: int = 8
    batch_size: int = 128
    manifest_path: Path = DATA_HOME / "ingestion_manifest.json"
    process_parse_documents: bool = True
    metadata_abstract_pattern: str = (
        r"(?is)\babstract\b\s*[:.\-]?\s*(?P<abstract>.{80,1600}?)"
        r"(?:\n\s*(?:keywords?|1\.?\s+introduction|introduction|references)\b)"
    )
    metadata_abstract_limit: int = 900
    metadata_keywords_pattern: str = r"(?im)^\s*key\s*words?\s*[:\-]\s*(?P<keywords>.+)$"
    metadata_keywords_limit: int = 400

    # Generation backend
    generation_backend: str = "huggingface"

    # Hugging Face generation
    hf_token: str = ""
    hf_llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_provider: str = "auto"
    hf_max_new_tokens: int = 700
    hf_temperature: float = 0.1
    hf_timeout_seconds: float = 90.0

    # Ollama generation
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_temperature: float = 0.1
    ollama_num_predict: int = 700
    ollama_timeout_seconds: float = 120.0

    # Web search
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
    web_search_timeout_seconds: float = 12.0
    web_search_cache_ttl_seconds: float = 600.0
    web_search_max_workers: int = 3
    enable_aggressive_web_fallback: bool = True
    web_search_fallback_max_results: int = 12

    # Backward-compatible Tavily settings
    tavily_max_results: int = 5
    tavily_timeout_seconds: float = 12.0

    # Retrieval
    retriever_k: int = 5
    max_history_turns: int = 6
    enable_query_rewrite: bool = True
    query_rewrite_min_history: int = 1
    query_rewrite_similarity_threshold: float = 0.92
    retrieval_score_threshold: float = 0.35
    retrieval_mode: str = "hybrid"
    rrf_constant: int = 60
    rrf_dense_weight: float = 1.0
    rrf_lexical_weight: float = 1.0
    rrf_semantic_boost: float = 0.25
    rrf_score_scale: float = 28.0
    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    reranker_top_k: int = 8
    rerank_semantic_weight: float = 0.52
    rerank_lexical_weight: float = 0.48
    rerank_phrase_bonus_full: float = 0.28
    rerank_phrase_bonus_partial: float = 0.16
    rerank_mismatch_penalty: float = 0.55
    rerank_mismatch_threshold: float = 0.72
    rerank_single_match_penalty: float = 0.78
    rerank_single_match_threshold: float = 0.78
    strong_local_score_threshold: float = 0.72
    strong_local_min_sources: int = 1
    answer_verification_mode: str = "heuristic"
    answer_verification_min_overlap: float = 0.18
    formula_detection_threshold: float = 0.55
    evidence_official_domains: tuple[str, ...] = DEFAULT_EVIDENCE_OFFICIAL_DOMAINS
    evidence_authority_boost: float = 0.35
    evidence_freshness_boost: float = 0.1
    evidence_freshness_decay_days: int = 365
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "docs_dir",
            Path(self.docs_dir).expanduser(),
        )

        object.__setattr__(
            self,
            "chroma_dir",
            Path(self.chroma_dir).expanduser(),
        )

        object.__setattr__(
            self,
            "manifest_path",
            Path(self.manifest_path).expanduser(),
        )

        object.__setattr__(
            self,
            "semantic_cache_path",
            Path(self.semantic_cache_path).expanduser(),
        )

        object.__setattr__(
            self,
            "embedding_cache_dir",
            Path(self.embedding_cache_dir).expanduser(),
        )

        object.__setattr__(
            self,
            "table_store_dir",
            Path(self.table_store_dir).expanduser(),
        )

        object.__setattr__(
            self,
            "knowledge_graph_path",
            Path(self.knowledge_graph_path).expanduser(),
        )

        object.__setattr__(
            self,
            "multimodal_store_path",
            Path(self.multimodal_store_path).expanduser(),
        )
        object.__setattr__(
            self,
            "formula_store_path",
            Path(self.formula_store_path).expanduser(),
        )
        object.__setattr__(
            self,
            "ocr_block_store_path",
            Path(self.ocr_block_store_path).expanduser(),
        )
        object.__setattr__(
            self,
            "structured_document_store_path",
            Path(self.structured_document_store_path).expanduser(),
        )

        object.__setattr__(
            self,
            "answer_style",
            normalize_answer_style(str(self.answer_style)),
        )
        object.__setattr__(
            self,
            "search_mode",
            normalize_search_mode(str(self.search_mode)),
        )
        object.__setattr__(
            self,
            "appearance",
            normalize_appearance(str(self.appearance)),
        )

        object.__setattr__(
            self,
            "generation_backend",
            normalize_generation_backend(str(self.generation_backend)),
        )

        object.__setattr__(
            self,
            "hf_provider",
            normalize_hf_provider(str(self.hf_provider)),
        )

        object.__setattr__(
            self,
            "chunk_size",
            max(100, int(self.chunk_size)),
        )

        object.__setattr__(
            self,
            "chunk_overlap",
            max(0, min(int(self.chunk_overlap), self.chunk_size - 1)),
        )

        object.__setattr__(
            self,
            "chunk_strategy",
            str(self.chunk_strategy or "semantic").strip().lower(),
        )

        object.__setattr__(
            self,
            "max_workers",
            max(1, int(self.max_workers)),
        )

        object.__setattr__(
            self,
            "batch_size",
            max(1, int(self.batch_size)),
        )

        object.__setattr__(
            self,
            "metadata_abstract_limit",
            max(80, int(self.metadata_abstract_limit)),
        )

        object.__setattr__(
            self,
            "metadata_keywords_limit",
            max(20, int(self.metadata_keywords_limit)),
        )

        object.__setattr__(
            self,
            "hf_max_new_tokens",
            max(32, int(self.hf_max_new_tokens)),
        )

        object.__setattr__(
            self,
            "hf_temperature",
            max(0.0, float(self.hf_temperature)),
        )

        object.__setattr__(
            self,
            "hf_timeout_seconds",
            max(5.0, float(self.hf_timeout_seconds)),
        )

        object.__setattr__(
            self,
            "ollama_temperature",
            max(0.0, float(self.ollama_temperature)),
        )

        object.__setattr__(
            self,
            "ollama_num_predict",
            max(32, int(self.ollama_num_predict)),
        )

        object.__setattr__(
            self,
            "ollama_timeout_seconds",
            max(5.0, float(self.ollama_timeout_seconds)),
        )

        object.__setattr__(
            self,
            "web_search_provider",
            normalize_web_search_provider(str(self.web_search_provider)),
        )

        object.__setattr__(
            self,
            "web_search_max_results",
            max(1, int(self.web_search_max_results)),
        )

        object.__setattr__(
            self,
            "web_search_timeout_seconds",
            max(5.0, float(self.web_search_timeout_seconds)),
        )

        object.__setattr__(
            self,
            "web_search_cache_ttl_seconds",
            max(0.0, float(self.web_search_cache_ttl_seconds)),
        )

        object.__setattr__(
            self,
            "web_search_max_workers",
            max(1, int(self.web_search_max_workers)),
        )
        object.__setattr__(
            self,
            "web_search_fallback_max_results",
            max(1, int(self.web_search_fallback_max_results)),
        )
        object.__setattr__(
            self,
            "semantic_cache_stable_ttl_seconds",
            max(0, int(self.semantic_cache_stable_ttl_seconds)),
        )
        object.__setattr__(
            self,
            "semantic_cache_current_ttl_seconds",
            max(0, int(self.semantic_cache_current_ttl_seconds)),
        )
        object.__setattr__(
            self,
            "semantic_cache_entity_ttl_seconds",
            max(0, int(self.semantic_cache_entity_ttl_seconds)),
        )
        object.__setattr__(
            self,
            "semantic_cache_local_ttl_seconds",
            max(0, int(self.semantic_cache_local_ttl_seconds)),
        )

        object.__setattr__(
            self,
            "tavily_max_results",
            max(1, int(self.tavily_max_results)),
        )

        object.__setattr__(
            self,
            "tavily_timeout_seconds",
            max(5.0, float(self.tavily_timeout_seconds)),
        )

        object.__setattr__(
            self,
            "retriever_k",
            max(1, int(self.retriever_k)),
        )

        object.__setattr__(
            self,
            "max_history_turns",
            max(0, int(self.max_history_turns)),
        )

        object.__setattr__(
            self,
            "retrieval_score_threshold",
            max(0.0, min(1.0, float(self.retrieval_score_threshold))),
        )
        object.__setattr__(
            self,
            "query_rewrite_min_history",
            max(0, int(self.query_rewrite_min_history)),
        )
        object.__setattr__(
            self,
            "query_rewrite_similarity_threshold",
            max(0.0, min(1.0, float(self.query_rewrite_similarity_threshold))),
        )
        object.__setattr__(
            self,
            "retrieval_mode",
            str(self.retrieval_mode or "hybrid").strip().lower(),
        )
        object.__setattr__(
            self,
            "rrf_constant",
            max(1, int(self.rrf_constant)),
        )
        object.__setattr__(
            self,
            "rrf_dense_weight",
            max(0.0, float(self.rrf_dense_weight)),
        )
        object.__setattr__(
            self,
            "rrf_lexical_weight",
            max(0.0, float(self.rrf_lexical_weight)),
        )
        object.__setattr__(
            self,
            "rrf_semantic_boost",
            max(0.0, float(self.rrf_semantic_boost)),
        )
        object.__setattr__(
            self,
            "rrf_score_scale",
            max(0.0, float(self.rrf_score_scale)),
        )
        object.__setattr__(
            self,
            "reranker_top_k",
            max(1, int(self.reranker_top_k)),
        )
        object.__setattr__(
            self,
            "rerank_semantic_weight",
            max(0.0, float(self.rerank_semantic_weight)),
        )
        object.__setattr__(
            self,
            "rerank_lexical_weight",
            max(0.0, float(self.rerank_lexical_weight)),
        )
        object.__setattr__(
            self,
            "rerank_phrase_bonus_full",
            max(0.0, float(self.rerank_phrase_bonus_full)),
        )
        object.__setattr__(
            self,
            "rerank_phrase_bonus_partial",
            max(0.0, float(self.rerank_phrase_bonus_partial)),
        )
        object.__setattr__(
            self,
            "rerank_mismatch_penalty",
            max(0.0, min(1.0, float(self.rerank_mismatch_penalty))),
        )
        object.__setattr__(
            self,
            "rerank_mismatch_threshold",
            max(0.0, min(1.0, float(self.rerank_mismatch_threshold))),
        )
        object.__setattr__(
            self,
            "rerank_single_match_penalty",
            max(0.0, min(1.0, float(self.rerank_single_match_penalty))),
        )
        object.__setattr__(
            self,
            "rerank_single_match_threshold",
            max(0.0, min(1.0, float(self.rerank_single_match_threshold))),
        )
        object.__setattr__(
            self,
            "strong_local_score_threshold",
            max(0.0, min(1.0, float(self.strong_local_score_threshold))),
        )
        object.__setattr__(
            self,
            "strong_local_min_sources",
            max(1, int(self.strong_local_min_sources)),
        )
        object.__setattr__(
            self,
            "answer_verification_mode",
            str(self.answer_verification_mode or "heuristic").strip().lower(),
        )
        object.__setattr__(
            self,
            "answer_verification_min_overlap",
            max(0.0, min(1.0, float(self.answer_verification_min_overlap))),
        )
        object.__setattr__(
            self,
            "formula_detection_threshold",
            max(0.1, min(0.95, float(self.formula_detection_threshold))),
        )
        object.__setattr__(
            self,
            "evidence_official_domains",
            tuple(str(item).strip().lower() for item in self.evidence_official_domains if str(item).strip()),
        )
        object.__setattr__(
            self,
            "evidence_authority_boost",
            max(0.0, float(self.evidence_authority_boost)),
        )
        object.__setattr__(
            self,
            "evidence_freshness_boost",
            max(0.0, float(self.evidence_freshness_boost)),
        )
        object.__setattr__(
            self,
            "evidence_freshness_decay_days",
            max(1, int(self.evidence_freshness_decay_days)),
        )
        object.__setattr__(
            self,
            "confidence_high_threshold",
            max(0.0, min(1.0, float(self.confidence_high_threshold))),
        )
        object.__setattr__(
            self,
            "confidence_medium_threshold",
            max(0.0, min(1.0, float(self.confidence_medium_threshold))),
        )

    @classmethod
    def from_env(cls) -> "AppSettings":
        _load_dotenv()
        defaults = cls()

        return cls(
            # App
            app_title=os.getenv("APP_TITLE", defaults.app_title),
            app_icon=os.getenv("APP_ICON", defaults.app_icon),
            app_password=os.getenv("APP_PASSWORD", defaults.app_password),
            max_chat_messages=_int(
                "MAX_CHAT_MESSAGES",
                defaults.max_chat_messages,
            ),
            show_local_sources=_bool(
                "SHOW_LOCAL_SOURCES",
                defaults.show_local_sources,
            ),
            answer_style=os.getenv(
                "ANSWER_STYLE",
                defaults.answer_style,
            ),
            search_mode=os.getenv(
                "SEARCH_MODE",
                defaults.search_mode,
            ),
            appearance=os.getenv(
                "VERILUME_APPEARANCE",
                defaults.appearance,
            ),
            # Storage
            docs_dir=_path("DOCS_DIR", defaults.docs_dir),
            chroma_dir=_path("CHROMA_DIR", defaults.chroma_dir),
            collection_name=os.getenv(
                "COLLECTION_NAME",
                defaults.collection_name,
            ),
            semantic_cache_enabled=_bool(
                "SEMANTIC_CACHE_ENABLED",
                defaults.semantic_cache_enabled,
            ),
            semantic_cache_path=_path(
                "SEMANTIC_CACHE_PATH",
                defaults.semantic_cache_path,
            ),
            semantic_cache_stable_ttl_seconds=_int(
                "SEMANTIC_CACHE_STABLE_TTL_SECONDS",
                defaults.semantic_cache_stable_ttl_seconds,
            ),
            semantic_cache_current_ttl_seconds=_int(
                "SEMANTIC_CACHE_CURRENT_TTL_SECONDS",
                defaults.semantic_cache_current_ttl_seconds,
            ),
            semantic_cache_entity_ttl_seconds=_int(
                "SEMANTIC_CACHE_ENTITY_TTL_SECONDS",
                defaults.semantic_cache_entity_ttl_seconds,
            ),
            semantic_cache_local_ttl_seconds=_int(
                "SEMANTIC_CACHE_LOCAL_TTL_SECONDS",
                defaults.semantic_cache_local_ttl_seconds,
            ),
            table_store_dir=_path(
                "TABLE_STORE_DIR",
                defaults.table_store_dir,
            ),
            knowledge_graph_path=_path(
                "KNOWLEDGE_GRAPH_PATH",
                defaults.knowledge_graph_path,
            ),
            enable_graphrag=_bool(
                "ENABLE_GRAPHRAG",
                defaults.enable_graphrag,
            ),
            multimodal_store_path=_path(
                "MULTIMODAL_STORE_PATH",
                defaults.multimodal_store_path,
            ),
            formula_store_path=_path(
                "FORMULA_STORE_PATH",
                defaults.formula_store_path,
            ),
            ocr_block_store_path=_path(
                "OCR_BLOCK_STORE_PATH",
                defaults.ocr_block_store_path,
            ),
            structured_document_store_path=_path(
                "STRUCTURED_DOCUMENT_STORE_PATH",
                defaults.structured_document_store_path,
            ),
            benchmark_mode=_bool(
                "BENCHMARK_MODE",
                defaults.benchmark_mode,
            ),
            # Embeddings
            embed_model=os.getenv("EMBED_MODEL", defaults.embed_model),
            embed_device=os.getenv("EMBED_DEVICE", defaults.embed_device),
            embedding_cache_enabled=_bool(
                "EMBEDDING_CACHE_ENABLED",
                defaults.embedding_cache_enabled,
            ),
            embedding_cache_dir=_path(
                "EMBEDDING_CACHE_DIR",
                defaults.embedding_cache_dir,
            ),
            # Ingestion
            chunk_size=_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            chunk_strategy=os.getenv("CHUNK_STRATEGY", defaults.chunk_strategy),
            reset_db=_bool("RESET_DB", defaults.reset_db),
            max_workers=_int("MAX_WORKERS", defaults.max_workers),
            batch_size=_int("BATCH_SIZE", defaults.batch_size),
            manifest_path=_path(
                "MANIFEST_PATH",
                defaults.manifest_path,
            ),
            process_parse_documents=_bool(
                "PROCESS_PARSE_DOCUMENTS",
                defaults.process_parse_documents,
            ),
            metadata_abstract_pattern=os.getenv(
                "METADATA_ABSTRACT_PATTERN",
                defaults.metadata_abstract_pattern,
            ),
            metadata_abstract_limit=_int(
                "METADATA_ABSTRACT_LIMIT",
                defaults.metadata_abstract_limit,
            ),
            metadata_keywords_pattern=os.getenv(
                "METADATA_KEYWORDS_PATTERN",
                defaults.metadata_keywords_pattern,
            ),
            metadata_keywords_limit=_int(
                "METADATA_KEYWORDS_LIMIT",
                defaults.metadata_keywords_limit,
            ),
            # Generation backend
            generation_backend=os.getenv(
                "GENERATION_BACKEND",
                defaults.generation_backend,
            ),
            # Hugging Face
            hf_token=os.getenv("HF_TOKEN", defaults.hf_token),
            hf_llm_model=os.getenv(
                "HF_LLM_MODEL",
                defaults.hf_llm_model,
            ),
            hf_provider=os.getenv("HF_PROVIDER", defaults.hf_provider),
            hf_max_new_tokens=_int(
                "HF_MAX_NEW_TOKENS",
                defaults.hf_max_new_tokens,
            ),
            hf_temperature=_float(
                "HF_TEMPERATURE",
                defaults.hf_temperature,
            ),
            hf_timeout_seconds=_float(
                "HF_TIMEOUT_SECONDS",
                defaults.hf_timeout_seconds,
            ),
            # Ollama
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL",
                defaults.ollama_base_url,
            ),
            ollama_model=os.getenv(
                "OLLAMA_MODEL",
                defaults.ollama_model,
            ),
            ollama_temperature=_float(
                "OLLAMA_TEMPERATURE",
                defaults.ollama_temperature,
            ),
            ollama_num_predict=_int(
                "OLLAMA_NUM_PREDICT",
                defaults.ollama_num_predict,
            ),
            ollama_timeout_seconds=_float(
                "OLLAMA_TIMEOUT_SECONDS",
                defaults.ollama_timeout_seconds,
            ),
            # Web search
            web_search_provider=os.getenv(
                "WEB_SEARCH_PROVIDER",
                defaults.web_search_provider,
            ),
            tavily_api_key=os.getenv(
                "TAVILY_API_KEY",
                defaults.tavily_api_key,
            ),
            brave_api_key=os.getenv(
                "BRAVE_API_KEY",
                defaults.brave_api_key,
            ),
            exa_api_key=os.getenv(
                "EXA_API_KEY",
                defaults.exa_api_key,
            ),
            serpapi_api_key=os.getenv(
                "SERPAPI_API_KEY",
                defaults.serpapi_api_key,
            ),
            bing_api_key=os.getenv(
                "BING_API_KEY",
                defaults.bing_api_key,
            ),
            google_cse_api_key=os.getenv(
                "GOOGLE_CSE_API_KEY",
                defaults.google_cse_api_key,
            ),
            google_cse_id=os.getenv(
                "GOOGLE_CSE_ID",
                defaults.google_cse_id,
            ),
            custom_web_search_provider=os.getenv(
                "CUSTOM_WEB_SEARCH_PROVIDER",
                defaults.custom_web_search_provider,
            ),
            custom_web_search_api_key=os.getenv(
                "CUSTOM_WEB_SEARCH_API_KEY",
                defaults.custom_web_search_api_key,
            ),
            custom_web_search_endpoint=os.getenv(
                "CUSTOM_WEB_SEARCH_ENDPOINT",
                defaults.custom_web_search_endpoint,
            ),
            enable_web_search=_bool(
                "ENABLE_WEB_SEARCH",
                defaults.enable_web_search,
            ),
            web_search_max_results=_int(
                "WEB_SEARCH_MAX_RESULTS",
                _int("TAVILY_MAX_RESULTS", defaults.web_search_max_results),
            ),
            web_search_timeout_seconds=_float(
                "WEB_SEARCH_TIMEOUT_SECONDS",
                _float(
                    "TAVILY_TIMEOUT_SECONDS",
                    defaults.web_search_timeout_seconds,
                ),
            ),
            web_search_cache_ttl_seconds=_float(
                "WEB_SEARCH_CACHE_TTL_SECONDS",
                defaults.web_search_cache_ttl_seconds,
            ),
            web_search_max_workers=_int(
                "WEB_SEARCH_MAX_WORKERS",
                defaults.web_search_max_workers,
            ),
            enable_aggressive_web_fallback=_bool(
                "ENABLE_AGGRESSIVE_WEB_FALLBACK",
                defaults.enable_aggressive_web_fallback,
            ),
            web_search_fallback_max_results=_int(
                "WEB_SEARCH_FALLBACK_MAX_RESULTS",
                defaults.web_search_fallback_max_results,
            ),
            tavily_max_results=_int(
                "TAVILY_MAX_RESULTS",
                defaults.tavily_max_results,
            ),
            tavily_timeout_seconds=_float(
                "TAVILY_TIMEOUT_SECONDS",
                defaults.tavily_timeout_seconds,
            ),
            # Retrieval
            retriever_k=_int("RETRIEVER_K", defaults.retriever_k),
            max_history_turns=_int(
                "MAX_HISTORY_TURNS",
                defaults.max_history_turns,
            ),
            enable_query_rewrite=_bool(
                "ENABLE_QUERY_REWRITE",
                defaults.enable_query_rewrite,
            ),
            query_rewrite_min_history=_int(
                "QUERY_REWRITE_MIN_HISTORY",
                defaults.query_rewrite_min_history,
            ),
            query_rewrite_similarity_threshold=_float(
                "QUERY_REWRITE_SIMILARITY_THRESHOLD",
                defaults.query_rewrite_similarity_threshold,
            ),
            retrieval_score_threshold=_float(
                "RETRIEVAL_SCORE_THRESHOLD",
                defaults.retrieval_score_threshold,
            ),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", defaults.retrieval_mode),
            rrf_constant=_int("RRF_CONSTANT", defaults.rrf_constant),
            rrf_dense_weight=_float("RRF_DENSE_WEIGHT", defaults.rrf_dense_weight),
            rrf_lexical_weight=_float(
                "RRF_LEXICAL_WEIGHT",
                defaults.rrf_lexical_weight,
            ),
            rrf_semantic_boost=_float(
                "RRF_SEMANTIC_BOOST",
                defaults.rrf_semantic_boost,
            ),
            rrf_score_scale=_float("RRF_SCORE_SCALE", defaults.rrf_score_scale),
            enable_reranker=_bool("ENABLE_RERANKER", defaults.enable_reranker),
            reranker_model=os.getenv("RERANKER_MODEL", defaults.reranker_model),
            reranker_device=os.getenv("RERANKER_DEVICE", defaults.reranker_device),
            reranker_top_k=_int("RERANKER_TOP_K", defaults.reranker_top_k),
            rerank_semantic_weight=_float(
                "RERANK_SEMANTIC_WEIGHT",
                defaults.rerank_semantic_weight,
            ),
            rerank_lexical_weight=_float(
                "RERANK_LEXICAL_WEIGHT",
                defaults.rerank_lexical_weight,
            ),
            rerank_phrase_bonus_full=_float(
                "RERANK_PHRASE_BONUS_FULL",
                defaults.rerank_phrase_bonus_full,
            ),
            rerank_phrase_bonus_partial=_float(
                "RERANK_PHRASE_BONUS_PARTIAL",
                defaults.rerank_phrase_bonus_partial,
            ),
            rerank_mismatch_penalty=_float(
                "RERANK_MISMATCH_PENALTY",
                defaults.rerank_mismatch_penalty,
            ),
            rerank_mismatch_threshold=_float(
                "RERANK_MISMATCH_THRESHOLD",
                defaults.rerank_mismatch_threshold,
            ),
            rerank_single_match_penalty=_float(
                "RERANK_SINGLE_MATCH_PENALTY",
                defaults.rerank_single_match_penalty,
            ),
            rerank_single_match_threshold=_float(
                "RERANK_SINGLE_MATCH_THRESHOLD",
                defaults.rerank_single_match_threshold,
            ),
            strong_local_score_threshold=_float(
                "STRONG_LOCAL_SCORE_THRESHOLD",
                defaults.strong_local_score_threshold,
            ),
            strong_local_min_sources=_int(
                "STRONG_LOCAL_MIN_SOURCES",
                defaults.strong_local_min_sources,
            ),
            answer_verification_mode=os.getenv(
                "ANSWER_VERIFICATION_MODE",
                defaults.answer_verification_mode,
            ),
            answer_verification_min_overlap=_float(
                "ANSWER_VERIFICATION_MIN_OVERLAP",
                defaults.answer_verification_min_overlap,
            ),
            formula_detection_threshold=_float(
                "FORMULA_DETECTION_THRESHOLD",
                defaults.formula_detection_threshold,
            ),
            evidence_official_domains=_csv(
                "EVIDENCE_OFFICIAL_DOMAINS",
                defaults.evidence_official_domains,
            ),
            evidence_authority_boost=_float(
                "EVIDENCE_AUTHORITY_BOOST",
                defaults.evidence_authority_boost,
            ),
            evidence_freshness_boost=_float(
                "EVIDENCE_FRESHNESS_BOOST",
                defaults.evidence_freshness_boost,
            ),
            evidence_freshness_decay_days=_int(
                "EVIDENCE_FRESHNESS_DECAY_DAYS",
                defaults.evidence_freshness_decay_days,
            ),
            confidence_high_threshold=_float(
                "CONFIDENCE_HIGH_THRESHOLD",
                defaults.confidence_high_threshold,
            ),
            confidence_medium_threshold=_float(
                "CONFIDENCE_MEDIUM_THRESHOLD",
                defaults.confidence_medium_threshold,
            ),
        )

    def with_overrides(self, **kwargs: Any) -> "AppSettings":
        clean = {key: value for key, value in kwargs.items() if value is not None}

        return replace(self, **clean)

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)

        secret_fields = (
            "hf_token",
            "tavily_api_key",
            "brave_api_key",
            "exa_api_key",
            "serpapi_api_key",
            "bing_api_key",
            "google_cse_api_key",
            "custom_web_search_api_key",
        )

        for key in secret_fields:
            values[key] = "***" if values.get(key) else ""

        return values

    def generation_backend_label(self) -> str:
        return GENERATION_BACKEND_LABELS.get(
            self.generation_backend,
            "Hugging Face",
        )

    def active_generation_model(self) -> str:
        if self.generation_backend == "ollama":
            return self.ollama_model

        return self.hf_llm_model

    def generation_ready(self) -> bool:
        if self.generation_backend == "ollama":
            return bool(self.ollama_model.strip())

        return bool(self.hf_token.strip() and self.hf_llm_model.strip())

    def web_search_provider_label(self) -> str:
        if self.web_search_provider == "custom" and self.custom_web_search_provider.strip():
            return self.custom_web_search_provider.strip()

        return WEB_SEARCH_PROVIDER_LABELS.get(
            self.web_search_provider,
            "Custom provider",
        )

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


# Backward-compatible name used elsewhere in the app
DEFAULT_MODEL_CHOICES = DEFAULT_HF_MODEL_CHOICES


def ensure_app_dirs(settings: AppSettings) -> None:
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.semantic_cache_path.parent.mkdir(parents=True, exist_ok=True)
    settings.table_store_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_graph_path.parent.mkdir(parents=True, exist_ok=True)
    settings.multimodal_store_path.parent.mkdir(parents=True, exist_ok=True)
    settings.formula_store_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ocr_block_store_path.parent.mkdir(parents=True, exist_ok=True)
    settings.structured_document_store_path.parent.mkdir(parents=True, exist_ok=True)
