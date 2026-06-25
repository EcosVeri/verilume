"""Canonical search mode values for Verilume."""

from __future__ import annotations

from enum import Enum


class SearchMode(str, Enum):
    AUTO = "auto"
    LOCAL_ONLY = "local_only"
    LOCAL_AI = "local_ai"
    LOCAL_AI_WEB = "local_ai_web"
    WEB_ONLY = "web_only"
    RESEARCH = "research"
    AI_ONLY = "ai_only"


def search_mode_from_settings(value: str | SearchMode | None) -> SearchMode:
    if isinstance(value, SearchMode):
        return value
    text = str(value or "auto").strip().lower()
    text = text.replace("+", " ").replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    aliases = {
        "auto": SearchMode.AUTO,
        "default": SearchMode.AUTO,
        "local": SearchMode.LOCAL_ONLY,
        "local only": SearchMode.LOCAL_ONLY,
        "local files": SearchMode.LOCAL_ONLY,
        "local ai": SearchMode.LOCAL_AI,
        "local model": SearchMode.LOCAL_AI,
        "local ai web": SearchMode.LOCAL_AI_WEB,
        "local model web": SearchMode.LOCAL_AI_WEB,
        "hybrid": SearchMode.LOCAL_AI_WEB,
        "web": SearchMode.WEB_ONLY,
        "web only": SearchMode.WEB_ONLY,
        "research": SearchMode.RESEARCH,
        "research mode": SearchMode.RESEARCH,
        "ai": SearchMode.AI_ONLY,
        "ai only": SearchMode.AI_ONLY,
        "model": SearchMode.AI_ONLY,
        "model only": SearchMode.AI_ONLY,
    }
    return aliases.get(text, SearchMode.AUTO)


def search_mode_label(mode: SearchMode | str) -> str:
    mode = search_mode_from_settings(mode)
    return {
        SearchMode.AUTO: "Auto",
        SearchMode.LOCAL_ONLY: "Local Only",
        SearchMode.LOCAL_AI: "Local + AI",
        SearchMode.LOCAL_AI_WEB: "Local + AI + Web",
        SearchMode.WEB_ONLY: "Web Only",
        SearchMode.RESEARCH: "Research Mode",
        SearchMode.AI_ONLY: "AI Only",
    }[mode]
