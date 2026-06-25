"""Strict filters for short person/entity lookup queries."""

from __future__ import annotations

import re
import unicodedata

_QUESTION_WORDS = {
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "which",
    "explain",
    "summarise",
    "summarize",
    "define",
    "list",
    "look",
    "search",
    "compare",
    "use",
}


def is_short_entity_query(question: str) -> bool:
    text = (question or "").strip()
    if not text or text.endswith("?"):
        return False
    words = text.split()
    if not 1 <= len(words) <= 5:
        return False
    if len(words) == 1 and not words[0][:1].isupper():
        return False
    if len(words) > 1:
        name_like_words = [word for word in words if word[:1].isupper()]
        if len(name_like_words) < max(2, len(words) - 1):
            return False
    return words[0].lower().strip(".,:;") not in _QUESTION_WORDS


def source_matches_entity(query: str, text: str) -> bool:
    q = normalize_entity_text(query)
    t = normalize_entity_text(text)
    if not q or not t:
        return False

    if re.search(rf"\b{re.escape(q)}\b", t):
        return True

    q_parts = [part for part in q.split() if len(part) > 2]
    if len(q_parts) >= 2:
        return sum(_word_present(part, t) for part in q_parts) >= len(q_parts) - 1

    if len(q_parts) == 1:
        return _word_present(q_parts[0], t)

    return False


def normalize_entity_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _word_present(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None
