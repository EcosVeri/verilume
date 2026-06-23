"""Domain-light query normalization and semantic-ish query matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache


STOPWORDS = {
    "a",
    "about",
    "an",
    "are",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "does",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "total",
    "was",
    "were",
    "what",
    "what's",
    "whats",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}
INTENT_PATTERNS = {
    "what": ("what", "whats", "what's", "define", "explain"),
    "who": ("who", "whose", "whom"),
    "when": ("when", "what date", "what year", "what time"),
    "where": ("where", "which location", "which place"),
    "why": ("why", "how come"),
    "how": ("how", "in what way"),
    "yes_no": ("is", "are", "was", "were", "do", "does", "did", "can", "could", "should"),
}
ATTRIBUTE_ALIASES = {
    "area": {
        "area",
        "big",
        "bigger",
        "extent",
        "landmass",
        "size",
        "surface",
    },
    "population": {
        "inhabitants",
        "people",
        "population",
        "residents",
    },
    "capital": {
        "capital",
        "capital city",
        "seat",
    },
    "date": {
        "date",
        "founded",
        "time",
        "when",
        "year",
    },
    "leader": {
        "chief executive officer",
        "ceo",
        "chancellor",
        "head",
        "leader",
        "president",
        "prime minister",
    },
}
ATTRIBUTE_LOOKUP = {
    alias: canonical
    for canonical, aliases in ATTRIBUTE_ALIASES.items()
    for alias in aliases
}


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    """Canonical representation used for cache keys, fanout, and diagnostics."""

    canonical: str
    original: str
    intent: str
    key_terms: tuple[str, ...] = field(default_factory=tuple)
    entities: tuple[str, ...] = field(default_factory=tuple)


class SemanticQueryNormalizer:
    """Small dependency-free normalizer for broad query equivalence."""

    def normalize(self, query: str) -> NormalizedQuery:
        original = re.sub(r"\s+", " ", (query or "").strip())
        cleaned = self.clean_text(original)
        intent = self.extract_intent(cleaned)
        entities = tuple(self.extract_entities(original))
        key_terms = tuple(self.extract_key_terms(cleaned, entities))
        canonical = self.canonical_from_terms(key_terms)
        return NormalizedQuery(
            canonical=canonical,
            original=original,
            intent=intent,
            key_terms=key_terms,
            entities=entities,
        )

    def clean_text(self, text: str) -> str:
        lowered = (text or "").lower().replace("'s", " ")
        lowered = re.sub(r"[^\w\s'-]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def extract_intent(self, cleaned: str) -> str:
        words = cleaned.split()
        if not words:
            return "statement"
        head = " ".join(words[:3])
        for intent, patterns in INTENT_PATTERNS.items():
            if any(head.startswith(pattern) or pattern in head for pattern in patterns):
                return intent
        return "statement"

    def extract_key_terms(self, cleaned: str, entities: tuple[str, ...]) -> list[str]:
        terms: list[str] = []
        entity_terms = {
            term
            for entity in entities
            for term in re.findall(r"[a-z0-9][a-z0-9'-]*", entity.lower())
        }
        for phrase, canonical in sorted(ATTRIBUTE_LOOKUP.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
                terms.append(canonical)
        for word in cleaned.split():
            if word in STOPWORDS or len(word) <= 2:
                continue
            normalized = ATTRIBUTE_LOOKUP.get(word, word)
            if normalized not in entity_terms:
                terms.append(normalized)
        for entity in entities:
            terms.extend(re.findall(r"[a-z0-9][a-z0-9'-]*", entity.lower()))
        return _unique(terms)[:12]

    def extract_entities(self, text: str) -> list[str]:
        values: list[str] = []
        for value in re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9'-]+)*\b", text or ""):
            folded = value.lower()
            if folded in STOPWORDS or folded in ATTRIBUTE_LOOKUP:
                continue
            values.append(value)
        values.extend(re.findall(r'"([^"]+)"', text or ""))
        lower = (text or "").lower()
        for pattern in (
            r"\b(?:of|in|for|from)\s+([a-z][a-z0-9'-]+(?:\s+[a-z][a-z0-9'-]+){0,4})\b",
            r"\b([a-z][a-z0-9'-]+)\s+(?:area|size|population|capital)\b",
        ):
            for match in re.finditer(pattern, lower):
                candidate = _clean_entity_candidate(match.group(1))
                if candidate:
                    values.append(candidate.title())
        return [
            _display_entity(value)
            for value in _unique([value.strip(" .,:;?!") for value in values if value.strip(" .,:;?!")])
        ]

    def canonical_from_terms(self, terms: tuple[str, ...] | list[str]) -> str:
        return " ".join(_unique([term.lower() for term in terms if term]))

    def variants(self, query: str) -> list[str]:
        normalized = self.normalize(query)
        values = [normalized.original, normalized.canonical]
        entity = " ".join(normalized.entities)
        entity_terms = set(_entity_terms(normalized.entities))
        core_terms = [term for term in normalized.key_terms if term not in entity_terms]
        term_text = " ".join(core_terms or normalized.key_terms)
        if entity and term_text:
            values.extend([f"{entity} {term_text}", f"{term_text} {entity}"])
        if normalized.intent in {"what", "statement"} and term_text:
            values.append(f"{term_text} information")
        if "area" in normalized.key_terms:
            values.extend([f"{entity or term_text} area square kilometers", f"{entity or term_text} total area"])
        if "population" in normalized.key_terms:
            values.append(f"{entity or term_text} population inhabitants")
        return _unique([re.sub(r"\s+", " ", value).strip() for value in values if value.strip()])

    @lru_cache(maxsize=2048)
    def are_semantically_similar(self, first: str, second: str, threshold: float = 0.8) -> bool:
        left = self.normalize(first)
        right = self.normalize(second)
        if not left.canonical or not right.canonical:
            return False
        if left.canonical == right.canonical:
            return True
        left_terms = set(left.key_terms)
        right_terms = set(right.key_terms)
        if left_terms and right_terms:
            overlap = len(left_terms & right_terms) / max(len(left_terms), len(right_terms))
            if overlap >= threshold:
                return True
            core_left = left_terms - set(_entity_terms(left.entities))
            core_right = right_terms - set(_entity_terms(right.entities))
            entity_overlap = set(_entity_terms(left.entities)) & set(_entity_terms(right.entities))
            if entity_overlap and core_left and core_left == core_right:
                return True
        return _string_similarity(left.canonical, right.canonical) >= threshold


DEFAULT_QUERY_NORMALIZER = SemanticQueryNormalizer()


def normalize_query(query: str) -> NormalizedQuery:
    """Normalize a query with the default dependency-free normalizer."""

    return DEFAULT_QUERY_NORMALIZER.normalize(query)


def are_queries_semantically_similar(first: str, second: str, threshold: float = 0.8) -> bool:
    """Return whether two query strings likely ask the same thing."""

    return DEFAULT_QUERY_NORMALIZER.are_semantically_similar(first, second, threshold)


def query_variants(query: str) -> list[str]:
    """Return progressive search variants for a query."""

    return DEFAULT_QUERY_NORMALIZER.variants(query)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value.strip().lower())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _clean_entity_candidate(value: str) -> str:
    words = [
        word
        for word in re.findall(r"[a-z][a-z0-9'-]*", value.lower())
        if word not in STOPWORDS and word not in ATTRIBUTE_LOOKUP
    ]
    return " ".join(words[-4:])


def _display_entity(value: str) -> str:
    words = []
    for word in re.split(r"\s+", value.strip()):
        if word.isupper() and len(word) <= 5:
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)


def _entity_terms(entities: tuple[str, ...]) -> list[str]:
    return [
        term
        for entity in entities
        for term in re.findall(r"[a-z0-9][a-z0-9'-]*", entity.lower())
    ]


def _string_similarity(first: str, second: str) -> float:
    token_left = set(first.split())
    token_right = set(second.split())
    token_score = (
        len(token_left & token_right) / max(len(token_left), len(token_right))
        if token_left and token_right
        else 0.0
    )
    return SequenceMatcher(None, first, second).ratio() * 0.55 + token_score * 0.45
