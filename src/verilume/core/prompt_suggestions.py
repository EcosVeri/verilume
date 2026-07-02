"""Context-aware prompt suggestions for the current local library."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from verilume.core.document_index import IndexedDocument

MAX_SUGGESTIONS = 6
LARGE_LIBRARY_THRESHOLD = 20
REQUIRED_LIBRARY_PROMPTS = (
    "Summarise uploaded documents",
    "List indexed documents",
    "Compare local and web evidence",
)


@dataclass(frozen=True, slots=True)
class PromptSuggestion:
    title: str
    prompt: str
    category: str
    priority: float
    document_id: str | None
    document_type: str | None
    # Filename of the document this prompt was derived from, so the pipeline
    # can focus retrieval on it instead of re-guessing the source from text.
    document_filename: str | None = None


def generate_suggested_prompts(
    document_index: Sequence[IndexedDocument],
    recent_history: Sequence[Any],
    settings: Any,
) -> list[PromptSuggestion]:
    """Generate ranked prompt suggestions from the current document index."""
    documents = list(document_index or [])
    if not documents:
        return _rank_and_limit(_empty_library_suggestions(), settings=settings)

    suggestions: list[PromptSuggestion] = []
    suggestions.extend(_generic_library_suggestions())
    suggestions.extend(_recent_activity_suggestions(recent_history))
    suggestions.extend(_collection_suggestions(documents))

    latest_document = _latest_document(documents)
    if latest_document is not None:
        suggestions.extend(_latest_upload_suggestions(latest_document, len(documents)))

    if len(documents) <= LARGE_LIBRARY_THRESHOLD:
        for document in _important_documents(documents)[:3]:
            suggestions.extend(_document_suggestions(document))
    else:
        suggestions.extend(_large_library_suggestions(documents))

    suggestions = _apply_history_boosts(suggestions, recent_history)
    return _rank_and_limit(suggestions, settings=settings)


def classify_document_type(document: IndexedDocument) -> str:
    """Return a stable prompt category for a document."""
    raw_type = _normalize_token(document.document_type)
    aliases = {
        "researchpaper": "scientific_paper",
        "scientificpaper": "scientific_paper",
        "paper": "scientific_paper",
        "journalarticle": "scientific_paper",
        "spreadsheet": "spreadsheet",
        "csv": "spreadsheet",
        "table": "table",
        "pptx": "presentation",
        "pptm": "presentation",
        "deck": "presentation",
        "identity": "identity_document",
        "identitydocument": "identity_document",
        "id": "identity_document",
        "image": "image_document",
        "imagedocument": "image_document",
        "document": "",
    }
    if raw_type in aliases:
        mapped = aliases[raw_type]
        if mapped:
            return mapped

    known = {
        "scientific_paper",
        "textbook",
        "thesis",
        "report",
        "presentation",
        "spreadsheet",
        "table",
        "identity_document",
        "certificate",
        "invoice",
        "letter",
        "contract",
        "manual",
        "policy",
        "book",
        "article",
        "notes",
        "meeting_minutes",
        "image_document",
    }
    if document.document_type in known:
        return document.document_type

    haystack = _document_haystack(document)
    suffix = Path(document.filename).suffix.lower()
    if suffix in {".pptx", ".pptm", ".ppsx", ".potx"}:
        return "presentation"
    if suffix == ".csv":
        return "spreadsheet"
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}:
        return "image_document"
    if _has_any(haystack, "passport", "identity card", "id card", "nationality"):
        return "identity_document"
    if _has_any(haystack, "invoice", "line item", "total due", "vat", "amount due"):
        return "invoice"
    if _has_any(haystack, "certificate", "attestation", "certifies", "issued by"):
        return "certificate"
    if _has_any(haystack, "doctoral thesis", "phd thesis", "dissertation", "thesis"):
        return "thesis"
    if _has_any(haystack, "abstract", "methodology", "doi", "references", "journal"):
        return "scientific_paper"
    if _has_any(haystack, "textbook", "chapter", "worked example", "exercise"):
        return "textbook"
    if _has_any(haystack, "presentation", "speaker notes", "slide deck"):
        return "presentation"
    if _has_any(haystack, "policy", "procedure", "compliance"):
        return "policy"
    if _has_any(haystack, "contract", "agreement", "terms and conditions"):
        return "contract"
    if _has_any(haystack, "minutes", "attendees", "action items"):
        return "meeting_minutes"
    if _has_any(haystack, "manual", "guide", "handbook", "instructions"):
        return "manual"
    if _has_any(haystack, "report", "findings", "recommendations"):
        return "report"
    if _has_any(haystack, "letter", "dear ", "sincerely"):
        return "letter"
    if _has_any(haystack, "notes", "note"):
        return "notes"
    if _has_any(haystack, "book"):
        return "book"
    if _has_any(haystack, "article"):
        return "article"
    return "unknown"


def _empty_library_suggestions() -> list[PromptSuggestion]:
    return [
        _suggestion("Upload documents", "How do I upload documents?", "onboarding", 1.0),
        _suggestion("Build your knowledge base", "How do I build my knowledge base?", "onboarding", 0.96),
        _suggestion("Supported file types", "Which file types can I upload?", "onboarding", 0.88),
        _suggestion("How do I start?", "How do I start using Verilume?", "onboarding", 0.84),
    ]


def _generic_library_suggestions() -> list[PromptSuggestion]:
    return [
        _suggestion("Summarise uploaded documents", "Summarise uploaded documents", "collection", 0.86),
        _suggestion("List indexed documents", "List indexed documents", "inventory", 0.79),
        _suggestion("What changed in my latest upload?", "What changed in my latest upload?", "recent_upload", 0.77),
        _suggestion("Compare local and web evidence", "Compare local and web evidence", "comparison", 0.72),
        _suggestion("Compare local evidence", "Compare local evidence", "comparison", 0.62),
        _suggestion("Search my documents", "Search my documents", "search", 0.58),
    ]


def _collection_suggestions(documents: Sequence[IndexedDocument]) -> list[PromptSuggestion]:
    if len(documents) < 2:
        return []
    document_types = {classify_document_type(document) for document in documents}
    suggestions = [
        _suggestion("Find common topics", "Find common topics across uploaded documents", "collection", 0.82),
        _suggestion("Compare documents", "Compare uploaded documents", "comparison", 0.8),
        _suggestion("List all document summaries", "List all document summaries", "collection", 0.7),
    ]
    academic_count = sum(
        1
        for document in documents
        if classify_document_type(document) in {"scientific_paper", "thesis", "textbook", "article"}
    )
    if academic_count >= 2:
        suggestions.extend(
            [
                _suggestion("Compare uploaded papers", "Compare uploaded papers", "comparison", 0.84),
                _suggestion("Create literature review", "Create a literature review from uploaded documents", "collection", 0.83),
            ]
        )
    if document_types & {"scientific_paper", "thesis", "textbook", "manual"}:
        suggestions.append(_suggestion("Which documents contain equations?", "Which documents contain equations?", "formula", 0.73))
    return suggestions


def _large_library_suggestions(documents: Sequence[IndexedDocument]) -> list[PromptSuggestion]:
    suggestions = [
        _suggestion("Find common research themes", "Find common research themes across uploaded documents", "collection", 0.92),
        _suggestion("Find duplicate information", "Find duplicate information across uploaded documents", "collection", 0.78),
    ]
    if any(classify_document_type(document) == "scientific_paper" for document in documents):
        suggestions.append(_suggestion("Create literature review", "Create a literature review from uploaded documents", "collection", 0.91))
    return suggestions


def _recent_activity_suggestions(recent_history: Sequence[Any]) -> list[PromptSuggestion]:
    if not _history_items(recent_history):
        return []
    return [
        _suggestion("Continue previous research", "Continue previous research", "recent_activity", 0.81),
        _suggestion("Compare with previous search", "Compare with previous search", "recent_activity", 0.68),
    ]


def _latest_upload_suggestions(
    document: IndexedDocument,
    document_count: int,
) -> list[PromptSuggestion]:
    document_type = classify_document_type(document)
    suggestions = [
        _suggestion("Summarise latest upload", "Summarise latest upload", "recent_upload", 0.94, document, document_type),
        _suggestion("Explain newest document", "Explain the newest document", "recent_upload", 0.87, document, document_type),
    ]
    if document_count > 1:
        suggestions.append(
            _suggestion(
                "Compare latest upload with library",
                "Compare latest upload with existing library",
                "recent_upload",
                0.84,
                document,
                document_type,
            )
        )
    return suggestions


def _document_suggestions(document: IndexedDocument) -> list[PromptSuggestion]:
    document_type = classify_document_type(document)
    subject = _document_subject(document, document_type)
    templates = _type_templates(document_type)
    suggestions: list[PromptSuggestion] = []
    for title_template, prompt_template, category, priority in templates:
        suggestions.append(
            _suggestion(
                title_template.format(subject=subject),
                prompt_template.format(subject=subject),
                category,
                priority + _document_importance(document) * 0.08,
                document,
                document_type,
            )
        )
    return suggestions


def _type_templates(document_type: str) -> tuple[tuple[str, str, str, float], ...]:
    templates: dict[str, tuple[tuple[str, str, str, float], ...]] = {
        "scientific_paper": (
            ("Summarise {subject}", "Summarise {subject}", "summary", 0.9),
            ("Key findings in {subject}", "What are the key findings in {subject}?", "analysis", 0.9),
            ("Explain the methodology", "Explain the methodology in {subject}", "analysis", 0.8),
            ("Extract main contributions", "Extract the main contributions from {subject}", "analysis", 0.76),
            ("What equations are introduced?", "What equations are introduced in {subject}?", "formula", 0.74),
        ),
        "thesis": (
            ("Summarise {subject}", "Summarise {subject}", "summary", 0.9),
            ("Research objectives", "Explain the research objectives in {subject}", "analysis", 0.84),
            ("Summarise methodology", "Summarise the methodology in {subject}", "analysis", 0.82),
            ("Key conclusions", "List key conclusions from {subject}", "analysis", 0.77),
            ("Extract important equations", "Extract important equations from {subject}", "formula", 0.72),
        ),
        "textbook": (
            ("Summarise {subject}", "Summarise {subject}", "summary", 0.88),
            ("Explain main topics", "Explain the main topics in {subject}", "analysis", 0.83),
            ("Generate study notes", "Generate study notes from {subject}", "study", 0.8),
            ("Find worked examples", "Find worked examples in {subject}", "study", 0.74),
            ("List formulas", "List formulas in {subject}", "formula", 0.72),
        ),
        "presentation": (
            ("Summarise {subject}", "Summarise {subject}", "summary", 0.89),
            ("Extract presentation topics", "Extract presentation topics from {subject}", "analysis", 0.88),
            ("Create speaker notes", "Create speaker notes from {subject}", "writing", 0.9),
            ("Extract key points", "Extract key points from {subject}", "analysis", 0.78),
        ),
        "report": (
            ("Summarise {subject}", "Summarise {subject}", "summary", 0.87),
            ("Recommendations", "What are the recommendations in {subject}?", "analysis", 0.84),
            ("Important findings", "List important findings from {subject}", "analysis", 0.82),
            ("Extract statistics", "Extract statistics from {subject}", "analysis", 0.75),
        ),
        "spreadsheet": (
            ("Summarise the data", "Summarise the data", "summary", 0.87),
            ("Calculate statistics", "Calculate statistics from the uploaded data", "table", 0.84),
            ("Find trends", "Find trends in the uploaded data", "table", 0.82),
            ("Show important columns", "Show important columns in the uploaded data", "table", 0.76),
            ("Detect anomalies", "Detect anomalies in the uploaded data", "table", 0.72),
        ),
        "identity_document": (
            ("Extract structured information", "Extract structured information", "structured", 0.9),
            ("List detected fields", "List detected fields", "structured", 0.84),
            ("Verify extracted text", "Verify extracted text", "structured", 0.78),
        ),
        "certificate": (
            ("Summarise certificate", "Summarise certificate", "summary", 0.88),
            ("Extract important fields", "Extract important fields from the certificate", "structured", 0.91),
            ("List important dates", "List important dates from the certificate", "structured", 0.82),
            ("Who issued the certificate?", "Who issued the certificate?", "structured", 0.78),
        ),
        "invoice": (
            ("Extract invoice information", "Extract invoice information", "structured", 0.9),
            ("List line items", "List line items", "table", 0.84),
            ("Calculate totals", "Calculate totals", "table", 0.82),
            ("Extract vendor details", "Extract vendor details", "structured", 0.76),
        ),
        "letter": (
            ("Summarise letter", "Summarise letter", "summary", 0.86),
            ("Who sent this?", "Who sent this?", "structured", 0.8),
            ("Extract important dates", "Extract important dates from the letter", "structured", 0.78),
            ("List action items", "List action items from the letter", "analysis", 0.74),
        ),
    }
    return templates.get(
        document_type,
        (
            ("Summarise {subject}", "Summarise {subject}", "summary", 0.82),
            ("Explain {subject}", "Explain {subject}", "analysis", 0.76),
            ("List key topics", "List key topics in {subject}", "analysis", 0.72),
        ),
    )


def _rank_and_limit(
    suggestions: Iterable[PromptSuggestion],
    *,
    settings: Any,
) -> list[PromptSuggestion]:
    del settings
    deduped = _dedupe_suggestions(suggestions)
    deduped.sort(key=lambda item: (-item.priority, item.title.lower()))
    selected = deduped[:MAX_SUGGESTIONS]
    if _contains_library_prompt_set(deduped):
        selected = _ensure_required_library_prompts(selected, deduped)
        selected.sort(key=lambda item: (-item.priority, item.title.lower()))
    return selected[:MAX_SUGGESTIONS]


def _contains_library_prompt_set(suggestions: Sequence[PromptSuggestion]) -> bool:
    titles = {suggestion.title for suggestion in suggestions}
    return all(title in titles for title in REQUIRED_LIBRARY_PROMPTS)


def _ensure_required_library_prompts(
    selected: Sequence[PromptSuggestion],
    ranked: Sequence[PromptSuggestion],
) -> list[PromptSuggestion]:
    values = list(selected)
    selected_titles = {suggestion.title for suggestion in values}
    required_by_title = {
        suggestion.title: suggestion
        for suggestion in ranked
        if suggestion.title in REQUIRED_LIBRARY_PROMPTS
    }
    for title in REQUIRED_LIBRARY_PROMPTS:
        if title in selected_titles or title not in required_by_title:
            continue
        replacement = _lowest_priority_optional_index(values)
        if replacement is None:
            continue
        values[replacement] = required_by_title[title]
        selected_titles = {suggestion.title for suggestion in values}
    return values


def _lowest_priority_optional_index(values: Sequence[PromptSuggestion]) -> int | None:
    optional = [
        (index, suggestion.priority)
        for index, suggestion in enumerate(values)
        if suggestion.title not in REQUIRED_LIBRARY_PROMPTS
    ]
    if not optional:
        return None
    return min(optional, key=lambda item: item[1])[0]


def _dedupe_suggestions(suggestions: Iterable[PromptSuggestion]) -> list[PromptSuggestion]:
    best_by_key: dict[str, PromptSuggestion] = {}
    for suggestion in suggestions:
        title = _display_text(suggestion.title)
        prompt = _display_text(suggestion.prompt)
        if not title or not prompt:
            continue
        cleaned = replace(suggestion, title=title, prompt=prompt)
        key = _normalize_token(prompt)
        previous = best_by_key.get(key)
        if previous is None or cleaned.priority > previous.priority:
            best_by_key[key] = cleaned
    return list(best_by_key.values())


def _apply_history_boosts(
    suggestions: Sequence[PromptSuggestion],
    recent_history: Sequence[Any],
) -> list[PromptSuggestion]:
    history = " ".join(_history_items(recent_history)).lower()
    if not history:
        return list(suggestions)
    boosted: list[PromptSuggestion] = []
    for suggestion in suggestions:
        priority = suggestion.priority
        category = suggestion.category
        if any(marker in history for marker in ("summarise", "summarize", "summary")) and category == "summary":
            priority += 0.05
        if any(marker in history for marker in ("equation", "formula", "math")) and category == "formula":
            priority += 0.24
        if any(marker in history for marker in ("compare", "difference")) and category == "comparison":
            priority += 0.06
        boosted.append(replace(suggestion, priority=priority))
    return boosted


def _important_documents(documents: Sequence[IndexedDocument]) -> list[IndexedDocument]:
    indexed = list(enumerate(documents))
    indexed.sort(
        key=lambda item: (
            _document_importance(item[1]),
            _document_timestamp(item[1]),
            item[0],
        ),
        reverse=True,
    )
    return [document for _index, document in indexed]


def _latest_document(documents: Sequence[IndexedDocument]) -> IndexedDocument | None:
    if not documents:
        return None
    indexed = list(enumerate(documents))
    indexed.sort(key=lambda item: (_document_timestamp(item[1]), item[0]), reverse=True)
    return indexed[0][1]


def _document_importance(document: IndexedDocument) -> float:
    page_score = min(0.36, max(0, document.page_count) / 300)
    chunk_score = min(0.3, max(0, document.chunk_count) / 500)
    metadata_score = 0.0
    if document.summary:
        metadata_score += 0.18
    if document.keywords:
        metadata_score += 0.1
    if _has_distinct_title(document):
        metadata_score += 0.06
    return page_score + chunk_score + metadata_score


def _document_timestamp(document: IndexedDocument) -> float:
    raw = str(document.last_updated or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    path = Path(str(document.source_path or ""))
    try:
        return path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        return 0.0


def _document_subject(document: IndexedDocument, document_type: str) -> str:
    title = _clean_title(document.title)
    stem = _clean_title(Path(document.filename).stem.replace("_", " ").replace("-", " "))
    if title and not _looks_like_generic_title(title):
        return _shorten_title(title)
    if stem and len(stem.split()) >= 2:
        return _shorten_title(stem)
    label = _friendly_document_type(document_type)
    return f"the {label}"


def _friendly_document_type(document_type: str) -> str:
    labels = {
        "scientific_paper": "paper",
        "identity_document": "identity document",
        "image_document": "image document",
        "meeting_minutes": "meeting minutes",
        "unknown": "document",
    }
    return labels.get(document_type, document_type.replace("_", " ") or "document")


def _has_distinct_title(document: IndexedDocument) -> bool:
    title = _clean_title(document.title)
    if not title or _looks_like_generic_title(title):
        return False
    stem = _clean_title(Path(document.filename).stem.replace("_", " ").replace("-", " "))
    return bool(title and title.lower() != stem.lower())


def _display_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_title(value: str) -> str:
    cleaned = _display_text(value)
    cleaned = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", cleaned).strip()
    return cleaned


def _shorten_title(title: str, limit: int = 64) -> str:
    if len(title) <= limit:
        return title
    shortened = title[: limit - 1].rsplit(" ", maxsplit=1)[0]
    return shortened or title[:limit]


def _looks_like_generic_title(value: str) -> bool:
    normalized = _normalize_token(value)
    return normalized in {"document", "unknown", "indexeddocument", "indexedlocaldocumentcontent"}


def _history_items(recent_history: Sequence[Any]) -> list[str]:
    items: list[str] = []
    for item in recent_history or []:
        if isinstance(item, dict):
            value = item.get("content") or item.get("prompt") or ""
        else:
            value = item
        text = _display_text(str(value))
        if text:
            items.append(text)
    return items


def _document_haystack(document: IndexedDocument) -> str:
    return " ".join(
        [
            document.filename,
            document.title,
            document.summary,
            " ".join(document.keywords),
            document.authors,
        ]
    ).lower()


def _has_any(haystack: str, *markers: str) -> bool:
    return any(marker in haystack for marker in markers)


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _suggestion(
    title: str,
    prompt: str,
    category: str,
    priority: float,
    document: IndexedDocument | None = None,
    document_type: str | None = None,
) -> PromptSuggestion:
    return PromptSuggestion(
        title=title,
        prompt=prompt,
        category=category,
        priority=priority,
        document_id=document.document_id if document else None,
        document_type=document_type,
        document_filename=document.filename if document else None,
    )
