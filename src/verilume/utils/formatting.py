"""Presentation-friendly formatting helpers."""

from __future__ import annotations

from urllib.parse import urlparse

from verilume.core.schemas import LocalSource, WebSource


def source_page_text(source: LocalSource) -> str:
    return f"Page {source.page}" if source.page else "Page not available"


def local_source_rows(sources: list[LocalSource]) -> list[dict[str, str | float]]:
    return [
        {
            "Citation": f"[{source.label}]",
            "Document": source.document,
            "Page": source_page_text(source),
            "Score": round(source.score, 3),
            "Confidence": local_source_confidence([source]),
            "Preview": compact_text(source.text, 220),
        }
        for source in sources
    ]


def web_source_rows(sources: list[WebSource]) -> list[dict[str, str | float | None]]:
    rows: list[dict[str, str | float | None]] = []
    for source in sources:
        source_type = web_source_type(source)
        rows.append(
            {
                "Citation": f"[{source.label}]",
                "Badge": source_badge(source_type),
                "Source": source_display_name(source),
                "Source type": source_type,
                "Confidence": source_confidence(source),
                "Title": source.title,
                "URL": source.url,
                "Date": source.published_date or ", ".join(source.metadata.get("visible_dates", [])),
                "Score": round(source.score, 3) if isinstance(source.score, float) else source.score,
                "Preview": compact_text(source.content, 220),
            }
        )
    return rows


def local_source_confidence(sources: list[LocalSource]) -> str:
    if not sources:
        return "High"
    best_score = max(source.score for source in sources)
    if best_score > 0.8:
        return "High"
    if best_score > 0.6:
        return "Medium"
    return "Low"


def web_source_type(source: WebSource) -> str:
    domain = _domain(source.url)
    haystack = f"{source.title} {source.url} {source.content}".lower()

    if _domain_contains(domain, ("github.com",)):
        return "GitHub"
    if _domain_contains(domain, ("gouv", "gov.", ".gov", "government", "public.lu", "royal.uk")):
        return "Government"
    if _domain_contains(domain, ("university", "uni.", ".edu", "edu.", "uni.lu", "college")):
        return "University"
    if _domain_contains(
        domain,
        (
            "acm.org",
            "arxiv.org",
            "doi.org",
            "ieee.org",
            "semanticscholar.org",
            "researchgate.net",
            "sciencedirect.com",
            "zenodo.org",
        ),
    ) or any(term in haystack for term in ("journal", "paper", "publication", "thesis", "research explorer")):
        return "Research"
    if _domain_contains(
        domain,
        ("bbc.", "reuters.", "apnews.", "rtl.", "today.rtl", "nytimes.", "guardian.", "euronews."),
    ) or "news" in haystack:
        return "News"
    if _domain_contains(domain, ("youtube.com", "youtu.be")):
        return "Video"
    if _domain_contains(domain, ("linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com")):
        return "Social media"
    return "Web"


def source_badge(source_type: str) -> str:
    badges = {
        "University": "\U0001f393 University",
        "GitHub": "\U0001f4bb GitHub",
        "Research": "\U0001f4da Research",
        "Government": "\U0001f3db Government",
        "News": "\U0001f4f0 News",
        "Video": "\U0001f3a5 Video",
        "Social media": "\U0001f464 Social",
        "Web": "\U0001f310 Web",
        "Current information": "\U0001f310 Current",
        "Local document": "\U0001f4c4 Local",
        "AI knowledge": "\U0001f9e0 AI",
        "Model knowledge": "\U0001f9e0 AI",
    }
    return badges.get(source_type, badges["Web"])


def source_confidence(source: WebSource) -> str:
    source_type = web_source_type(source)
    if source_type in {"GitHub", "Government", "University", "Research"}:
        return "High"
    if source_type in {"News", "Video"}:
        return "Medium"
    if source_type == "Social media":
        return "Low"
    if _domain_contains(_domain(source.url), ("blog", "forum", "reddit.com", "medium.com")):
        return "Low"
    return "Medium" if source.content.strip() else "Low"


def source_display_name(source: WebSource) -> str:
    title = compact_text(source.title or "", 58)
    domain = _domain(source.url)
    if not title:
        return domain or "Web source"
    generic_titles = {"official site", "home", "homepage", "videos"}
    if title.lower() in generic_titles and domain:
        return domain
    return title


def compact_text(text: str, limit: int = 240) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _domain(url: str) -> str:
    return urlparse(url or "").netloc.lower().removeprefix("www.")


def _domain_contains(domain: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in domain for pattern in patterns)
