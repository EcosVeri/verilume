"""Chat export helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from verilume.core.schemas import RAGResponse


def chat_to_markdown(messages: Sequence[dict[str, Any]], title: str = "Verilume Chat") -> str:
    lines = [
        f"# {title}",
        "",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for message in messages:
        role = str(message.get("role", "assistant")).title()
        content = str(message.get("content", "")).strip()
        timestamp = _message_timestamp(message)
        lines.extend([f"## {role}", "", content or "_No content_", ""])
        if timestamp:
            lines.extend([f"_Timestamp: {timestamp}_", ""])
        response = message.get("response")
        if isinstance(response, RAGResponse):
            lines.extend(_sources_markdown(response))
    return "\n".join(lines).strip() + "\n"


def chat_to_pdf(messages: Sequence[dict[str, Any]], title: str = "Verilume Chat") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        title=title,
    )
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#12161f")
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    story.append(Paragraph(datetime.now().strftime("Exported: %Y-%m-%d %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 18))

    for message in messages:
        role = str(message.get("role", "assistant")).title()
        content = _escape(str(message.get("content", "")).strip() or "No content")
        story.append(Paragraph(role, styles["Heading2"]))
        timestamp = _message_timestamp(message)
        if timestamp:
            story.append(Paragraph(_escape(timestamp), styles["Italic"]))
        story.append(Paragraph(content.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 10))
        response = message.get("response")
        if isinstance(response, RAGResponse):
            for line in _sources_markdown(response):
                if line.startswith("###"):
                    story.append(Paragraph(line.replace("#", "").strip(), styles["Heading3"]))
                elif line.startswith("- "):
                    story.append(Paragraph(_escape(line[2:]), styles["Normal"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()


def _sources_markdown(response: RAGResponse) -> list[str]:
    lines: list[str] = []
    if response.local_sources:
        lines.extend(["### Local Citations", ""])
        for source in response.local_sources:
            page = f", page {source.page}" if source.page else ""
            lines.append(f"- [{source.label}] {source.document}{page}")
        lines.append("")
    if response.web_sources:
        lines.extend(["### Web Citations", ""])
        for source in response.web_sources:
            date_text = source.published_date or ", ".join(source.metadata.get("visible_dates", []))
            suffix = f" ({date_text})" if date_text else ""
            lines.append(f"- [{source.label}] [{source.title}]({source.url}){suffix}")
        lines.append("")
    return lines


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _message_timestamp(message: dict[str, Any]) -> str:
    value = message.get("timestamp")
    if not value:
        return ""
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    return timestamp.strftime("%Y-%m-%d %H:%M")
