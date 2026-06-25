"""Context-aware OCR cleaning helpers."""

from __future__ import annotations

import re
import unicodedata


def clean_ocr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"([.!?,;:])\1{2,}", r"\1", normalized)
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    lines = []
    for raw_line in normalized.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(_clean_scanner_artifacts(line))
    return "\n".join(lines).strip()


def correct_ocr_token(token: str, field_type: str | None = None) -> str:
    value = token or ""
    kind = (field_type or "").lower()
    if kind in {"document_number", "reference_number", "amount", "date", "issue_date", "expiry_date", "date_of_birth"}:
        value = re.sub(r"(?<=\d)[Oo](?=\d)", "0", value)
        value = re.sub(r"(?<=\d)[Il](?=\d)", "1", value)
        value = re.sub(r"(?<=\d)S(?=\d)", "5", value)
    if kind in {"date", "issue_date", "expiry_date", "date_of_birth"}:
        value = re.sub(r"[|]", "/", value)
        value = re.sub(r"\s*([./-])\s*", r"\1", value)
    return value.strip()


def _clean_scanner_artifacts(line: str) -> str:
    value = re.sub(r"^[|:_\-~• ]{2,}", "", line)
    value = re.sub(r"[|:_\-~ ]{3,}$", "", value)
    return value.strip()
