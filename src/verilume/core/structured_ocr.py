"""Generic structured-document extraction from cleaned OCR/text pages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from verilume.core.field_labels import canonicalize_label, load_field_aliases
from verilume.core.ocr_blocks import OCRBlock
from verilume.core.ocr_cleaning import correct_ocr_token


@dataclass(slots=True)
class StructuredField:
    field_name: str
    canonical_name: str
    value: str
    raw_label: str | None
    raw_value: str
    field_type: str
    confidence: float
    page: int | None
    bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class StructuredDocument:
    document_id: str
    document: str
    page: int | None
    document_type: str
    fields: list[StructuredField] = field(default_factory=list)
    confidence: float = 0.0


def extract_structured_document(
    text: str,
    *,
    ocr_blocks: list[OCRBlock] | None = None,
    document: str,
    page: int | None,
) -> StructuredDocument | None:
    cleaned = text or ""
    fields = _extract_label_value_fields(cleaned, page)
    fields.extend(_extract_pattern_fields(cleaned, page))
    fields = _dedupe_fields(fields)
    document_type = classify_structured_document(cleaned, fields)
    if not fields and document_type == "unknown_structured_document":
        return None
    confidence = _document_confidence(document_type, fields)
    document_id = _structured_document_id(document, page, cleaned)
    return StructuredDocument(
        document_id=document_id,
        document=document,
        page=page,
        document_type=document_type,
        fields=fields,
        confidence=confidence,
    )


def classify_structured_document(text: str, fields: list[StructuredField] | None = None) -> str:
    haystack = (text or "").lower()
    fields = fields or []
    field_names = {field.canonical_name for field in fields}
    if re.search(r"\b(passport|nationality|date of birth|surname|given names|identity card)\b", haystack):
        return "identity_document"
    if re.search(r"\b(certificate|certificat|attestation|diploma|degree)\b", haystack):
        return "academic_document" if re.search(r"\b(university|degree|diploma|thesis)\b", haystack) else "certificate"
    if re.search(r"\b(invoice|bill to|subtotal|vat|total due)\b", haystack):
        return "invoice"
    if re.search(r"\b(receipt|paid|payment)\b", haystack):
        return "receipt"
    if re.search(r"\b(dear|to whom it may concern|sincerely|letter)\b", haystack):
        return "letter"
    if re.search(r"\b(contract|agreement|party|clause)\b", haystack):
        return "contract"
    if {"document_number", "reference_number", "issue_date"} & field_names:
        return "form"
    return "unknown_structured_document"


def _extract_label_value_fields(text: str, page: int | None) -> list[StructuredField]:
    aliases = load_field_aliases()
    fields: list[StructuredField] = []
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in lines:
        match = re.match(r"^(?P<label>[A-Za-zÀ-ÿ0-9 ./'()_-]{2,50})\s*[:|]\s*(?P<value>.+)$", line)
        if not match:
            match = re.match(
                r"^(?P<label>(?:date of birth|date of issue|expiry date|document number|reference|nationality|surname|given names|amount|total|issuer|issued by|authority))\s+(?P<value>.+)$",
                line,
                flags=re.IGNORECASE,
            )
        if not match:
            continue
        raw_label = match.group("label").strip()
        raw_value = match.group("value").strip()
        canonical, field_type, label_confidence = canonicalize_label(raw_label, aliases)
        if canonical == "unknown" or not raw_value:
            continue
        value = correct_ocr_token(raw_value, field_type)
        fields.append(
            StructuredField(
                field_name=raw_label,
                canonical_name=canonical,
                value=value,
                raw_label=raw_label,
                raw_value=raw_value,
                field_type=field_type,
                confidence=min(0.98, label_confidence + 0.02),
                page=page,
                bbox=None,
            )
        )
    return fields


def _extract_pattern_fields(text: str, page: int | None) -> list[StructuredField]:
    fields: list[StructuredField] = []
    for index, match in enumerate(re.finditer(r"\b[A-Z0-9]{2,}[-/][A-Z0-9][A-Z0-9-/]{3,}\b", text or ""), start=1):
        fields.append(
            StructuredField(
                field_name=f"detected_code_{index}",
                canonical_name="document_number",
                value=correct_ocr_token(match.group(0), "document_number"),
                raw_label=None,
                raw_value=match.group(0),
                field_type="document_number",
                confidence=0.62,
                page=page,
            )
        )
    for match in re.finditer(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b", text or ""):
        fields.append(
            StructuredField(
                field_name="detected_date",
                canonical_name="date",
                value=correct_ocr_token(match.group(0), "date"),
                raw_label=None,
                raw_value=match.group(0),
                field_type="date",
                confidence=0.58,
                page=page,
            )
        )
    for match in re.finditer(r"\b(?:EUR|USD|GBP|€|\$|£)\s?\d+(?:[.,]\d{2})?\b|\b\d+(?:[.,]\d{2})?\s?(?:EUR|USD|GBP|€|\$|£)\b", text or ""):
        fields.append(
            StructuredField(
                field_name="detected_amount",
                canonical_name="amount",
                value=match.group(0).strip(),
                raw_label=None,
                raw_value=match.group(0),
                field_type="amount",
                confidence=0.6,
                page=page,
            )
        )
    for canonical, field_type, pattern in (
        ("email", "email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        ("phone", "phone", r"\+?\d[\d .()/-]{6,}\d"),
    ):
        for match in re.finditer(pattern, text or ""):
            fields.append(
                StructuredField(
                    field_name=f"detected_{canonical}",
                    canonical_name=canonical,
                    value=match.group(0).strip(),
                    raw_label=None,
                    raw_value=match.group(0),
                    field_type=field_type,
                    confidence=0.62,
                    page=page,
                )
            )
    return fields


def _dedupe_fields(fields: list[StructuredField]) -> list[StructuredField]:
    unique: list[StructuredField] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(fields, key=lambda field_item: field_item.confidence, reverse=True):
        key = (item.canonical_name, item.value.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _document_confidence(document_type: str, fields: list[StructuredField]) -> float:
    if not fields:
        return 0.0
    average = sum(field.confidence for field in fields) / len(fields)
    type_bonus = 0.08 if document_type != "unknown_structured_document" else 0.0
    return max(0.0, min(1.0, average + type_bonus))


def _structured_document_id(document: str, page: int | None, text: str) -> str:
    digest = hashlib.blake2b(f"{document}:{page}:{text[:180]}".encode("utf-8"), digest_size=10)
    return digest.hexdigest()
