"""Structured-field retrieval for scanned/form-like document questions."""

from __future__ import annotations

import re

from verilume.core.field_labels import canonicalize_label
from verilume.core.schemas import LocalSource
from verilume.core.structured_document_store import StructuredDocumentStore

STRUCTURED_QUERY_RE = re.compile(
    r"\b(?:field|fields|document number|reference number|issued|issue date|expire|expiry|"
    r"date of birth|nationality|surname|given names|amount|invoice|certificate|passport|"
    r"identity document|who issued|organization|organisation)\b",
    re.IGNORECASE,
)


def is_structured_document_query(question: str) -> bool:
    return bool(STRUCTURED_QUERY_RE.search(question or ""))


class StructuredRetriever:
    def __init__(self, store: StructuredDocumentStore) -> None:
        self.store = store

    def retrieve(self, question: str, *, limit: int = 5) -> list[LocalSource]:
        if not is_structured_document_query(question):
            return []
        target = structured_query_target(question)
        rows = self.store.search_fields(
            question,
            canonical_name=target.get("canonical_name"),
            field_type=target.get("field_type"),
            document_type=target.get("document_type"),
            limit=limit,
        )
        sources: list[LocalSource] = []
        for index, row in enumerate(rows, start=1):
            text = _structured_source_text(row)
            metadata = {
                "content_type": "structured_field",
                "structured_document_id": row.get("structured_document_id"),
                "document_type": row.get("document_type"),
                "canonical_name": row.get("canonical_name"),
                "field_type": row.get("field_type"),
                "raw_label": row.get("raw_label"),
                "raw_value": row.get("raw_value"),
                "structured_confidence": row.get("confidence"),
                "retrieval": "structured",
            }
            sources.append(
                LocalSource(
                    label=f"S{index}",
                    document=str(row.get("document") or "Unknown document"),
                    page=int(row["page"]) if row.get("page") else None,
                    chunk_id=f"structured:{row.get('id')}",
                    text=text,
                    score=max(0.0, min(1.0, float(row.get("confidence") or 0.0))),
                    metadata=metadata,
                )
            )
        return sources


def structured_query_target(question: str) -> dict[str, str | None]:
    normalized = (question or "").lower()
    canonical = None
    field_type = None
    for label in (
        "date of birth",
        "issue date",
        "expiry date",
        "document number",
        "reference number",
        "nationality",
        "surname",
        "given names",
        "amount",
        "authority",
        "organization",
    ):
        if label in normalized:
            canonical, field_type, _score = canonicalize_label(label)
            break
    if canonical is None:
        if "expire" in normalized or "valid until" in normalized:
            canonical, field_type = "expiry_date", "expiry_date"
        elif "issued" in normalized or "issuer" in normalized:
            canonical, field_type = "issue_date", "issue_date"
        elif "number" in normalized:
            canonical, field_type = "document_number", "document_number"
        elif "date" in normalized:
            canonical, field_type = None, "date"
        elif "amount" in normalized or "total" in normalized:
            canonical, field_type = "amount", "amount"
    document_type = None
    if "passport" in normalized or "identity" in normalized:
        document_type = "identity_document"
    elif "invoice" in normalized:
        document_type = "invoice"
    elif "certificate" in normalized:
        document_type = "certificate"
    return {"canonical_name": canonical, "field_type": field_type, "document_type": document_type}


def _structured_source_text(row: dict) -> str:
    return "\n".join(
        part
        for part in (
            f"Structured field: {row.get('canonical_name')}",
            f"Value: {row.get('value')}",
            f"Raw label: {row.get('raw_label') or 'detected pattern'}",
            f"Document type: {row.get('document_type')}",
            f"Confidence: {float(row.get('confidence') or 0.0):.2f}",
        )
        if part
    )
