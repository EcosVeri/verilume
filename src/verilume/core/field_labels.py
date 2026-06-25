"""Generic structured-document field label aliases."""

from __future__ import annotations

import json
import re
from pathlib import Path

FIELD_LABEL_ALIASES: dict[str, list[str]] = {
    "surname": ["surname", "last name", "nom", "family name"],
    "given_names": ["given names", "first name", "forenames", "prénoms", "prenoms"],
    "person_name": ["name", "full name", "candidate", "student", "holder"],
    "organization_name": ["organization", "organisation", "company", "institution", "university"],
    "nationality": ["nationality", "nationalité", "citizenship"],
    "date_of_birth": ["date of birth", "birth date", "date de naissance", "dob"],
    "issue_date": ["issue date", "date of issue", "issued on", "date de délivrance"],
    "expiry_date": ["expiry date", "expiration date", "valid until", "date d'expiration", "expires"],
    "document_number": ["document no", "document number", "number", "no.", "passport no", "certificate no", "reference"],
    "reference_number": ["reference", "reference no", "ref", "ref no", "case number"],
    "authority": ["authority", "issuer", "issued by", "autorité", "signed by"],
    "amount": ["amount", "total", "balance", "price", "sum"],
    "currency": ["currency"],
    "address": ["address", "adresse", "residence"],
    "country": ["country", "country code", "pays"],
    "place": ["place", "place of issue", "birthplace", "location"],
    "email": ["email", "e-mail"],
    "phone": ["phone", "telephone", "tel", "mobile"],
    "title": ["title", "subject"],
    "role": ["role", "position", "function"],
}

FIELD_TYPES: dict[str, str] = {
    "surname": "person_name",
    "given_names": "person_name",
    "person_name": "person_name",
    "organization_name": "organization_name",
    "nationality": "nationality",
    "date_of_birth": "date_of_birth",
    "issue_date": "issue_date",
    "expiry_date": "expiry_date",
    "document_number": "document_number",
    "reference_number": "reference_number",
    "authority": "authority",
    "amount": "amount",
    "currency": "currency",
    "address": "address",
    "country": "country",
    "place": "place",
    "email": "email",
    "phone": "phone",
    "title": "title",
    "role": "role",
}


def load_field_aliases(path: Path | None = None) -> dict[str, list[str]]:
    aliases = {key: list(values) for key, values in FIELD_LABEL_ALIASES.items()}
    if path is None:
        path = Path.home() / ".verilume" / "field_aliases.json"
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return aliases
    if isinstance(payload, dict):
        for key, values in payload.items():
            if isinstance(values, list):
                aliases.setdefault(str(key), []).extend(str(value) for value in values)
    return aliases


def canonicalize_label(label: str, aliases: dict[str, list[str]] | None = None) -> tuple[str, str, float]:
    aliases = aliases or load_field_aliases()
    normalized = _normalize_label(label)
    if normalized in {"reference", "ref", "reference no", "ref no", "case number"}:
        return "reference_number", "reference_number", 0.98
    best = ("unknown", "unknown", 0.0)
    for canonical, candidates in aliases.items():
        for candidate in [canonical, *candidates]:
            candidate_norm = _normalize_label(candidate)
            if not candidate_norm:
                continue
            if normalized == candidate_norm:
                return canonical, FIELD_TYPES.get(canonical, "free_text"), 0.98
            if candidate_norm in normalized or normalized in candidate_norm:
                score = min(len(candidate_norm), len(normalized)) / max(len(candidate_norm), len(normalized))
                if score > best[2]:
                    best = (canonical, FIELD_TYPES.get(canonical, "free_text"), max(0.55, score))
    return best


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()
