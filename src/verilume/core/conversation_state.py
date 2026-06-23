"""Structured conversation memory for semantic follow-up resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ConversationState:
    """Working memory shared by the UI, query interpreter, and RAG pipeline.

    The first group of fields is the stable semantic state used by the new
    interpreter. The remaining fields keep compatibility with the older routing
    code while the evidence pipeline is migrated behind the structured plan.
    """

    active_topic: str = ""
    active_country: str = ""
    active_person: str = ""
    active_document: str = ""
    active_news_story: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    preferred_sources: list[str] = field(default_factory=list)
    last_answer_summary: str = ""
    last_resolved_question: str = ""

    active_entities: list[str] = field(default_factory=list)
    active_topics: list[str] = field(default_factory=list)
    active_documents: list[str] = field(default_factory=list)
    active_web_sources: list[str] = field(default_factory=list)
    active_dates: list[str] = field(default_factory=list)
    active_role: str = ""
    active_company: str = ""
    active_organization: str = ""
    active_law: str = ""
    active_research_topic: str = ""
    active_dataset: str = ""
    intent: str = ""
    expires_after: int = 10
    active_event: str = ""

    def get_person_by_role(self, role: str) -> str:
        return self.roles.get(_role_key(role), "")

    def remember_role(self, role: str, person: str) -> None:
        role_key = _role_key(role)
        cleaned_person = _clean_name(person)
        if role_key and cleaned_person:
            self.roles[role_key] = cleaned_person

    def resolve_role_reference(self, query: str) -> str:
        resolved = query or ""
        for role, person in sorted(self.roles.items(), key=lambda item: len(item[0]), reverse=True):
            if not person:
                continue
            role_pattern = re.escape(role).replace(r"\ ", r"\s+")
            resolved = re.sub(
                rf"\b(?:the\s+)?{role_pattern}\b",
                person,
                resolved,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\s+", " ", resolved).strip()

    def remember_entity(self, name: str, entity_type: str, role: str | None = None) -> None:
        cleaned = _clean_name(name)
        if not cleaned:
            return
        existing = {
            (str(item.get("name", "")).casefold(), str(item.get("type", "")).casefold())
            for item in self.entities
        }
        key = (cleaned.casefold(), entity_type.casefold())
        if key not in existing:
            entity: dict[str, Any] = {"name": cleaned, "type": entity_type}
            if role:
                entity["role"] = role
            self.entities.append(entity)
        if cleaned not in self.active_entities:
            self.active_entities.insert(0, cleaned)


def _role_key(role: str) -> str:
    return re.sub(r"\s+", " ", (role or "").strip().lower())


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", (name or "").strip(" .,:;!?\n\t"))
    return cleaned
