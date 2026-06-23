"""Search planning adapters for interpreted Verilume queries."""

from __future__ import annotations

from dataclasses import dataclass, field

from verilume.core.agents import SearchPlan
from verilume.core.query_interpreter import InterpretedQuery


@dataclass(slots=True)
class ResolvedSearchPlan:
    """Search plan derived from the Query Interpretation Agent output."""

    intent: str
    need_local: bool
    need_web: bool
    need_model: bool
    search_queries: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    response_mode: str = "answer"
    freshness_required: bool = False

    def to_legacy_plan(self) -> SearchPlan:
        return SearchPlan(
            intent=self.intent or "general",
            preferred_sources=list(self.preferred_sources),
            need_local=self.need_local,
            need_web=self.need_web,
            need_model=self.need_model,
            freshness_required=self.freshness_required,
            response_mode=self.response_mode,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "need_local": self.need_local,
            "need_web": self.need_web,
            "need_model": self.need_model,
            "search_queries": self.search_queries,
            "preferred_sources": self.preferred_sources,
            "response_mode": self.response_mode,
            "freshness_required": self.freshness_required,
        }


class SearchPlanner:
    """Convert interpreted intent into retrieval toggles and query candidates."""

    def plan(self, interpretation: InterpretedQuery) -> ResolvedSearchPlan:
        intent = interpretation.intent or "general"
        freshness_required = intent in {"current_or_public_fact", "government", "news"} or any(
            source.lower() in {"official government", "recent web evidence"}
            for source in interpretation.preferred_sources
        )
        response_mode = _response_mode(intent)
        return ResolvedSearchPlan(
            intent=intent,
            need_local=interpretation.use_local,
            need_web=interpretation.use_web,
            need_model=interpretation.use_model_knowledge,
            search_queries=interpretation.normalized_search_queries(),
            preferred_sources=list(interpretation.preferred_sources),
            response_mode=response_mode,
            freshness_required=freshness_required,
        )


def _response_mode(intent: str) -> str:
    if intent == "local_document":
        return "document-grounded"
    if intent == "news":
        return "current-summary"
    if intent in {"current_or_public_fact", "government"}:
        return "current-fact"
    if intent == "scientific_explanation":
        return "explanation"
    if intent == "public_knowledge":
        return "public-answer"
    return "answer"
