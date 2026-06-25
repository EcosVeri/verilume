"""Explicit action planner for Verilume's evidence pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from verilume.core.evidence import EvidencePolicy, FactType, classify_question
from verilume.core.query_interpreter import InterpretedQuery
from verilume.settings import normalize_search_mode


SEARCH_LOCAL = "search_local"
ANSWER_MODEL = "answer_model"
SEARCH_WEB = "search_web"
CALCULATE = "calculate"
SUMMARIZE_DOCUMENTS = "summarize_documents"
EXTRACT_TABLE = "extract_table"
BUILD_GRAPH_CONTEXT = "build_graph_context"
RETRIEVE_MULTIMODAL = "retrieve_multimodal"


@dataclass(frozen=True, slots=True)
class ActionPlan:
    actions: list[str]
    reason: str
    policy: str
    question_type: str
    search_queries: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)

    def diagnostics(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "reason": self.reason,
            "policy": self.policy,
            "question_type": self.question_type,
            "search_queries": list(self.search_queries),
            "expected_outputs": list(self.expected_outputs),
        }


class AgenticPlanner:
    """Map interpreted questions into explicit retrieval and tool actions."""

    def plan(
        self,
        question: str,
        interpretation: InterpretedQuery,
        settings: Any,
    ) -> ActionPlan:
        understanding = classify_question(question)
        search_mode = normalize_search_mode(str(getattr(settings, "search_mode", "Auto")))
        web_enabled = bool(getattr(settings, "enable_web_search", False))
        web_ready = _web_ready(settings)
        question_type = _question_type(question, interpretation, understanding.fact_type)
        policy = understanding.evidence_policy.value
        actions: list[str] = []
        expected_outputs: list[str] = []
        reasons: list[str] = []

        if search_mode == "Web Only":
            actions.append(SEARCH_WEB)
            expected_outputs.append("web evidence")
            reasons.append("Web Only mode asks the planner to use web evidence.")
            return _plan(actions, reasons, policy, question_type, interpretation, expected_outputs)

        actions.append(SEARCH_LOCAL)
        expected_outputs.append("local evidence")
        reasons.append("Local evidence is checked first for answerable questions.")

        if question_type == "table_calculation":
            actions.extend([EXTRACT_TABLE, CALCULATE])
            expected_outputs.extend(["relevant table", "calculation result"])
            reasons.append("The question asks for a numeric table calculation.")

        if question_type == "local_document_summary":
            actions.append(SUMMARIZE_DOCUMENTS)
            expected_outputs.append("document summaries")
            reasons.append("The question asks to summarize local documents.")

        if search_mode == "Local Only":
            reasons.append("Local Only mode blocks model and web actions.")
            return _plan(actions, reasons, EvidencePolicy.LOCAL_ONLY.value, question_type, interpretation, expected_outputs)

        dynamic = understanding.fact_type in {FactType.DYNAMIC, FactType.NEWS}
        explicit_web = bool(interpretation.use_web or _explicit_web_request(question))
        local_only_policy = understanding.evidence_policy == EvidencePolicy.LOCAL_ONLY

        if not local_only_policy and not dynamic and search_mode in {"Auto", "Local + AI", "Local + AI + Web", "Research Mode"}:
            actions.append(ANSWER_MODEL)
            expected_outputs.append("AI knowledge support")
            reasons.append("Stable questions can use AI knowledge as supporting evidence.")

        if dynamic:
            reasons.append("Current or changing questions avoid AI as factual evidence.")

        should_use_web = (
            search_mode in {"Local + AI + Web", "Research Mode"}
            or explicit_web
            or dynamic
            or (web_enabled and not local_only_policy and search_mode == "Auto")
        )
        if should_use_web and web_ready:
            actions.append(SEARCH_WEB)
            expected_outputs.append("web evidence")
            reasons.append("Web evidence is enabled or required by the question.")
        elif should_use_web and not web_ready:
            reasons.append("Web evidence was planned but the provider is not configured.")

        return _plan(actions, reasons, policy, question_type, interpretation, expected_outputs)


def _plan(
    actions: list[str],
    reasons: list[str],
    policy: str,
    question_type: str,
    interpretation: InterpretedQuery,
    expected_outputs: list[str],
) -> ActionPlan:
    return ActionPlan(
        actions=_dedupe(actions),
        reason=" ".join(reasons).strip(),
        policy=policy,
        question_type=question_type,
        search_queries=interpretation.normalized_search_queries(),
        expected_outputs=_dedupe(expected_outputs),
    )


def _question_type(question: str, interpretation: InterpretedQuery, fact_type: FactType) -> str:
    normalized = (question or "").lower()
    if _table_question(normalized):
        return "table_calculation"
    if interpretation.intent == "local_document" and _summary_question(normalized):
        return "local_document_summary"
    if fact_type == FactType.LOCAL_DOCUMENT:
        return "local_document"
    if fact_type == FactType.NEWS:
        return "news"
    if fact_type == FactType.DYNAMIC:
        return "current_dynamic_fact"
    if re.search(r"\b(?:what is|define|explain|meaning of)\b", normalized):
        return "definition"
    if fact_type == FactType.SCIENTIFIC:
        return "scientific_explanation"
    if fact_type == FactType.PERSON_LOOKUP:
        return "person_lookup"
    if fact_type == FactType.COMPANY_LOOKUP:
        return "company_lookup"
    if re.search(r"\b(?:compare|versus|vs\.?|difference between)\b", normalized):
        return "comparison"
    if re.search(r"\b(?:translate|translation)\b", normalized):
        return "translation"
    if re.search(r"\b(?:calculate|compute|solve)\b", normalized):
        return "calculation"
    if re.search(r"\b(?:recommend|suggest|best)\b", normalized):
        return "recommendation"
    if re.search(r"\b(?:why|reason|infer|deduce)\b", normalized):
        return "reasoning"
    return "general"


def _table_question(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:average|mean|sum|total|maximum|minimum|median|count|how many|trend|"
            r"correlation|group by|percentage|increase|decrease|ratio|difference)\b",
            normalized,
        )
        and re.search(r"\b(?:csv|table|column|row|value|price|amount|data|dataset)\b", normalized)
    )


def _summary_question(normalized: str) -> bool:
    return bool(re.search(r"\b(?:summari[sz]e|summary|overview)\b", normalized))


def _explicit_web_request(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:search the web|search web|web search|look up|online|internet|reuters|bbc|news)\b",
            (question or "").lower(),
        )
    )


def _web_ready(settings: Any) -> bool:
    if not bool(getattr(settings, "enable_web_search", False)):
        return False
    checker = getattr(settings, "web_search_ready", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return True


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
