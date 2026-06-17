"""Hugging Face text generation."""

from __future__ import annotations

import re
from datetime import date
from collections.abc import Sequence

from huggingface_hub import InferenceClient

from verilume.core.schemas import ChatMessage, LocalSource, WebSource
from verilume.settings import AppSettings


class GenerationError(RuntimeError):
    """Raised when Hugging Face generation cannot complete."""


MODEL_SELECTION_WARNING = (
    "The selected Hugging Face model is temporarily unavailable, out of capacity, "
    "or incompatible with the current provider/token. Select another model in the sidebar "
    "or enter a compatible custom Hugging Face model ID."
)
LOCAL_UNKNOWN = "I do not know from the local files."
MODEL_UNKNOWN = "I do not know from the model."
CITATION_PATTERN = re.compile(r"\[(?:S|W)\d+\]")


def _history_messages(history: Sequence[ChatMessage], max_turns: int) -> list[dict[str, str]]:
    trimmed = list(history)[-(max_turns * 2) :]
    return [{"role": item.role, "content": item.content} for item in trimmed if item.content.strip()]


def _local_context(sources: Sequence[LocalSource]) -> str:
    lines: list[str] = []
    for source in sources:
        page = f", page {source.page}" if source.page else ""
        lines.append(
            f"[{source.label}] {source.document}{page}\n"
            f"{source.text.strip()[:1800]}"
        )
    return "\n\n".join(lines)


def _web_context(sources: Sequence[WebSource]) -> str:
    lines: list[str] = []
    for source in sources:
        date_lines = []
        if source.published_date:
            date_lines.append(f"Published date: {source.published_date}")
        visible_dates = source.metadata.get("visible_dates", [])
        if visible_dates:
            date_lines.append(f"Visible dates in source: {', '.join(map(str, visible_dates))}")
        date_lines.append(f"Freshness: {_freshness_note(source)}")
        lines.append(
            f"[{source.label}] {source.title}\n"
            f"URL: {source.url}\n"
            f"{chr(10).join(date_lines)}\n"
            f"{source.content.strip()[:1400]}"
        )
    return "\n\n".join(lines)


def _freshness_note(source: WebSource) -> str:
    years: list[int] = []
    if source.published_date:
        years.extend(int(match) for match in re.findall(r"\b(20\d{2}|19\d{2})\b", source.published_date))
    visible_dates = source.metadata.get("visible_dates", [])
    for value in visible_dates:
        years.extend(int(match) for match in re.findall(r"\b(20\d{2}|19\d{2})\b", str(value)))
    if not years:
        return "unknown; do not treat as current status unless the page itself is an official current profile."
    newest_year = max(years)
    current_year = date.today().year
    if newest_year < current_year:
        return (
            f"appears old; newest visible year is {newest_year}. Use as historical evidence, "
            "not as proof of current status."
        )
    return f"appears current for {current_year} if the source content supports the claim."


def _answer_style_instruction(style: str) -> str:
    instructions = {
        "Short": (
            "Answer style: Short. Use 2-4 direct sentences unless a tiny bullet list is clearly clearer."
        ),
        "Standard": (
            "Answer style: Standard. Give a concise, complete answer with light structure when useful."
        ),
        "Detailed": (
            "Answer style: Detailed. Use clear Markdown sections or bullets, include key details, "
            "and add an example when it improves understanding."
        ),
        "Research": (
            "Answer style: Research. Be source-aware and rigorous: define terms, state assumptions "
            "or limits, compare evidence when relevant, and preserve citations carefully."
        ),
    }
    return instructions.get(style, instructions["Standard"])


class HuggingFaceGenerator:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._client_instance = None

    def _client(self):
        if self._client_instance is None:
            provider = self.settings.hf_provider.strip()
            kwargs = {
                "model": self.settings.hf_llm_model,
                "token": self.settings.hf_token,
                "timeout": self.settings.hf_timeout_seconds,
            }
            if provider and provider.lower() != "auto":
                kwargs["provider"] = provider
            self._client_instance = InferenceClient(**kwargs)
        return self._client_instance

    def answer(
        self,
        question: str,
        history: Sequence[ChatMessage],
        local_sources: Sequence[LocalSource],
        web_sources: Sequence[WebSource] | None = None,
    ) -> str:
        if not self.settings.hf_token.strip():
            raise GenerationError("Enter a Hugging Face token to generate an answer.")

        web_sources = web_sources or []
        style_instruction = _answer_style_instruction(self.settings.answer_style)
        system = (
            "You are Verilume, a precise desktop research assistant. "
            "Prefer LOCAL DOCUMENT CONTEXT over general model knowledge. "
            "When local context supports a claim, cite it with the exact local labels like [S1]. "
            "When web context supports a claim, cite it with exact web labels like [W1]. "
            "Use local and web citations separately; never invent labels. "
            "If no source label is present in the context, do not include any [S] or [W] citation. "
            "If local context does not answer the question, you may answer from reliable general "
            "model knowledge without local citations. If neither local context nor reliable general "
            "model knowledge can answer, respond exactly WEB_SEARCH_NEEDED. For conceptual "
            "questions asking what something is, organize the answer with concise Markdown "
            "sections: Definition, Key Concepts, Applications, and Example when those sections fit. "
            f"{style_instruction}"
        )
        context = (
            "LOCAL DOCUMENT CONTEXT:\n"
            f"{_local_context(local_sources) or 'No local source labels are available.'}\n\n"
            "WEB CONTEXT:\n"
            f"{_web_context(web_sources) or 'No web source labels are available.'}"
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(_history_messages(history, self.settings.max_history_turns))
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer directly. Include citation labels in the answer wherever sources are used."
                ),
            }
        )
        allowed_labels = [source.label for source in local_sources]
        allowed_labels.extend(source.label for source in web_sources)
        return _sanitize_citations(self._chat(messages), allowed_labels)

    def answer_local(
        self,
        question: str,
        history: Sequence[ChatMessage],
        local_sources: Sequence[LocalSource],
    ) -> str:
        if not local_sources:
            return LOCAL_UNKNOWN
        if not self.settings.hf_token.strip():
            raise GenerationError("Enter a Hugging Face token to generate an answer.")

        style_instruction = _answer_style_instruction(self.settings.answer_style)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful local RAG assistant. Answer using ONLY the provided "
                    "local document context. When using local context, cite exact labels like "
                    "[S1] or [S2]. If the answer is not clearly in the local context, say "
                    f"exactly: {LOCAL_UNKNOWN} Do not use model knowledge. Do not use web search. "
                    "Do not invent facts or citations. For explanatory concept answers, use concise "
                    "Markdown sections when helpful: Definition, Key Concepts, Applications, Example. "
                    f"{style_instruction}"
                ),
            }
        ]
        messages.extend(_history_messages(history, self.settings.max_history_turns))
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Local context:\n{_local_context(local_sources)}"
                ),
            }
        )
        return _sanitize_citations(self._chat(messages), [source.label for source in local_sources])

    def answer_model_knowledge(
        self,
        question: str,
        history: Sequence[ChatMessage],
    ) -> str:
        if not self.settings.hf_token.strip():
            raise GenerationError("Enter a Hugging Face token to generate an answer.")

        style_instruction = _answer_style_instruction(self.settings.answer_style)
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer from reliable general model knowledge only. Do not use local files. "
                    "Do not use web search. Do not cite [S] or [W] labels. If you are unsure, "
                    "if the answer depends on current information, if the fact may have "
                    "changed recently, or if the question asks for a person's current role/status, "
                    f"say exactly: {MODEL_UNKNOWN} For conceptual questions asking what something "
                    "is, organize the answer with concise Markdown sections: Definition, Key "
                    "Concepts, Applications, and Example when those sections fit. "
                    f"{style_instruction}"
                ),
            }
        ]
        messages.extend(_history_messages(history, self.settings.max_history_turns))
        messages.append({"role": "user", "content": question})
        return _sanitize_citations(self._chat(messages), [])

    def answer_with_web(
        self,
        question: str,
        history: Sequence[ChatMessage],
        local_answer: str,
        model_answer: str,
        local_sources: Sequence[LocalSource],
        web_sources: Sequence[WebSource],
    ) -> str:
        return self.answer_final(
            question=question,
            history=history,
            local_answer=local_answer,
            model_answer=model_answer,
            local_sources=local_sources,
            web_sources=web_sources,
        )

    def answer_final(
        self,
        question: str,
        history: Sequence[ChatMessage],
        local_answer: str,
        model_answer: str,
        local_sources: Sequence[LocalSource],
        web_sources: Sequence[WebSource],
    ) -> str:
        if not self.settings.hf_token.strip():
            raise GenerationError("Enter a Hugging Face token to generate an answer.")

        style_instruction = _answer_style_instruction(self.settings.answer_style)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Verilume. Produce the final answer from staged evidence. "
                    "Weight evidence in this order: local document evidence is primary for facts "
                    "about uploaded files; web evidence is primary for current or public facts; "
                    "model knowledge is background only. Compare the model-knowledge answer against "
                    "the web evidence before writing the final answer. Never allow model knowledge "
                    "to override newer or more reliable web evidence. If local and web evidence conflict, say so "
                    "briefly instead of hiding the conflict. Preserve exact citation labels that "
                    "appear in useful source material, such as [S1] or [W1]. Cite local claims with "
                    "[S] labels and web claims with [W] labels. Do not cite model knowledge. Never "
                    "invent labels, URLs, or sources. Always give the best available direct answer "
                    "when the staged evidence reasonably supports one; refuse only if the available "
                    "evidence is clearly unrelated, contradictory, or would make the answer false. "
                    "Use up to five of the strongest web sources when web results are relevant. "
                    "For current events, public figures, recent elections, prices, laws, regulations, "
                    "company roles, sports results, or any other time-sensitive topic, use local or "
                    "web evidence as the source of truth; model knowledge may provide background "
                    "only. Do not state that a person currently holds a role based only on old, "
                    "event-based, or undated sources. If sources are old, "
                    "write 'as of the source date' or 'older sources say' and avoid present-tense "
                    "current-status claims. For current-information questions, validate evidence "
                    "before answering: prefer official government/organization sources, then "
                    "university or major news sources, then reference sources; require either one "
                    "official source or two independent credible sources; ignore archived, cached, "
                    "past-office, or old article evidence for current claims. If sources disagree, "
                    "state that evidence conflict was detected and select the newest high-authority "
                    "evidence. If the user has supplied a correction in the chat "
                    "history, treat that correction as important context unless cited sources "
                    "directly disprove it. Keep the answer direct and concise. For explanatory "
                    "concept answers, use concise Markdown sections when helpful: Definition, "
                    "Key Concepts, Applications, Example. "
                    f"{style_instruction}"
                ),
            }
        ]
        messages.extend(_history_messages(history, self.settings.max_history_turns))
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Local document context:\n"
                    f"{_local_context(local_sources) or 'No local source labels are available.'}\n\n"
                    f"Local answer:\n{local_answer}\n\n"
                    f"Model-knowledge answer:\n{model_answer}\n\n"
                    f"Web results:\n"
                    f"{_web_context(web_sources) or 'No web results are available.'}\n\n"
                    f"Current date: {date.today().isoformat()}\n\n"
                    "Write the final answer only. Prefer local facts when they directly answer the "
                    "question, and use web results to update or verify public/current information. "
                    "When web evidence disagrees with the model-knowledge answer, prefer the latest "
                    "reliable web source and cite it. "
                    "If snippets are enough to identify an answer, answer from those snippets with "
                    "web citations instead of saying you cannot answer. "
                    "Exclude claims from local or web snippets that do not explicitly mention the "
                    "person/entity being asked about."
                ),
            }
        )
        allowed_labels = [source.label for source in local_sources]
        allowed_labels.extend(source.label for source in web_sources)
        return _sanitize_citations(self._chat(messages), allowed_labels)

    def rewrite_query(self, question: str, history: Sequence[ChatMessage]) -> str:
        if not self.settings.hf_token.strip() or not history:
            return question
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's latest question as one concise standalone retrieval query "
                    "only if it clearly depends on the previous conversation. If the question is "
                    "already standalone or starts a new topic, return it unchanged. Do not connect "
                    "unrelated topics. Return only the query."
                ),
            }
        ]
        messages.extend(_history_messages(history, min(2, self.settings.max_history_turns)))
        messages.append({"role": "user", "content": question})
        try:
            rewritten = self._chat(messages, max_tokens=80, temperature=0.0).strip()
            return rewritten.strip('"') or question
        except Exception:
            return question

    def _chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        client = self._client()
        max_tokens = max_tokens or self.settings.hf_max_new_tokens
        temperature = self.settings.hf_temperature if temperature is None else temperature
        chat_errors: list[Exception] = []
        try:
            completion = client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = _completion_to_text(completion)
            if content:
                return content
        except Exception as exc:
            chat_errors.append(exc)
        try:
            completion = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = _completion_to_text(completion)
            if content:
                return content
        except Exception as exc:
            chat_errors.append(exc)

        if _should_try_text_generation(chat_errors):
            prompt = self._messages_to_prompt(messages)
            try:
                text = client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    return_full_text=False,
                )
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception as exc:
                chat_errors.append(exc)
        if chat_errors:
            raise GenerationError(_humanize_generation_error(chat_errors[-1]))
        raise GenerationError("Hugging Face returned an empty response.")

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        parts = []
        for message in messages:
            role = message["role"].upper()
            parts.append(f"{role}:\n{message['content']}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)


def _completion_to_text(completion) -> str:
    message = completion.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            value = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else None)
            if value:
                text_parts.append(str(value))
        return "\n".join(text_parts).strip()
    return ""


def _should_try_text_generation(errors: Sequence[Exception]) -> bool:
    blocked_markers = (
        "supported task: conversational",
        "chat completion",
        "chat-completion",
        "conversational",
    )
    for error in errors:
        text = str(error).lower()
        if any(marker in text for marker in blocked_markers):
            return False
    return True


def _sanitize_citations(text: str, allowed_labels: Sequence[str]) -> str:
    allowed = {f"[{label}]".lower() for label in allowed_labels}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.lower() in allowed else ""

    sanitized = CITATION_PATTERN.sub(replace, text or "")
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized)
    return sanitized.strip()


def _humanize_generation_error(error: Exception) -> str:
    text = str(error)
    lower = text.lower()
    if _is_model_selection_error(lower):
        return MODEL_SELECTION_WARNING
    if "401 unauthorized" in lower or "authentication" in lower:
        return (
            "The selected Hugging Face model could not be used with the current token or provider. "
            "Try another model, choose a compatible provider, or use a token with inference access."
        )
    if "supported task: conversational" in lower:
        return (
            "The selected Hugging Face model requires conversational routing that is not available with "
            "the current token or provider. Try another model or provider."
        )
    return text


def is_model_selection_warning(message: str) -> bool:
    return message.strip() == MODEL_SELECTION_WARNING


def _is_model_selection_error(lower_error: str) -> bool:
    markers = (
        "out of capacity",
        "capacity",
        "overloaded",
        "too busy",
        "currently loading",
        "temporarily unavailable",
        "service unavailable",
        "resource exhausted",
        "rate limit",
        "rate_limit",
        "quota",
        "429",
        "503",
        "no inference provider",
        "provider is not available",
        "model is not supported",
        "not supported for task",
        "supported task: conversational",
        "conversational routing",
    )
    return any(marker in lower_error for marker in markers)
