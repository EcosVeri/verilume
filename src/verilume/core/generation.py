"""Generation backends for Verilume.

Supports:
- Hugging Face Inference Providers
- Ollama local models
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import requests
from huggingface_hub import InferenceClient

from verilume.core.schemas import ChatMessage, LocalSource, WebSource
from verilume.settings import ANSWER_STYLE_PROFILES, AppSettings, AnswerStyleProfile


LOCAL_UNKNOWN = "LOCAL_UNKNOWN"
MODEL_UNKNOWN = "MODEL_UNKNOWN"
MODEL_SELECTION_WARNING = (
    "Select another model. The selected model is not supported for chat generation."
)


MODEL_SELECTION_ERROR_MARKERS = (
    "not a chat model",
    "model_not_supported",
    "currently loading",
    "temporarily unavailable",
    "provider mapping",
    "not supported",
    "select another model",
    "bad request",
    "404",
    "503",
)


class GenerationError(RuntimeError):
    """Raised when a generation backend fails."""


def is_model_selection_warning(message: str) -> bool:
    lower = (message or "").lower()
    return any(marker in lower for marker in MODEL_SELECTION_ERROR_MARKERS)


class BaseGenerator(ABC):
    """Common interface for all generation backends."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @property
    def answer_style_profile(self) -> AnswerStyleProfile:
        return ANSWER_STYLE_PROFILES.get(
            self.settings.answer_style,
            ANSWER_STYLE_PROFILES["Standard"],
        )

    @property
    def style_instruction(self) -> str:
        return self.answer_style_profile.style_instruction

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from chat messages."""

    def rewrite_query(
        self,
        question: str,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
    ) -> str:
        if not history:
            return question

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question into a standalone search query. "
                    "Only rewrite if the question depends on prior context. "
                    "If it is already standalone, return it unchanged. "
                    "Do not answer the question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{_format_history(history)}\n\n"
                    f"Current question:\n{question}\n\n"
                    "Standalone question:"
                ),
            },
        ]

        try:
            rewritten = self.chat(messages).strip()
        except GenerationError:
            return question

        return rewritten or question

    def answer_local(
        self,
        question: str,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
        local_sources: list[LocalSource],
    ) -> str:
        if not local_sources:
            return LOCAL_UNKNOWN

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Verilume, a careful local retrieval assistant. "
                    "Answer using ONLY the provided local document context. "
                    "Cite local sources using [S1], [S2], etc. "
                    f"If the answer is not clearly present, say exactly: {LOCAL_UNKNOWN}. "
                    "Do not use general knowledge. Do not guess. "
                    f"Style: {self.style_instruction}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{_format_history(history)}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Local document context:\n{_format_local_sources(local_sources)}"
                ),
            },
        ]

        return self.chat(messages).strip()

    def answer_model_knowledge(
        self,
        question: str,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Verilume using AI knowledge only. "
                    "Do not use web search. "
                    "You may answer stable general knowledge, definitions, scientific explanations, "
                    "math, coding, geography, and historical facts from model knowledge. "
                    "If the question asks for current or changing information such as current office holders, "
                    "recent events, latest prices, live schedules, current laws, regulations, or news, "
                    "do not present the answer as verified current fact. Provide only stable background if useful "
                    "and clearly say that current verification requires web or source evidence. "
                    f"If you cannot provide even stable background, say exactly: {MODEL_UNKNOWN}. "
                    "If you are unsure, say exactly the same. "
                    f"Style: {self.style_instruction}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{_format_history(history)}\n\nQuestion:\n{question}"
                ),
            },
        ]

        return self.chat(messages).strip()

    def answer_final(
        self,
        question: str,
        history: list[ChatMessage] | tuple[ChatMessage, ...],
        local_answer: str,
        model_answer: str,
        local_sources: list[LocalSource],
        web_sources: list[WebSource],
    ) -> str:
        has_web = bool(web_sources)

        system = (
            "You are Verilume, an evidence-first AI assistant. "
            "Use the available evidence in this priority order: "
            "local files first, web evidence second, AI knowledge last. "
            "For stable non-current questions, combine useful AI knowledge with local file evidence when both are available, "
            "but give more weight to local files and never override local facts with AI knowledge. "
            "When web evidence is available for a stable non-current question, combine the web evidence with AI knowledge "
            "and cite the web evidence for externally sourced claims. "
            "For current, recent, live, or otherwise changeable information, use web evidence as the source of truth and do not rely on AI knowledge. "
            "Never let AI knowledge override local files or newer web evidence. "
            "Cite local sources as [S1], [S2]. "
            "Cite web sources as [W1], [W2]. "
            "Do not invent citations. "
            "If sources disagree, mention the conflict and prefer the newest high-authority evidence. "
            f"Style: {self.style_instruction}"
        )

        if has_web:
            system += (
                " Web evidence is available. For stable questions, synthesize it with AI knowledge. "
                "For current or changeable questions, keep the answer web-grounded."
            )

        messages = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{_format_history(history)}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Local answer:\n{local_answer}\n\n"
                    f"AI knowledge answer:\n{model_answer}\n\n"
                    f"Local sources:\n{_format_local_sources(local_sources)}\n\n"
                    f"Web sources:\n{_format_web_sources(web_sources)}\n\n"
                    "Write the final answer only."
                ),
            },
        ]

        return self.chat(messages).strip()


class HuggingFaceGenerator(BaseGenerator):
    """Hugging Face Inference Providers chat backend."""

    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings)
        self.client = InferenceClient(
            provider=settings.hf_provider,
            token=settings.hf_token or None,
            timeout=settings.hf_timeout_seconds,
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.settings.hf_token:
            raise GenerationError(
                "Hugging Face token is missing. Add a valid token or switch to Ollama."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.settings.hf_llm_model,
                messages=messages,
                max_tokens=_style_max_tokens(
                    self.settings.hf_max_new_tokens,
                    self.answer_style_profile,
                ),
                temperature=_style_temperature(
                    self.settings.hf_temperature,
                    self.answer_style_profile,
                ),
            )

            content = response.choices[0].message.content

            if not content:
                raise GenerationError("The Hugging Face model returned an empty response.")

            return str(content)

        except Exception as exc:
            message = _clean_error(exc)

            if is_model_selection_warning(message):
                raise GenerationError(
                    "The selected Hugging Face model could not generate a response. "
                    "Try Qwen/Qwen2.5-7B-Instruct, verify your token, or use a custom chat model. "
                    f"Details: {message}"
                ) from exc

            raise GenerationError(message) from exc


class OllamaGenerator(BaseGenerator):
    """Ollama local chat backend."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"

        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": _style_temperature(
                    self.settings.ollama_temperature,
                    self.answer_style_profile,
                ),
                "num_predict": _style_max_tokens(
                    self.settings.ollama_num_predict,
                    self.answer_style_profile,
                ),
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.settings.ollama_timeout_seconds,
            )
            response.raise_for_status()

            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "")

            if not content:
                raise GenerationError(
                    "Ollama returned an empty response. Check that the selected model is installed."
                )

            return str(content)

        except requests.exceptions.ConnectionError as exc:
            raise GenerationError(
                "Could not connect to Ollama. Make sure Ollama is running locally, "
                "then run `ollama serve` or open the Ollama app."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            detail = _response_text(exc.response)
            raise GenerationError(
                "Ollama request failed. "
                f"Check that model `{self.settings.ollama_model}` is installed. "
                f"You can install it with: `ollama pull {self.settings.ollama_model}`. "
                f"Details: {detail}"
            ) from exc

        except Exception as exc:
            raise GenerationError(_clean_error(exc)) from exc


class GeneratorRegistry:
    """Registry for built-in and third-party generation backends."""

    _generators: dict[str, type[BaseGenerator]] = {}

    @classmethod
    def register(cls, name: str, generator_class: type[BaseGenerator]) -> None:
        key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            raise ValueError("Generator backend name cannot be empty.")
        if not issubclass(generator_class, BaseGenerator):
            raise TypeError("generator_class must inherit from BaseGenerator.")
        cls._generators[key] = generator_class

    @classmethod
    def create(cls, settings: AppSettings) -> BaseGenerator:
        generator_class = cls._generators.get(settings.generation_backend)
        if generator_class is None:
            generator_class = cls._generators["huggingface"]
        return generator_class(settings)

    @classmethod
    def list_backends(cls) -> list[str]:
        return sorted(cls._generators)


def create_generator(settings: AppSettings) -> BaseGenerator:
    """Create the selected generation backend."""

    return GeneratorRegistry.create(settings)


GeneratorRegistry.register("huggingface", HuggingFaceGenerator)
GeneratorRegistry.register("ollama", OllamaGenerator)


# Backward-compatible name for existing rag.py imports
# You can later replace HuggingFaceGenerator(settings) with create_generator(settings).
# For full backend switching, rag.py should use:
#
#     self.generator = create_generator(settings)


def _format_history(
    history: list[ChatMessage] | tuple[ChatMessage, ...],
    max_chars: int = 4000,
) -> str:
    if not history:
        return "No previous conversation."

    lines = []

    for item in history[-8:]:
        role = item.role
        content = item.content.strip()

        if not content:
            continue

        lines.append(f"{role}: {content}")

    text = "\n".join(lines)

    if len(text) > max_chars:
        return text[-max_chars:]

    return text


def _format_local_sources(
    sources: list[LocalSource],
    max_chars_per_source: int = 1400,
) -> str:
    if not sources:
        return "No local sources."

    blocks = []

    for source in sources:
        page = f", page {source.page}" if source.page else ""
        text = source.text.strip()

        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "..."

        blocks.append(f"[{source.label}] {source.document}{page}\n{text}")

    return "\n\n".join(blocks)


def _format_web_sources(
    sources: list[WebSource],
    max_chars_per_source: int = 1400,
) -> str:
    if not sources:
        return "No web sources."

    blocks = []

    for source in sources:
        text = source.content.strip()

        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "..."

        metadata = ""

        if source.published_date:
            metadata += f"\nPublished date: {source.published_date}"

        visible_dates = source.metadata.get("visible_dates") if source.metadata else None

        if visible_dates:
            metadata += f"\nVisible dates: {visible_dates}"

        blocks.append(
            f"[{source.label}] {source.title}\nURL: {source.url}{metadata}\nContent: {text}"
        )

    return "\n\n".join(blocks)


def _clean_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())

    try:
        payload = json.loads(message)
        return json.dumps(payload, indent=2)
    except Exception:
        return message[:1000]


def _style_max_tokens(configured: int, profile: AnswerStyleProfile) -> int:
    configured = max(32, int(configured))
    if profile.verbosity == "brief":
        return min(configured, profile.max_tokens)
    return max(configured, profile.max_tokens)


def _style_temperature(configured: float, profile: AnswerStyleProfile) -> float:
    configured = max(0.0, float(configured))
    if profile.verbosity == "brief":
        return min(configured, profile.temperature)
    return max(configured, profile.temperature)


def _response_text(response: Any) -> str:
    if response is None:
        return ""

    try:
        return response.text[:1000]
    except Exception:
        return ""
