"""Strict source-use policy derived from the selected search mode."""

from __future__ import annotations

from dataclasses import dataclass

from verilume.core.search_modes import SearchMode, search_mode_from_settings


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    mode: SearchMode
    use_local: bool
    use_ai: bool
    use_web: bool
    ai_as_evidence: bool
    benchmark_allowed: bool
    reason: str

    @property
    def sources_searched(self) -> list[str]:
        sources: list[str] = []
        if self.use_local:
            sources.append("local")
        if self.use_ai:
            sources.append("ai")
        if self.use_web:
            sources.append("web")
        return sources


def policy_for_mode(
    mode: SearchMode | str,
    *,
    web_enabled: bool,
    current_or_dynamic: bool,
) -> SearchPolicy:
    mode = search_mode_from_settings(mode)

    if mode == SearchMode.LOCAL_ONLY:
        return SearchPolicy(mode, True, False, False, False, True, "Local Only selected.")

    if mode == SearchMode.LOCAL_AI:
        return SearchPolicy(
            mode,
            True,
            True,
            False,
            not current_or_dynamic,
            True,
            "Local + AI selected.",
        )

    if mode == SearchMode.LOCAL_AI_WEB:
        return SearchPolicy(
            mode,
            True,
            not current_or_dynamic,
            web_enabled,
            not current_or_dynamic,
            True,
            "Local + AI + Web selected.",
        )

    if mode == SearchMode.WEB_ONLY:
        return SearchPolicy(mode, False, False, web_enabled, False, True, "Web Only selected.")

    if mode == SearchMode.RESEARCH:
        return SearchPolicy(
            mode,
            True,
            not current_or_dynamic,
            web_enabled,
            not current_or_dynamic,
            True,
            "Research Mode selected.",
        )

    if mode == SearchMode.AI_ONLY:
        return SearchPolicy(
            mode,
            False,
            True,
            False,
            not current_or_dynamic,
            True,
            "AI Only benchmark strategy selected.",
        )

    return SearchPolicy(
        mode,
        True,
        not current_or_dynamic,
        web_enabled,
        not current_or_dynamic,
        True,
        "Auto selected: search all allowed sources and rank the best evidence.",
    )
