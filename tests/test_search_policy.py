from __future__ import annotations

from verilume.core.search_modes import SearchMode, search_mode_from_settings
from verilume.core.search_policy import policy_for_mode


def test_local_only_policy_blocks_ai_and_web() -> None:
    policy = policy_for_mode(SearchMode.LOCAL_ONLY, web_enabled=True, current_or_dynamic=False)

    assert policy.use_local is True
    assert policy.use_ai is False
    assert policy.use_web is False
    assert policy.sources_searched == ["local"]


def test_auto_static_policy_uses_all_enabled_sources() -> None:
    policy = policy_for_mode(SearchMode.AUTO, web_enabled=True, current_or_dynamic=False)

    assert policy.use_local is True
    assert policy.use_ai is True
    assert policy.use_web is True
    assert policy.ai_as_evidence is True


def test_auto_current_policy_demotes_ai() -> None:
    policy = policy_for_mode(SearchMode.AUTO, web_enabled=True, current_or_dynamic=True)

    assert policy.use_local is True
    assert policy.use_ai is False
    assert policy.use_web is True
    assert policy.ai_as_evidence is False


def test_web_only_policy_blocks_local_and_ai() -> None:
    policy = policy_for_mode(SearchMode.WEB_ONLY, web_enabled=True, current_or_dynamic=False)

    assert policy.use_local is False
    assert policy.use_ai is False
    assert policy.use_web is True


def test_search_mode_parser_accepts_existing_labels() -> None:
    assert search_mode_from_settings("Local + AI + Web") == SearchMode.LOCAL_AI_WEB
    assert search_mode_from_settings("research_mode") == SearchMode.RESEARCH
    assert search_mode_from_settings("web only") == SearchMode.WEB_ONLY
