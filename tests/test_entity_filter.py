from __future__ import annotations

from verilume.core.entity_filter import is_short_entity_query, source_matches_entity


def test_short_entity_query_detects_name_like_prompts() -> None:
    assert is_short_entity_query("Rene") is True
    assert is_short_entity_query("Christophe Ley") is True
    assert is_short_entity_query("Gabriella Vinco") is True


def test_short_entity_query_ignores_lowercase_concepts_and_questions() -> None:
    assert is_short_entity_query("photonics") is False
    assert is_short_entity_query("what is photonics") is False
    assert is_short_entity_query("Christophe Ley?") is False


def test_source_matches_entity_requires_exact_single_word_match() -> None:
    assert source_matches_entity("Rene", "The author is Rene Muller.") is True
    assert source_matches_entity("Rene", "General Sproochentest preparation pages.") is False


def test_source_matches_entity_allows_most_multi_name_parts() -> None:
    assert source_matches_entity("Christophe Ley", "Professor Ley works on directional statistics.") is True
    assert source_matches_entity("Christophe Ley", "Gabriella Vinco is a doctoral researcher.") is False
