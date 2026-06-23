from __future__ import annotations

from verilume.core.query_preprocessing import (
    are_queries_semantically_similar,
    normalize_query,
    query_variants,
)


def test_area_query_variants_normalize_to_same_canonical_form() -> None:
    first = normalize_query("What is the size of Cameroon?")
    second = normalize_query("Total area of Cameroon")

    assert first.canonical == "area cameroon"
    assert second.canonical == "area cameroon"
    assert are_queries_semantically_similar("What is the size of Cameroon?", "How big is Cameroon")


def test_normalizer_keeps_different_attributes_apart() -> None:
    assert not are_queries_semantically_similar("Size of Cameroon", "Population of Cameroon")
    assert normalize_query("Size of Africa").entities == ("Africa",)


def test_query_variants_include_factoid_search_forms() -> None:
    variants = query_variants("Size of Africa")

    assert "area africa" in variants
    assert any("area square kilometers" in variant for variant in variants)
