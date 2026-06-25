from __future__ import annotations

from verilume.core.equation_repair import (
    looks_like_equation,
    repair_equation_line,
    repair_known_regression_equation,
    repair_math_text,
)


def test_looks_like_equation_is_conservative() -> None:
    assert looks_like_equation("PRICE = B0 + B1M2 + B2AGE + B3DIS + epsilon") is True
    assert looks_like_equation("This sentence mentions price but has no equation.") is False


def test_repair_known_regression_equation() -> None:
    repaired = repair_known_regression_equation("PRICE = B0 + B1M2 + B2AGE + B3DIS")

    assert repaired == "PRICE = β₀ + β₁M2 + β₂AGE + β₃DIS + ε"


def test_repair_equation_line_normalizes_beta_and_epsilon() -> None:
    repaired = repair_equation_line("PRICE=B0+B1 M2+B2 AGE+B3 DIS+epsilon")

    assert "β₀" in repaired
    assert "β₁" in repaired
    assert "ε" in repaired
    assert " = " in repaired


def test_repair_math_text_leaves_ordinary_text_unchanged() -> None:
    text = "Regression is useful.\nPRICE = B0 + B1M2 + B2AGE + B3DIS"
    repaired = repair_math_text(text)

    assert repaired.splitlines()[0] == "Regression is useful."
    assert repaired.splitlines()[1] == "PRICE = β₀ + β₁M2 + β₂AGE + β₃DIS + ε"
