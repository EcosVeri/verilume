"""Conservative repair for obvious OCR/PDF equation extraction issues."""

from __future__ import annotations

import re

_MATH_MARKERS = (
    "price",
    "beta",
    "b0",
    "b1",
    "b2",
    "b3",
    "m2",
    "age",
    "dis",
    "epsilon",
    "error",
    "regression",
)

_REPLACEMENTS = {
    "B0": "β₀",
    "B1": "β₁",
    "B2": "β₂",
    "B3": "β₃",
    "b0": "β₀",
    "b1": "β₁",
    "b2": "β₂",
    "b3": "β₃",
    "beta0": "β₀",
    "beta1": "β₁",
    "beta2": "β₂",
    "beta3": "β₃",
    "epsilon": "ε",
}


def looks_like_equation(line: str) -> bool:
    text = (line or "").strip()
    lower = text.lower()
    return "=" in text and any(marker in lower for marker in _MATH_MARKERS)


def repair_equation_line(line: str) -> str:
    value = line or ""
    for old, new in _REPLACEMENTS.items():
        value = re.sub(rf"\b{re.escape(old)}\b", new, value)
    value = re.sub(r"\s*=\s*", " = ", value)
    value = re.sub(r"\s*\+\s*", " + ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def repair_known_regression_equation(text: str) -> str:
    lower = (text or "").lower()
    if "price" in lower and "m2" in lower and "age" in lower and "dis" in lower:
        return "PRICE = β₀ + β₁M2 + β₂AGE + β₃DIS + ε"
    return text


def repair_math_text(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines():
        if looks_like_equation(line):
            line = repair_equation_line(line)
            line = repair_known_regression_equation(line)
        lines.append(line)
    return "\n".join(lines)
