"""Conservative formula extraction and repair for local scientific documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


FORMULA_TYPES = {
    "linear_model",
    "probability",
    "likelihood",
    "posterior",
    "prior",
    "expectation",
    "variance",
    "integral",
    "summation",
    "differential_equation",
    "matrix_equation",
    "optimization",
    "identity",
    "unknown",
}

MATH_SYMBOLS = set("=≈∝≤≥<>+*/^_∑Σ∫√∂∇Π∞∀∈∉⊂⊆→⇒↦")
GREEK_SYMBOLS = set("αβγδεζηθκλμνξπρστυφχψωΑΒΓΔΕΖΗΘΚΛΜΝΞΠΡΣΤΥΦΧΨΩ")
ASCII_MATH_WORDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "lambda",
    "mu",
    "sigma",
    "omega",
    "pi",
    "sqrt",
    "sum",
    "integral",
    "log",
    "exp",
    "var",
    "cov",
    "corr",
    "mean",
    "median",
    "likelihood",
    "posterior",
    "prior",
    "probability",
    "density",
    "distribution",
    "gradient",
    "matrix",
    "vector",
}
GREEK_WORDS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "sigma": "σ",
    "omega": "ω",
    "pi": "π",
}
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
VARIABLE_RE = re.compile(
    r"(?<![A-Za-z])(?:[βθλμσπ][₀₁₂₃₄₅₆₇₈₉²³]?|[A-Za-zα-ωΑ-Ω][A-Za-z]?(?:[_^]?[0-9A-Za-z]+|[₀₁₂₃₄₅₆₇₈₉²³])?)"
)


@dataclass(slots=True)
class FormulaItem:
    formula_id: str
    document: str
    page: int | None
    raw_text: str
    repaired_text: str
    latex: str | None
    surrounding_text: str
    variables: dict[str, str] = field(default_factory=dict)
    formula_type: str | None = "unknown"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def formula_likelihood_score(text: str) -> float:
    """Score whether text is formula-like without relying on one domain."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return 0.0

    lower = cleaned.lower()
    token_count = max(1, len(re.findall(r"\w+", cleaned)))
    symbol_count = sum(1 for char in cleaned if char in MATH_SYMBOLS)
    greek_count = sum(1 for char in cleaned if char in GREEK_SYMBOLS)
    ascii_words = sum(1 for word in ASCII_MATH_WORDS if re.search(rf"\b{re.escape(word)}\b", lower))
    variables = {match.group(0) for match in VARIABLE_RE.finditer(cleaned)}
    score = 0.0

    if "=" in cleaned or "≈" in cleaned or "∝" in cleaned:
        score += 0.24
    if symbol_count:
        score += min(0.28, symbol_count * 0.055)
    if greek_count:
        score += min(0.22, greek_count * 0.06)
    if ascii_words:
        score += min(0.24, ascii_words * 0.055)
    if len(variables) >= 2 and symbol_count:
        score += 0.16
    if re.search(r"\b(?:p|f|g|h|E|Var|Cov|Pr)\s*\([^)]{1,80}\)", cleaned):
        score += 0.18
    if re.search(r"\b[a-zA-Z]\s*[_^]\s*[0-9A-Za-z]+|\b(?:x|y|b|B|beta|theta|lambda|sigma)[_ ]?\d\b", cleaned):
        score += 0.14
    if re.search(r"(?:∑|Σ|sum\s*\(|∫|integral|d\s*[a-zA-Z]\s*/\s*d\s*[a-zA-Z])", lower):
        score += 0.22

    density = symbol_count / max(1, len(cleaned))
    if token_count <= 18 and density >= 0.05:
        score += 0.12
    if re.search(r"\\(?:alpha|beta|theta|lambda|sigma|sum|int|sqrt)", cleaned):
        score += 0.18

    prose_penalty = 0.0
    if token_count > 35 and symbol_count < 2 and ascii_words < 2:
        prose_penalty += 0.2
    if cleaned.endswith(".") and token_count > 22 and "=" not in cleaned:
        prose_penalty += 0.12
    return max(0.0, min(1.0, score - prose_penalty))


def looks_like_formula(text: str, threshold: float = 0.55) -> bool:
    return formula_likelihood_score(text) >= threshold


def repair_formula_text(text: str, threshold: float = 0.55) -> str:
    """Repair only formula-like text, with no global OCR character rewrites."""
    if not looks_like_formula(text, threshold=threshold):
        return text
    repaired = text or ""
    repaired = repaired.replace("<=", "≤").replace(">=", "≥").replace("->", "→").replace("=>", "⇒")
    repaired = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK_WORDS.get(m.group(1).lower(), m.group(0)), repaired)

    for word, symbol in GREEK_WORDS.items():
        repaired = re.sub(
            rf"\b{word}[_ ]?([0-9]+)\b",
            lambda m, sym=symbol: _indexed_greek(sym, m.group(1)),
            repaired,
            flags=re.IGNORECASE,
        )
        repaired = re.sub(rf"\b{word}\b", symbol, repaired, flags=re.IGNORECASE)

    repaired = re.sub(r"\bsigma\s*\^\s*2\b", "σ²", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\bσ\s*2\b", "σ²", repaired)
    repaired = re.sub(r"\bb([0-9])\b", lambda m: f"b{m.group(1).translate(SUBSCRIPT_DIGITS)}", repaired)
    repaired = re.sub(r"\b([xy])_([0-9A-Za-z])\b", r"\1_\2", repaired)
    repaired = re.sub(r"\s*([=≈∝≤≥+*/])\s*", r" \1 ", repaired)
    repaired = re.sub(r"\s*([→⇒])\s*", r" \1 ", repaired)
    repaired = re.sub(r"\s+", " ", repaired)
    return repaired.strip()


def extract_formulas(
    text: str,
    *,
    document: str,
    page: int | None,
    threshold: float = 0.55,
) -> list[FormulaItem]:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    blocks: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if not looks_like_formula(line, threshold=threshold):
            index += 1
            continue
        start = index
        values = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if not next_line:
                break
            previous = values[-1]
            should_merge = (
                looks_like_formula(next_line, threshold=max(0.45, threshold - 0.1))
                or previous.endswith(("+", "-", "=", "∝", "/", "*", "(", "[", "{"))
            )
            if not should_merge:
                break
            values.append(next_line)
            index += 1
        blocks.append((start, index - 1, "\n".join(values)))
    items: list[FormulaItem] = []
    for block_index, (start, end, raw) in enumerate(blocks, start=1):
        surrounding = _surrounding_text(lines, start, end)
        repaired = repair_formula_text(raw, threshold=max(0.45, threshold - 0.1))
        formula_id = _formula_id(document, page, block_index, raw)
        items.append(
            FormulaItem(
                formula_id=formula_id,
                document=document,
                page=page,
                raw_text=raw,
                repaired_text=repaired,
                latex=_latex_if_present(raw),
                surrounding_text=surrounding,
                variables=extract_formula_variables(repaired, surrounding),
                formula_type=classify_formula_type(repaired, surrounding),
                confidence=formula_likelihood_score(raw),
                metadata={"block_index": block_index},
            )
        )
    return items


def extract_formula_variables(formula_text: str, surrounding_text: str) -> dict[str, str]:
    variables = _formula_variables(formula_text)
    definitions = _nearby_variable_definitions(surrounding_text)
    return {variable: definitions.get(variable, "not defined in nearby text") for variable in variables}


def classify_formula_type(formula_text: str, surrounding_text: str = "") -> str:
    text = f"{formula_text} {surrounding_text}".lower()
    if re.search(r"\by\s*[=~]\s*|β|beta|b₀|b₁", text):
        return "linear_model"
    if "posterior" in text or re.search(r"p\s*\(.+\|.+\)", text):
        return "posterior"
    if "likelihood" in text or re.search(r"\bl\s*\(", text):
        return "likelihood"
    if "prior" in text:
        return "prior"
    if "var" in text or "σ²" in formula_text or "sigma^2" in text:
        return "variance"
    if re.search(r"\b(?:e|expectation)\s*\(", text):
        return "expectation"
    if "∫" in formula_text or "integral" in text:
        return "integral"
    if "∑" in formula_text or "Σ" in formula_text or "sum" in text:
        return "summation"
    if "argmin" in text or "argmax" in text or "minimize" in text:
        return "optimization"
    if re.search(r"\b(matrix|vector)\b|\[[^\]]+;[^\]]+\]", text):
        return "matrix_equation"
    if re.search(r"\bd\s*[a-z]\s*/\s*d\s*[a-z]", text):
        return "differential_equation"
    if re.search(r"\bp\s*\(", text) or "probability" in text or "density" in text:
        return "probability"
    if "=" in formula_text:
        return "identity"
    return "unknown"


def formula_to_text(item: FormulaItem) -> str:
    variables = "; ".join(f"{name}: {meaning}" for name, meaning in item.variables.items())
    parts = [
        f"Formula: {item.repaired_text or item.raw_text}",
        f"Raw formula: {item.raw_text}",
        f"Type: {item.formula_type or 'unknown'}",
        f"Variables: {variables}" if variables else "",
        f"Surrounding text: {item.surrounding_text}",
    ]
    return "\n".join(part for part in parts if part.strip())


def _indexed_greek(symbol: str, digits: str) -> str:
    if symbol == "σ" and digits == "2":
        return "σ²"
    return f"{symbol}{digits.translate(SUBSCRIPT_DIGITS)}"


def _formula_variables(formula_text: str) -> list[str]:
    values = []
    for match in VARIABLE_RE.finditer(formula_text or ""):
        value = match.group(0).strip()
        if len(value) > 12 or value.lower() in ASCII_MATH_WORDS:
            continue
        if value not in values:
            values.append(value)
    return values[:16]


def _nearby_variable_definitions(text: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    normalized = re.sub(r"\s+", " ", text or "")
    for match in re.finditer(
        r"(?P<var>[A-Za-zα-ωΑ-Ωβθλμσπ][A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉²³]*)\s+"
        r"(?:is|denotes|represents|means|refers to)\s+(?P<meaning>[^.;,\n]{3,140})",
        normalized,
        flags=re.IGNORECASE,
    ):
        meaning = re.split(
            r"\s+and\s+[A-Za-zα-ωΑ-Ωβθλμσπ][A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉²³]*\s+"
            r"(?:is|denotes|represents|means|refers to)\s+",
            match.group("meaning").strip(),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        definitions[match.group("var")] = meaning
    where_match = re.search(r"\bwhere\s+(?P<body>[^.]{3,260})", normalized, flags=re.IGNORECASE)
    if where_match:
        for piece in re.split(r",| and ", where_match.group("body")):
            submatch = re.match(
                r"\s*(?P<var>[A-Za-zα-ωΑ-Ωβθλμσπ][A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉²³]*)\s+"
                r"(?:is|denotes|represents)?\s*(?P<meaning>[^,;]{3,100})",
                piece.strip(),
                flags=re.IGNORECASE,
            )
            if submatch:
                definitions.setdefault(submatch.group("var"), submatch.group("meaning").strip())
    return definitions


def _surrounding_text(lines: list[str], start: int, end: int, radius: int = 2) -> str:
    selected = []
    for line in lines[max(0, start - radius) : min(len(lines), end + radius + 1)]:
        stripped = line.strip()
        if stripped:
            selected.append(stripped)
    return " ".join(selected)


def _latex_if_present(text: str) -> str | None:
    return text if re.search(r"\\(?:alpha|beta|theta|lambda|sigma|sum|int|sqrt)|[_^]", text or "") else None


def _formula_id(document: str, page: int | None, index: int, raw_text: str) -> str:
    digest = hashlib.blake2b(f"{document}:{page}:{index}:{raw_text}".encode("utf-8"), digest_size=10)
    return digest.hexdigest()
