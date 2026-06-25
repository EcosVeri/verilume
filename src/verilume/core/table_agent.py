"""Safe pandas calculations for retrieved local tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from verilume.core.table_store import TableMetadata


@dataclass(frozen=True, slots=True)
class TableAnswer:
    answer: str
    calculation: str
    columns_used: list[str]
    result: float | int
    citation: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TableAgent:
    def answer_with_pandas(
        self,
        question: str,
        df: pd.DataFrame,
        *,
        metadata: TableMetadata,
        citation_label: str = "S1",
    ) -> TableAnswer:
        operation = _operation(question)
        column = _best_numeric_column(question, df)
        if column is None:
            raise ValueError("No numeric column matched this table question.")

        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            raise ValueError(f"Column {column} does not contain numeric values.")

        result = _calculate(operation, numeric)
        calculation = f"{operation}({column})"
        result_text = _format_number(result)
        answer = (
            f"The {operation} of `{column}` is {result_text} [{citation_label}].\n\n"
            f"Calculation: `{calculation}`\n\n"
            f"Source table: {metadata.document}"
        )
        return TableAnswer(
            answer=answer,
            calculation=calculation,
            columns_used=[str(column)],
            result=result,
            citation=citation_label,
            diagnostics={
                "table_id": metadata.table_id,
                "document": metadata.document,
                "operation": operation,
                "row_count": metadata.row_count,
            },
        )


def _operation(question: str) -> str:
    normalized = (question or "").lower()
    if re.search(r"\b(?:average|mean)\b", normalized):
        return "mean"
    if re.search(r"\b(?:sum|total)\b", normalized):
        return "sum"
    if re.search(r"\b(?:maximum|max|highest|largest)\b", normalized):
        return "max"
    if re.search(r"\b(?:minimum|min|lowest|smallest)\b", normalized):
        return "min"
    if re.search(r"\bmedian\b", normalized):
        return "median"
    if re.search(r"\b(?:count|how many)\b", normalized):
        return "count"
    return "mean"


def _best_numeric_column(question: str, df: pd.DataFrame) -> str | None:
    numeric_columns = list(df.select_dtypes(include="number").columns)
    if not numeric_columns:
        for column in df.columns:
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().any():
                numeric_columns.append(column)
    if not numeric_columns:
        return None

    question_terms = _terms(question)
    for column in numeric_columns:
        column_terms = _terms(str(column).replace("_", " "))
        if question_terms & column_terms:
            return str(column)
    return str(numeric_columns[0])


def _calculate(operation: str, series: pd.Series) -> float | int:
    if operation == "sum":
        return float(series.sum())
    if operation == "max":
        return float(series.max())
    if operation == "min":
        return float(series.min())
    if operation == "median":
        return float(series.median())
    if operation == "count":
        return int(series.count())
    return float(series.mean())


def _format_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", (text or "").lower())
        if len(token) > 1
    }
