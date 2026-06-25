from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from verilume.core.formula_extraction import (
    FormulaItem,
    extract_formula_variables,
    formula_likelihood_score,
    looks_like_formula,
    repair_formula_text,
)
from verilume.core.formula_retrieval import FormulaRetriever
from verilume.core.formula_store import FormulaStore
from verilume.core.structured_document_store import StructuredDocumentStore
from verilume.core.structured_ocr import StructuredDocument, StructuredField, extract_structured_document
from verilume.core.structured_retrieval import StructuredRetriever
from verilume.rag import VerilumeRAG
from verilume.settings import AppSettings


class EmptyRetriever:
    def __init__(self) -> None:
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return []

    def count(self) -> int:
        return 0


def test_formula_detection_and_conservative_repair() -> None:
    text = "y = beta0 + beta1 x_i + epsilon"

    assert formula_likelihood_score(text) >= 0.55
    assert looks_like_formula(text)
    repaired = repair_formula_text(text)

    assert "β₀" in repaired
    assert "β₁" in repaired
    assert "ε" in repaired
    assert repair_formula_text("The beta release was delayed.") == "The beta release was delayed."


def test_formula_variable_extraction_uses_nearby_text() -> None:
    variables = extract_formula_variables(
        "y = β₀ + β₁ x + ε",
        "where β₀ is the intercept and ε is the error term.",
    )

    assert variables["β₀"] == "the intercept"
    assert variables["ε"] == "the error term"


def test_formula_store_and_retriever_return_formula_sources() -> None:
    with TemporaryDirectory() as tmp:
        store = FormulaStore(Path(tmp) / "formulas.sqlite")
        store.add_formula(
            FormulaItem(
                formula_id="f1",
                document="regression.pdf",
                page=3,
                raw_text="y = beta0 + beta1 x + epsilon",
                repaired_text="y = β₀ + β₁ x + ε",
                latex=None,
                surrounding_text="β₀ is the intercept.",
                variables={"β₀": "the intercept"},
                formula_type="linear_model",
                confidence=0.91,
                metadata={},
            )
        )

        sources = FormulaRetriever(store).retrieve("What equation is used for the model?")

    assert len(sources) == 1
    assert sources[0].document == "regression.pdf"
    assert sources[0].metadata["content_type"] == "formula"
    assert "β₀" in sources[0].text


def test_structured_extraction_is_generic_label_value() -> None:
    structured = extract_structured_document(
        "Certificate\nReference: CERT-2026-445\nIssue Date: 25/06/2026\nTotal: EUR 42.00",
        document="certificate.pdf",
        page=1,
    )

    assert structured is not None
    assert structured.document_type in {"certificate", "form"}
    values = {field.canonical_name: field.value for field in structured.fields}
    assert values["reference_number"] == "CERT-2026-445"
    assert values["issue_date"] == "25/06/2026"
    assert values["amount"] == "EUR 42.00"


def test_structured_store_and_retriever_return_field_sources() -> None:
    with TemporaryDirectory() as tmp:
        store = StructuredDocumentStore(Path(tmp) / "structured.sqlite")
        store.add_structured_document(
            StructuredDocument(
                document_id="doc1",
                document="form.pdf",
                page=1,
                document_type="form",
                confidence=0.9,
                fields=[
                    StructuredField(
                        field_name="Reference",
                        canonical_name="document_number",
                        value="ABC-12345",
                        raw_label="Reference",
                        raw_value="ABC-12345",
                        field_type="document_number",
                        confidence=0.88,
                        page=1,
                    )
                ],
            )
        )

        sources = StructuredRetriever(store).retrieve("What is the document number?")

    assert len(sources) == 1
    assert sources[0].document == "form.pdf"
    assert sources[0].metadata["content_type"] == "structured_field"
    assert "ABC-12345" in sources[0].text


def test_rag_answers_formula_question_from_formula_store() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = AppSettings(
            docs_dir=root / "docs",
            chroma_dir=root / "chroma",
            manifest_path=root / "manifest.json",
            formula_store_path=root / "formulas.sqlite",
            ocr_block_store_path=root / "ocr.sqlite",
            structured_document_store_path=root / "structured.sqlite",
            enable_web_search=False,
            semantic_cache_enabled=False,
            enable_graphrag=False,
        )
        rag = VerilumeRAG(settings)
        rag.retriever = EmptyRetriever()
        rag.formula_store.add_formula(
            FormulaItem(
                formula_id="f1",
                document="regression.pdf",
                page=3,
                raw_text="y = beta0 + beta1 x + epsilon",
                repaired_text="y = β₀ + β₁ x + ε",
                latex=None,
                surrounding_text="β₀ is the intercept.",
                variables={"β₀": "the intercept"},
                formula_type="linear_model",
                confidence=0.92,
                metadata={},
            )
        )

        result = rag.ask("What equation is used for the model?")

    assert result.confidence == "local-grounded"
    assert result.local_sources[0].metadata["content_type"] == "formula"
    assert "Formula:" in result.answer
    assert "β₀" in result.answer


def test_rag_answers_structured_question_from_structured_store() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = AppSettings(
            docs_dir=root / "docs",
            chroma_dir=root / "chroma",
            manifest_path=root / "manifest.json",
            formula_store_path=root / "formulas.sqlite",
            ocr_block_store_path=root / "ocr.sqlite",
            structured_document_store_path=root / "structured.sqlite",
            enable_web_search=False,
            semantic_cache_enabled=False,
            enable_graphrag=False,
        )
        rag = VerilumeRAG(settings)
        rag.retriever = EmptyRetriever()
        rag.structured_store.add_structured_document(
            StructuredDocument(
                document_id="doc1",
                document="form.pdf",
                page=1,
                document_type="form",
                confidence=0.9,
                fields=[
                    StructuredField(
                        field_name="Document Number",
                        canonical_name="document_number",
                        value="ABC-12345",
                        raw_label="Document Number",
                        raw_value="ABC-12345",
                        field_type="document_number",
                        confidence=0.9,
                        page=1,
                    )
                ],
            )
        )

        result = rag.ask("What is the document number?")

    assert result.confidence == "local-grounded"
    assert result.local_sources[0].metadata["content_type"] == "structured_field"
    assert "ABC-12345" in result.answer
