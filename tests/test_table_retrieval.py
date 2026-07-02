from __future__ import annotations

import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from verilume.core.table_agent import TableAgent
from verilume.core.table_retrieval import TableRetrieval
from verilume.core.table_store import TableStore
from verilume.rag import VerilumeRAG
from verilume.settings import AppSettings


CSV_TEXT = """city,price,size_m2,rooms
Luxembourg,650000,70,2
Esch,420000,65,2
Differdange,390000,60,1
Luxembourg,720000,80,3
"""


class TableRetrievalTests(unittest.TestCase):
    def test_table_store_indexes_csv_and_retrieves_by_column(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "apartments.csv").write_text(CSV_TEXT, encoding="utf-8")
            store = TableStore(Path(tmp) / "tables")

            indexed = store.index_local_tables(docs)
            matches = store.search_tables("What is the average price?")

            self.assertEqual(len(indexed), 1)
            self.assertEqual(matches[0].document, "apartments.csv")
            self.assertIn("price", matches[0].columns)

    def test_delete_document_removes_tables_and_frame_files(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TableStore(Path(tmp) / "tables")
            metadata = store.add_table(pd.read_csv(StringIO(CSV_TEXT)), document="apartments.csv")
            self.assertIn("apartments.csv", store.documents())
            self.assertTrue(metadata.dataframe_path.exists())

            store.delete_document("apartments.csv")

            self.assertNotIn("apartments.csv", store.documents())
            self.assertEqual(store.list_tables(), [])
            self.assertFalse(metadata.dataframe_path.exists())

    def test_table_agent_computes_average_without_inventing_values(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TableStore(Path(tmp) / "tables")
            metadata = store.add_table(
                pd.read_csv(StringIO(CSV_TEXT)),
                document="apartments.csv",
            )
            df = store.load_table(metadata.table_id)

            answer = TableAgent().answer_with_pandas(
                "What is the average price?",
                df,
                metadata=metadata,
            )

            self.assertEqual(answer.calculation, "mean(price)")
            self.assertEqual(answer.result, 545000)
            self.assertIn("545,000", answer.answer)

    def test_table_retrieval_finds_best_table(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TableStore(Path(tmp) / "tables")
            store.add_table(pd.read_csv(StringIO(CSV_TEXT)), document="apartments.csv")

            table = TableRetrieval(store).find_best_table("average apartment price")

            self.assertIsNotNone(table)
            self.assertEqual(table.document, "apartments.csv")

    def test_rag_answers_table_question_from_local_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "apartments.csv").write_text(CSV_TEXT, encoding="utf-8")
            settings = AppSettings(
                docs_dir=docs,
                chroma_dir=Path(tmp) / "chroma",
                manifest_path=Path(tmp) / "manifest.json",
                table_store_dir=Path(tmp) / "tables",
                semantic_cache_enabled=False,
                enable_web_search=False,
                hf_token="token",
            )
            rag = VerilumeRAG(settings)

            response = rag.ask("What is the average price in this CSV?")

            self.assertIn("545,000", response.answer)
            self.assertEqual(response.diagnostics["table_calculation"], "mean(price)")
            self.assertTrue(response.diagnostics["table_answer"])
            self.assertEqual(response.local_sources[0].document, "apartments.csv")


if __name__ == "__main__":
    unittest.main()
