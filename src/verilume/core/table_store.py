"""SQLite-backed table metadata and CSV storage for local calculations."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass(frozen=True, slots=True)
class TableMetadata:
    table_id: str
    document: str
    page: int | None
    columns: list[str]
    row_count: int
    column_types: dict[str, str]
    summary: str
    dataframe_path: Path
    created_at: str
    source_path: str = ""
    file_signature: str = ""


class TableStore:
    def __init__(self, table_dir: Path | str) -> None:
        self.table_dir = Path(table_dir).expanduser()
        self.frames_dir = self.table_dir / "frames"
        self.db_path = self.table_dir / "tables.sqlite3"
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_table(
        self,
        df: pd.DataFrame,
        *,
        document: str,
        page: int | None = None,
        source_path: Path | str | None = None,
        table_id: str | None = None,
    ) -> TableMetadata:
        clean_df = _clean_dataframe(df)
        source = Path(source_path).expanduser() if source_path else None
        signature = _file_signature(source) if source else ""
        table_id = table_id or _table_id(document, signature, clean_df)
        dataframe_path = self.frames_dir / f"{table_id}.csv"
        clean_df.to_csv(dataframe_path, index=False)
        metadata = TableMetadata(
            table_id=table_id,
            document=document,
            page=page,
            columns=[str(column) for column in clean_df.columns],
            row_count=int(len(clean_df)),
            column_types={str(column): str(dtype) for column, dtype in clean_df.dtypes.items()},
            summary=_table_summary(document, clean_df),
            dataframe_path=dataframe_path,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_path=str(source or ""),
            file_signature=signature,
        )
        self._upsert_metadata(metadata)
        return metadata

    def index_local_tables(self, docs_dir: Path | str) -> list[TableMetadata]:
        docs_path = Path(docs_dir).expanduser()
        if not docs_path.exists():
            return []
        indexed: list[TableMetadata] = []
        for path in sorted(docs_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
                continue
            signature = _file_signature(path)
            existing = self._metadata_for_source(path, signature)
            if existing is not None:
                indexed.append(existing)
                continue
            try:
                separator = "\t" if path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(path, sep=separator)
            except Exception:
                continue
            indexed.append(self.add_table(df, document=path.name, source_path=path))
        return indexed

    def list_tables(self) -> list[TableMetadata]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tables ORDER BY created_at DESC").fetchall()
        return [_metadata_from_row(row) for row in rows]

    def load_table(self, table_id: str) -> pd.DataFrame:
        metadata = self.get_table(table_id)
        if metadata is None:
            raise KeyError(f"Unknown table: {table_id}")
        return pd.read_csv(metadata.dataframe_path)

    def get_table(self, table_id: str) -> TableMetadata | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tables WHERE table_id = ?", (table_id,)).fetchone()
        return _metadata_from_row(row) if row is not None else None

    def search_tables(self, question: str) -> list[TableMetadata]:
        question_terms = _terms(question)
        scored: list[tuple[float, TableMetadata]] = []
        for table in self.list_tables():
            haystack = " ".join([table.document, table.summary, *table.columns])
            table_terms = _terms(haystack)
            if not table_terms:
                continue
            score = len(question_terms & table_terms) / max(1, len(question_terms))
            if score > 0:
                scored.append((score, table))
        return [table for _, table in sorted(scored, key=lambda item: item[0], reverse=True)]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tables(
                    table_id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    page INTEGER,
                    columns_json TEXT NOT NULL,
                    column_types_json TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    dataframe_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source_path TEXT,
                    file_signature TEXT
                )
                """
            )

    def _upsert_metadata(self, metadata: TableMetadata) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tables(
                    table_id, document, page, columns_json, column_types_json,
                    row_count, summary, dataframe_path, created_at, source_path, file_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.table_id,
                    metadata.document,
                    metadata.page,
                    json.dumps(metadata.columns),
                    json.dumps(metadata.column_types),
                    metadata.row_count,
                    metadata.summary,
                    str(metadata.dataframe_path),
                    metadata.created_at,
                    metadata.source_path,
                    metadata.file_signature,
                ),
            )

    def _metadata_for_source(self, path: Path, signature: str) -> TableMetadata | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tables WHERE source_path = ? AND file_signature = ?",
                (str(path), signature),
            ).fetchone()
        return _metadata_from_row(row) if row is not None else None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()


def _metadata_from_row(row: sqlite3.Row) -> TableMetadata:
    return TableMetadata(
        table_id=str(row["table_id"]),
        document=str(row["document"]),
        page=row["page"],
        columns=list(json.loads(row["columns_json"])),
        row_count=int(row["row_count"]),
        column_types=dict(json.loads(row["column_types_json"])),
        summary=str(row["summary"]),
        dataframe_path=Path(row["dataframe_path"]),
        created_at=str(row["created_at"]),
        source_path=str(row["source_path"] or ""),
        file_signature=str(row["file_signature"] or ""),
    )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(column).strip() or f"column_{index + 1}" for index, column in enumerate(clean.columns)]
    return clean


def _table_summary(document: str, df: pd.DataFrame) -> str:
    columns = ", ".join(str(column) for column in df.columns)
    numeric_columns = ", ".join(str(column) for column in df.select_dtypes(include="number").columns)
    numeric_text = f" Numeric columns: {numeric_columns}." if numeric_columns else ""
    return f"{document}: {len(df)} rows. Columns: {columns}.{numeric_text}"


def _table_id(document: str, signature: str, df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(document.encode("utf-8"))
    digest.update(signature.encode("utf-8"))
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _file_signature(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", (text or "").lower())
        if len(token) > 1
    }
