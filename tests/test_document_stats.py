from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from verilume.settings import AppSettings
from verilume.utils.document_stats import _persisted_collection_count


class FakeCollection:
    def count(self) -> int:
        return 7


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, path: str) -> None:
        self.path = path
        self.closed = False
        self.cache_cleared = False
        FakeClient.instances.append(self)

    def get_or_create_collection(self, name: str, metadata: dict[str, str]) -> FakeCollection:
        return FakeCollection()

    def close(self) -> None:
        self.closed = True

    def clear_system_cache(self) -> None:
        self.cache_cleared = True


class DocumentStatsTests(unittest.TestCase):
    def test_persisted_collection_count_closes_client(self) -> None:
        FakeClient.instances.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = AppSettings(
                docs_dir=Path(tmp_dir) / "docs",
                chroma_dir=Path(tmp_dir) / "chroma",
                manifest_path=Path(tmp_dir) / "manifest.json",
            )

            with patch("verilume.utils.document_stats.chromadb.PersistentClient", FakeClient):
                count = _persisted_collection_count(settings)

        self.assertEqual(count, 7)
        self.assertEqual(len(FakeClient.instances), 1)
        self.assertTrue(FakeClient.instances[0].closed)
        self.assertTrue(FakeClient.instances[0].cache_cleared)


if __name__ == "__main__":
    unittest.main()
