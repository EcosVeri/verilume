from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_release_artifacts import find_forbidden_artifacts  # noqa: E402


class ReleaseArtifactScanTests(unittest.TestCase):
    def test_clean_tree_has_no_offenders(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app").mkdir()
            (root / "app" / "main.py").write_text("print('hi')", encoding="utf-8")
            self.assertEqual(find_forbidden_artifacts(root), [])

    def test_flags_private_data(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".verilume").mkdir()
            (root / "chroma_db").mkdir()
            (root / "config.env").write_text("HF_TOKEN=secret", encoding="utf-8")
            (root / "store.sqlite").write_text("x", encoding="utf-8")
            offenders = find_forbidden_artifacts(root)
            names = {Path(p).name for p in offenders}
            self.assertIn(".verilume", names)
            self.assertIn("chroma_db", names)
            self.assertIn("config.env", names)
            self.assertIn("store.sqlite", names)

    def test_missing_root_returns_empty(self) -> None:
        self.assertEqual(find_forbidden_artifacts("/no/such/path/xyz"), [])


if __name__ == "__main__":
    unittest.main()
