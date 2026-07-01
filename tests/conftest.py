"""Session-wide test isolation.

``verilume.settings`` resolves ``DATA_HOME`` from the real home directory at
import time, and ``AppSettings()`` bakes that into its path fields. If a test
then builds a ``VerilumeRAG`` it would open the developer's real ``~/.verilume``
Chroma/SQLite stores — the non-hermetic risk called out in the project review.

Redirecting ``HOME`` here, at conftest import, happens before pytest imports any
test module (and therefore before the first ``import verilume``), so every path
default lands inside a throwaway temp directory instead.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Executed at conftest import — earlier than any test module's `import verilume`,
# and therefore before settings resolves DATA_HOME from the home directory.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="verilume-tests-"))
os.environ["HOME"] = str(_TEST_HOME)
os.environ["USERPROFILE"] = str(_TEST_HOME)  # Windows equivalent of HOME

_DATA_HOME = _TEST_HOME / ".verilume"
# Pin the documented path env vars too, for tests that go through from_env().
os.environ.setdefault("DOCS_DIR", str(_DATA_HOME / "documents"))
os.environ.setdefault("CHROMA_DIR", str(_DATA_HOME / "chroma_db"))
os.environ.setdefault("TABLE_STORE_DIR", str(_DATA_HOME / "tables"))
