"""Scan a build/release directory for private user data before shipping.

Release artifacts must never bundle a developer's local Verilume data — indexed
documents, the Chroma store, SQLite stores, saved config (which may hold API
keys), or the semantic cache. Run this against `dist/` (or a mounted .app) as a
release gate:

    python scripts/check_release_artifacts.py dist

Exits non-zero and lists offenders if any are found.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Directory names that indicate bundled user data.
FORBIDDEN_DIR_NAMES = {".verilume", "chroma_db", "documents"}
# Exact file names that must never ship.
FORBIDDEN_FILE_NAMES = {
    "config.env",
    ".env",
    "ingestion_manifest.json",
    "semantic_cache.json",
    "knowledge_graph.sqlite",
    "formulas.sqlite",
    "ocr_blocks.sqlite",
    "structured_documents.sqlite",
    "multimodal.sqlite",
}
# Suffixes that must never ship (local stores / raw uploads).
FORBIDDEN_SUFFIXES = {".sqlite", ".sqlite3"}


def find_forbidden_artifacts(root: str | Path) -> list[str]:
    """Return repo-relative paths of any private-data artifacts found under root."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    offenders: list[str] = []
    for path in root_path.rglob("*"):
        name = path.name
        if path.is_dir():
            if name in FORBIDDEN_DIR_NAMES:
                offenders.append(str(path))
            continue
        if name in FORBIDDEN_FILE_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(str(path))
    return sorted(offenders)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    target = args[0] if args else "dist"
    offenders = find_forbidden_artifacts(target)
    if offenders:
        print(f"Release check FAILED: private data found under {target}:")
        for offender in offenders:
            print(f"  - {offender}")
        return 1
    print(f"Release check passed: no private data found under {target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
