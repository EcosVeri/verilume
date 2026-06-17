"""Build a macOS .app bundle with PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[mac]"], cwd=ROOT, check=True)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--name",
            "Verilume",
            "--windowed",
            "--noconfirm",
            "--clean",
            str(ROOT / "launcher.py"),
        ],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
