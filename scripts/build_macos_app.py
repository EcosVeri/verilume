"""Build a macOS .app bundle with PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from shutil import which
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    uv = which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build the macOS app")

    subprocess.run([uv, "pip", "install", "-e", ".[mac]"], cwd=ROOT, check=True)
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
            "--copy-metadata",
            "streamlit",
            "--collect-data",
            "streamlit",
            "--add-data",
            f"{ROOT / 'src' / 'verilume' / 'app.py'}:verilume",
            str(ROOT / "launcher.py"),
        ],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
