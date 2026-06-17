# Contributing

Thanks for helping improve Verilume. The project is intentionally small and local-first, so changes should keep the desktop app easy for non-programmers to launch and understand.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Local Checks

Run these before opening a pull request:

```bash
python -m ruff check src tests launcher.py scripts
python -m compileall -q src tests launcher.py scripts/build_macos_app.py
python -m unittest discover -s tests -v
verilume doctor
```

## Pull Requests

- Keep user-facing Streamlit behavior clear and documented.
- Do not commit `.env`, tokens, local Chroma databases, uploaded documents, build outputs, or logs.
- Add focused tests for routing, ingestion, settings, exports, or UI helper behavior when changing those areas.
- Preserve citation behavior: local citations use `[S1]`, `[S2]`; web citations use `[W1]`, `[W2]`.
