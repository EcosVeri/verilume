# Contributing

Thanks for helping improve Verilume. The project is local-first and desktop-oriented, so changes should keep the app easy for non-programmers to launch, understand, and trust.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Installation

```bash
git clone https://github.com/DamingoNdiwa/verilume.git
cd verilume
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
verilume run
```

## Coding Style

- Prefer small, focused changes that preserve the existing package structure.
- Keep local-first behavior intact: documents, Chroma data, manifests, cache, and settings should stay under user-controlled local paths.
- Preserve citation behavior: local citations use `[S1]`, `[S2]`; web citations use `[W1]`, `[W2]`.
- Keep UI copy clear and useful for non-programmers.

## Formatting

Run Ruff before submitting:

```bash
python -m ruff check src tests launcher.py scripts
```

Optionally, install the pre-commit hooks so this runs automatically:

```bash
pre-commit install
```

## Tests

Run the local validation suite:

```bash
python -m compileall -q src tests launcher.py scripts/build_macos_app.py
python -m unittest discover -s tests -v
verilume doctor
```

Add focused tests when changing routing, ingestion, settings, exports, citation formatting, retrieval, or UI helper behavior.

## Submitting PRs

- Fork the repository, create a feature branch, and open a pull request against `main`.
- Describe what changed and why.
- Include screenshots or GIFs for visible UI changes.
- Link related issues when possible.
- Do not commit `.env`, tokens, local Chroma databases, uploaded documents, build outputs, logs, or private user files.
- Keep PRs scoped enough to review without losing the thread.

## License

By contributing to Verilume, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), the same license that covers the project (Apache-2.0, Section 5 — inbound = outbound).
