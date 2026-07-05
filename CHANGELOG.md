# Changelog

## Unreleased

- Repositioned Verilume as a privacy-first desktop AI assistant for documents, research, evidence verification, and hybrid local plus web search.
- Restructured the README for launch: value proposition, badges, launch banner, demo, quick start, feature grid, "Who Is Verilume For", documentation links, and an expanded comparison table.
- Refreshed the README demo recording and launch screenshots.
- Added GitHub issue templates and a pull request template for community contributions.
- Expanded contributor, security, and roadmap documentation.
- Added a NOTICE file and explicit Apache-2.0 contribution terms in the contributing guide.
- Trimmed the roadmap and TODO to remaining work only.
- Bumped the package version to 1.0.0 with Beta status and a Python 3.13 classifier.
- Added pre-commit hooks and Dependabot updates for pip and GitHub Actions.

## 0.1.0 - 2026-06-23

- Initial local-first Streamlit desktop app.
- Local document ingestion for PDF, DOCX, TXT, Markdown, and CSV.
- Chroma vector storage with incremental ingestion.
- Hugging Face generation with local, model-knowledge, and web-assisted answer stages.
- Optional provider-based web search with separate clickable web citations.
- Markdown and PDF chat exports.
- CLI commands for launch, ingestion, stats, config, and health checks.
- macOS launcher script and PyInstaller build helper.
