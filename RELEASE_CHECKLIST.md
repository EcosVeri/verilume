# Release Checklist

Run through this before publishing a macOS build or a PyPI release.

## Build integrity

- [ ] `python -m ruff check src tests launcher.py scripts` passes.
- [ ] `python -m pytest tests -q` passes (hermetic; touches no real `~/.verilume`).
- [ ] `pip-audit --strict` reviewed; no unaddressed high-severity advisories.
- [ ] `uv.lock` is up to date and used for the build.

## No private data in artifacts

- [ ] `python scripts/check_release_artifacts.py dist` passes.
- [ ] For a macOS `.app`, mount/inspect the bundle and re-run the scanner against it.
- [ ] Confirm no `.verilume/`, `chroma_db/`, `documents/`, `config.env`, `.env`,
      `*.sqlite`, `ingestion_manifest.json`, or `semantic_cache.json` are bundled.
- [ ] Confirm no API keys are embedded in code, config, or logs.

## macOS distribution

- [ ] App built in an isolated environment (clean checkout, fresh venv).
- [ ] Binary is code-signed.
- [ ] Binary is notarized and stapled before distribution to non-technical users.

## Metadata

- [ ] `CHANGELOG.md` updated.
- [ ] Version bumped in `pyproject.toml` / `CITATION.cff`.
- [ ] README install instructions verified against the new artifact.
