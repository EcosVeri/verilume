# Security

Verilume stores user data locally by default under `~/.verilume`.

## Secrets

Do not commit Hugging Face tokens, web search provider API keys, `.env`, `.streamlit/secrets.toml`, `~/.verilume/config.env`, uploaded documents, or Chroma database files. The app masks tokens in the UI and `verilume config` output.

## Reporting

For a public repository, open a private security advisory if GitHub security advisories are enabled. Otherwise, contact the maintainer listed on the repository before publishing exploit details.
