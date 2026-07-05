# Verilume

**Privacy-first AI research assistant for your local documents.**

Search PDFs, Office documents, scanned files, and images with hybrid retrieval, evidence verification, and optional web search — as a desktop app or a Python package. Your documents and search index never leave your machine; answers are generated through Hugging Face or fully locally with Ollama.

> **Evidence First. Answers Second.** Verilume doesn't just answer questions — it evaluates evidence, verifies claims, calibrates confidence, and shows you exactly why each source won.

[![CI](https://github.com/EcosVeri/verilume/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/EcosVeri/verilume/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)
![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-green)
[![macOS](https://img.shields.io/badge/macOS-download-black)](https://github.com/EcosVeri/verilume/releases/latest)
![Windows Soon](https://img.shields.io/badge/Windows-planned-lightgrey)
![PyPI Soon](https://img.shields.io/badge/PyPI-coming%20soon-blue)
[![Ruff](https://img.shields.io/badge/linting-ruff-261230)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)
![Active Development](https://img.shields.io/badge/status-active%20development-f59e0b)

<p align="center">
  <a href="#downloads"><strong>Download macOS</strong></a>
  · <a href="#quick-start">Quick Start</a>
  · <a href="#documentation">Documentation</a>
  · <a href="#roadmap">Roadmap</a>
  · <a href="#about-ecosveri">EcosVeri</a>
</p>

## 🚀 Version 1.0 Launch

Verilume 1.0 is out: the macOS desktop app is on [GitHub Releases](https://github.com/EcosVeri/verilume/releases/latest), and the Python package installs from GitHub today. The PyPI release is next.

Verilume is developed under the **EcosVeri** organization, with future releases, documentation, and related apps hosted under the EcosVeri ecosystem.

## Demo

![Verilume demo showing the first-run setup checklist, source controls, upload action, export controls, example prompts, and chat input.](docs/assets/verilume-demo-v2.gif)

- 🔒 **Private by design** — documents, embeddings, and the vector database stay on your machine.
- 🧾 **Evidence-verified answers** — claim-level support checks and calibrated confidence on every answer.
- 🔍 **Hybrid retrieval** — BM25 + embeddings + reranking, with optional authority-grouped web search.
- 📊 **Benchmark mode** — compare Local vs AI vs Web vs Full retrieval on any question.

## Quick Start

**Python package** (not yet on PyPI — install from GitHub):

```bash
python -m pip install "verilume @ git+https://github.com/EcosVeri/verilume.git@main"
verilume run
```

**Desktop app**: download `Verilume-macOS-arm64.zip` from the [v1.0.0 release](https://github.com/EcosVeri/verilume/releases/latest), unzip, and open `Verilume.app`.

Then upload documents, build the knowledge base, and ask questions — setup takes about a minute.

### What stays local, what's optional

- **Always local**: your documents, the search index (Chroma), embeddings, caches, chat history, and exports — everything is stored under `~/.verilume` and never uploaded.
- **Answer generation**: defaults to the Hugging Face Inference API (needs a token). Switch the backend to **Ollama** to generate answers fully on your machine.
- **Web search**: only runs against a provider you configure (e.g. Tavily) and can be turned off entirely for local-only research.

## Downloads

| Platform | Status |
| --- | --- |
| macOS (Apple Silicon) | [v1.0.0 on GitHub Releases](https://github.com/EcosVeri/verilume/releases/latest) |
| Windows | Planned |
| Linux | Planned |
| Python package | Not yet on PyPI — [install from GitHub](#install-from-github) |

## Why Verilume?

Modern AI assistants ask you to trust them. Verilume shows its work:

- **Transparent evidence verification** — every answer carries an Evidence Summary: confidence, winning source, agreement, claim support, and *why* the winner won.
- **Calibrated confidence** — the badge is evidence-capped: a freshness conflict caps it to Medium, zero supported claims force it to Low. Fluent prose can't fake certainty.
- **Local-first retrieval with page-level citations** — hybrid BM25 + embeddings + reranking over your own files, cited as `[S1]` with document and page. Nothing is uploaded.
- **Source-quality grouping** — web evidence is grouped and weighted by authority: government, research, university, news, then the open web.
- **Built-in benchmark mode** — compare Local vs AI vs Web vs Full retrieval on any question, with per-mode answers, latency, and faithfulness.
- **A desktop workspace** — manage, index, and query a personal knowledge base with document-aware suggested prompts.

Everything is designed around privacy, transparency and reproducible answers.

## Features

<table>
  <tr>
    <td>🔒 <strong>Privacy First</strong><br>Runs locally without uploading files.</td>
    <td>🧾 <strong>Evidence Verification</strong><br>Claim-level support checks, agreement, conflicts, and calibrated confidence on every answer.</td>
    <td>🔍 <strong>Hybrid Search</strong><br>BM25 + embeddings + reranking over local documents, plus optional authority-grouped web search.</td>
  </tr>
  <tr>
    <td>📊 <strong>Benchmark Mode</strong><br>Compare Local vs AI vs Web vs Full retrieval on any question.</td>
    <td>📚 <strong>Transparent Citations</strong><br>Page-level local citations and clickable web sources, exportable to Markdown or PDF.</td>
    <td>💻 <strong>Desktop Ready</strong><br>Streamlit app with a macOS launcher; Hugging Face or Ollama generation backends. Apache-2.0, built in public.</td>
  </tr>
  <tr>
    <td>📄 <strong>PDF &amp; Office</strong><br>PDF, Word, PowerPoint, and table-aware Excel ingestion.</td>
    <td>🖼 <strong>OCR</strong><br>Scanned PDFs and image uploads become searchable, citable evidence.</td>
    <td>💬 <strong>Chat Memory</strong><br>Conversation-aware follow-ups with exportable chat history.</td>
  </tr>
</table>

## Who Is Verilume For?

- **Researchers and students** who need cited, verifiable answers from papers, theses, and reports.
- **Engineers and analysts** searching manuals, specs, and internal documentation.
- **Legal, finance, healthcare, government, and energy teams** who cannot upload sensitive documents to cloud AI services.
- **Anyone building a private knowledge base** who wants transparent evidence instead of confident-sounding guesses.

## Screenshots

### Dark Mode

![Verilume dark launch screen showing the sidebar, status pills, research actions, and chat input.](docs/assets/verilume-launch-dark.png)

### Light Mode

![Verilume light launch screen after using the appearance toggle.](docs/assets/verilume-launch-light.png)

## Architecture

```mermaid
flowchart TD
    Q[Question] --> U[Query Understanding]
    U --> H[Hybrid Retrieval]
    H --> L[Local Search]
    H --> M[AI Knowledge]
    H --> W[Optional Web Search]
    L --> R[Hybrid Ranking + Rerank]
    M --> R
    W --> R
    R --> E[Evidence Verification]
    E --> C[Conflict Resolution]
    C --> A[Answer + Citations]
```

## Why Verilume Is Different

| Feature | Verilume | ChatGPT | NotebookLM | AnythingLLM |
| --- | --- | --- | --- | --- |
| Local documents | Yes | Partial | Yes | Yes |
| Local execution | Yes | No | No | Yes |
| Optional web search | Yes | Yes | No | Partial |
| Evidence verification | Yes | Partial | Partial | No |
| Evidence-capped confidence scoring | Yes | No | No | No |
| Benchmark across retrieval modes (Local / AI / Web / Full) | Yes | No | No | No |
| Source-authority grouping | Yes | No | No | No |
| Page-level citations | Yes | Partial | Yes | Partial |
| Offline mode | Soon | No | No | Yes |

## Citations

Local document citations use `[S1]`, `[S2]`, `[S3]` and show document names plus page metadata when available.

Web citations use `[W1]`, `[W2]`, `[W3]` and are shown separately as clickable sources.

## Roadmap

### Version 1.0 (shipped)

- PDF support
- Word support
- Excel and table-aware retrieval foundations
- OCR
- Hybrid search
- Citations
- Desktop app

### Version 1.1

- Windows builds
- PyPI release
- Better ranking
- Conversation memory polish

### Version 2.0

- Ollama-first local setup
- Vision models
- Local embeddings controls
- Knowledge graphs
- Multi-agent pipeline

### Future

- EcosVeri ecosystem
- ecosveri.dev
- Plugin system
- Cloud sync
- API
- Enterprise edition

See [ROADMAP.md](ROADMAP.md) for the full plan.

## Install

### Desktop

Download the latest `Verilume-macOS-*.zip` from [GitHub Releases](https://github.com/EcosVeri/verilume/releases/latest), unzip it, and open `Verilume.app`.

macOS may show a first-run security warning because builds are not notarized yet. If that happens, Control-click `Verilume.app`, choose Open, then confirm.

On macOS, you can also double-click the source launcher:

```text
Verilume.command
```

### Install from GitHub

Install the v1.0.0 release:

```bash
python -m pip install "verilume @ git+https://github.com/EcosVeri/verilume.git@v1.0.0"
verilume run
```

Or install the latest committed version:

```bash
python -m pip install "verilume @ git+https://github.com/EcosVeri/verilume.git@main"
verilume run
```

Verilume is not yet on PyPI. Until the PyPI release is published, install from GitHub or run from source.

### Developers

```bash
git clone git@github.com:EcosVeri/verilume.git
cd verilume
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
verilume run
```

Run directly with Streamlit:

```bash
python -m streamlit run src/verilume/app.py
```

## Basic Use

1. Launch the app.
2. Enter a Hugging Face token.
3. Enter a Tavily API key when web search is needed.
4. Select a model.
5. Upload documents.
6. Build the knowledge base.
7. Ask questions.
8. Review local and web citations separately.
9. Export the chat to Markdown or PDF.

## CLI

```bash
verilume run
verilume ingest
verilume stats
verilume config
verilume doctor
```

## Benchmarks

Planned benchmark coverage will compare retrieval and answer quality across:

```text
Question -> Needle -> Needlite -> Verilume
```

Future comparisons will include LangChain, LlamaIndex, Haystack, and NotebookLM-style workflows.

## Documentation

Documentation covering installation, architecture, retrieval, evidence, OCR, and FAQ material is being organized and will be published under the EcosVeri ecosystem. In the meantime:

- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Security policy](SECURITY.md)

## Cite Verilume

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21210006.svg)](https://doi.org/10.5281/zenodo.21210006)

If you use Verilume in your research or software, please cite:

> Mingo Ndiwago, D. (2026). *Verilume: Privacy-first AI research assistant for your local documents*. Zenodo. https://doi.org/10.5281/zenodo.21210006

This DOI represents all versions of Verilume and always resolves to the latest release.

## About EcosVeri

Verilume is the first project of EcosVeri, an open-source ecosystem focused on trustworthy AI, evidence verification, semantic search, and research tools.

Future EcosVeri projects include:

- Needlite
- VeriSearch
- VeriAgents
- Additional AI developer tools

Website: ecosveri.dev (Coming Soon).

## License

Apache-2.0. See [LICENSE](LICENSE).

