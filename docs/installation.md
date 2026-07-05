# Installation

Verilume can run from source today. A PyPI package and desktop installers are planned release assets.

## Install From GitHub

```bash
git clone git@github.com:DamingoNdiwa/verilume.git
cd verilume
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
verilume run
```

## Run With Streamlit

```bash
python -m streamlit run src/verilume/app.py
```

## macOS Launcher

On macOS, double-click:

```text
Verilume.command
```

The first launch may download local embedding models. Uploaded documents, Chroma data, and local settings are stored under `~/.verilume`.

## PyPI

After publication:

```bash
python -m pip install verilume
verilume run
```

## CLI

```bash
verilume run
verilume ingest
verilume ingest --reset
verilume stats
verilume config
verilume doctor
```
