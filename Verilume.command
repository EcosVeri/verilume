#!/bin/zsh
set -e
cd "$(dirname "$0")"
python3 -m pip install -e .
python3 launcher.py
