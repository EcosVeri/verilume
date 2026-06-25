#!/bin/zsh
set -euo pipefail

function keep_open_on_error() {
  local status=$?
  if [[ $status -ne 0 && $status -ne 130 ]]; then
    echo
    echo "Verilume failed to launch. Review the message above."
    read -k 1 "?Press any key to close this window."
  fi
}

trap keep_open_on_error EXIT

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Please install Python 3 and try again."
  exit 1
fi

python3 -m pip install -e .
python3 launcher.py
