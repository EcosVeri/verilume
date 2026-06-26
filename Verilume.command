#!/bin/zsh
set -euo pipefail

function keep_open_on_error() {
  local exit_status=$?
  if [[ $exit_status -ne 0 && $exit_status -ne 130 ]]; then
    echo
    echo "Verilume failed to launch. Review the message above."
    read -k 1 "?Press any key to close this window."
  fi
}

trap keep_open_on_error EXIT

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Please install Python 3 and try again."
  exit 1
fi

export VERILUME_PORT="${VERILUME_PORT:-8501}"
APP_URL="http://localhost:${VERILUME_PORT}"
echo "Starting Verilume from $APP_DIR"
echo "Target URL: $APP_URL"
echo "Use this launcher for local development; README.md emphasizes local data and architecture."

function verilume_is_running() {
  if command -v lsof >/dev/null 2>&1; then
    if lsof -ti "TCP:${VERILUME_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
  fi

  local attempt
  for attempt in {1..8}; do
    if command -v curl >/dev/null 2>&1; then
      if curl --fail --silent --show-error --max-time 3 "$APP_URL" >/dev/null 2>&1; then
        return 0
      fi
    else

      if python3 - "$APP_URL" <<'PY'
import sys
from urllib.request import urlopen

url = sys.argv[1]
try:
    with urlopen(url, timeout=1.5) as response:
        raise SystemExit(0 if 200 <= response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
      then
        return 0
      fi
    fi
    sleep .5
  done
  return 1
}

if verilume_is_running; then
  echo "Verilume is already running at $APP_URL"
  if command -v open >/dev/null 2>&1; then
    open "$APP_URL"
  fi
  exit 0
fi

python3 -m pip install --disable-pip-version-check -e .
python3 launcher.py
