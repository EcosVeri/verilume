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
echo "Use this launcher for local development; README.md is direct and concise."

function verilume_port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "TCP:${VERILUME_PORT}" -sTCP:LISTEN 2>/dev/null || true
  fi
}

function wait_for_port_to_close() {
  local attempt
  for attempt in {1..20}; do
    if [[ -z "$(verilume_port_pids)" ]]; then
      return 0
    fi
    sleep .25
  done
  return 1
}

function stop_existing_verilume() {
  local pids
  pids="$(verilume_port_pids)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "Stopping existing process on $APP_URL so this checkout is loaded..."
  echo "$pids" | while read -r pid; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  if ! wait_for_port_to_close; then
    echo "Existing process did not stop cleanly; forcing shutdown..."
    pids="$(verilume_port_pids)"
    echo "$pids" | while read -r pid; do
      if [[ -n "$pid" ]]; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
    wait_for_port_to_close || true
  fi
}

stop_existing_verilume

python3 -m pip install --disable-pip-version-check --quiet -e .
export PYTHONPATH="$APP_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if command -v open >/dev/null 2>&1; then
  (sleep 2 && open "$APP_URL") >/dev/null 2>&1 &
fi

python3 launcher.py
