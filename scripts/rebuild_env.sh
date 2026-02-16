#!/usr/bin/env zsh
set -euo pipefail

# scripts/rebuild_env.sh
#
# Rebuild local Python environment deterministically.
#
# Usage:
#   chmod +x scripts/rebuild_env.sh
#   ./scripts/rebuild_env.sh
#
# Notes:
# - Uses python3 if available, otherwise python.
# - Deletes and recreates .venv each run.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-}"

if [[ -z "${PY_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
  else
    echo "FAIL: python3/python not found on PATH"
    exit 1
  fi
fi

echo "== env: remove existing .venv (if any) =="
rm -rf .venv || true

echo "== env: create venv =="
"$PY_BIN" -m venv .venv

echo "== env: activate =="
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== env: upgrade pip tooling =="
python -m pip install --upgrade pip setuptools wheel

echo "== env: install requirements =="
python -m pip install -r requirements.txt

echo "== env: sanity =="
python -c 'import sys; print("python:", sys.version)'
python -c 'import sklearn, pandas, fastapi, uvicorn; print("deps: ok")'

echo "OK: environment rebuilt"