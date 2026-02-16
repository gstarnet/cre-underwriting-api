#!/usr/bin/env zsh
set -euo pipefail

# scripts/rebuild_all.sh
#
# Rebuild all generated artifacts from scratch (safe to run repeatedly).
# - Removes generated files if present (no-op if already removed)
# - Regenerates synthetic data, trains model, runs TS validation, explainability
#
# Usage:
#   chmod +x scripts/rebuild_all.sh
#   ./scripts/rebuild_all.sh
#   BASE=http://127.0.0.1:8000 ./scripts/rebuild_all.sh   # if you later add API checks
#
# Assumes:
#   - venv already created + activated
#   - dependencies installed: pip install -r requirements.txt

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "== clean: generated artifacts =="

# Safe deletes (no failure if missing)
rm -f data/raw/cre_deals.csv || true
rm -f models/model.joblib || true
rm -f reports/metrics.csv || true
rm -f reports/ts_cv_metrics.csv || true
rm -f reports/ts_cv_summary.csv || true
rm -f reports/feature_importance.csv || true
rm -f reports/feature_importance.json || true

# Ensure directories exist (in case they were removed)
mkdir -p data/raw models reports

echo "== build: synthetic dataset =="
"$PYTHON" -m src.synth_data

echo "== build: train model =="
"$PYTHON" -m src.train

echo "== build: time-series validation =="
"$PYTHON" -m src.validate_ts

echo "== build: explainability artifacts =="
"$PYTHON" -m src.explain

echo "OK: rebuild completed"
