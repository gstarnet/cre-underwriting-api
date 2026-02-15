#!/usr/bin/env zsh
set -euo pipefail

# scripts/test_explainability.sh
#
# Purpose:
# - Verifies explainability artifacts are generated and valid
# - Optionally validates API exposure if /explainability exists
#
# Usage:
#   chmod +x scripts/test_explainability.sh
#   ./scripts/test_explainability.sh
#   BASE=http://127.0.0.1:8000 ./scripts/test_explainability.sh
#
# Requires:
# - jq (brew install jq)
#
# Notes:
# - Assumes you have already trained a model at models/model.joblib (python -m src.train)
# - API check is optional and will be skipped if endpoint is missing.

BASE="${BASE:-http://127.0.0.1:8000}"

command -v jq >/dev/null || { echo "jq is required (brew install jq)"; exit 1; }

echo "== precheck: model exists =="
[[ -f models/model.joblib ]] || { echo "FAIL: models/model.joblib not found. Run: python -m src.train"; exit 2; }
echo "OK: model exists"

echo "== run: src.explain =="
python -m src.explain >/dev/null
echo "OK: src.explain ran"

echo "== check: output files exist =="
[[ -f reports/feature_importance.csv ]] || { echo "FAIL: reports/feature_importance.csv missing"; exit 3; }
[[ -f reports/feature_importance.json ]] || { echo "FAIL: reports/feature_importance.json missing"; exit 3; }
echo "OK: output files exist"

echo "== validate: CSV schema =="
# Ensure header contains expected columns
head -n 1 reports/feature_importance.csv | grep -q '^feature,importance_mean,importance_std' \
  || { echo "FAIL: CSV header mismatch"; exit 4; }
# Ensure at least 2 lines (header + 1 row)
[[ "$(wc -l < reports/feature_importance.csv | tr -d ' ')" -ge 2 ]] \
  || { echo "FAIL: CSV has no rows"; exit 4; }
echo "OK: CSV schema"

echo "== validate: JSON schema =="

# Must parse
jq -e '.' reports/feature_importance.json >/dev/null

# Must have expected top-level keys and rows array
jq -e '
  type=="object" and
  has("method") and
  has("scoring") and
  has("n_repeats") and
  has("rows") and
  (.rows | type=="array") and
  (.rows | length >= 1) and
  (.rows[0] | has("feature")) and
  (.rows[0] | has("importance_mean")) and
  (.rows[0] | has("importance_std"))
' reports/feature_importance.json >/dev/null \
  || { echo "FAIL: JSON schema invalid"; exit 5; }

echo "OK: JSON schema"

echo "== show: top 5 features (from JSON) =="
jq -r '.feature_importance[:5] | to_entries[] | "\(.key+1)) \(.value.feature) mean=\(.value.importance_mean) std=\(.value.importance_std)"' \
  reports/feature_importance.json

echo "== optional: API /explainability (skip if missing) =="
# Health check first (skip API checks if server not running)
if curl -sS --connect-timeout 2 --max-time 5 "$BASE/health" >/dev/null 2>&1; then
  code="$(curl -sS --connect-timeout 2 --max-time 10 -o /tmp/explain_api.json -w "%{http_code}" "$BASE/explainability" || true)"
  if [[ "$code" == "200" ]]; then
    jq -e 'has("feature_importance") and (.feature_importance | length) >= 1' /tmp/explain_api.json >/dev/null \
      || { echo "FAIL: /explainability returned 200 but schema invalid"; rm -f /tmp/explain_api.json; exit 6; }
    echo "OK: API /explainability"
  else
    # 404 is acceptable if you haven't wired the endpoint yet
    if [[ "$code" == "404" ]]; then
      echo "SKIP: API /explainability not implemented (404)"
    else
      echo "SKIP: API /explainability returned HTTP $code"
      head -c 300 /tmp/explain_api.json || true
      echo
    fi
  fi
  rm -f /tmp/explain_api.json
else
  echo "SKIP: API not running at $BASE"
fi

echo "OK: explainability test passed"
