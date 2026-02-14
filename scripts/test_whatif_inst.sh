#!/usr/bin/env zsh
set -euo pipefail

# scripts/test_whatif_inst.sh
#
# Purpose:
# - Smoke test for institutional what-if endpoint: POST /whatif_inst
# - Adds hardening checks:
#   1) Happy path returns scenarios with required fields
#   2) Scenarios are sorted (descending) by IRR when sort_by="irr"
#   3) Invalid sort_by is rejected with HTTP 422 and a helpful validation message
#   4) Oversized scenario grid triggers guardrail (expects 4xx)
#
# Usage:
#   chmod +x scripts/test_whatif_inst.sh
#   BASE=http://127.0.0.1:8000 ./scripts/test_whatif_inst.sh
#
# Requires:
# - jq (brew install jq)
#
# Notes:
# - Assumes API is already running: python -m src
# - Uses curl timeouts to avoid hanging test runs.

BASE="${BASE:-http://127.0.0.1:8000}"

command -v jq >/dev/null || { echo "jq is required (brew install jq)"; exit 1; }

echo "== health =="
curl -sf --connect-timeout 3 --max-time 10 "$BASE/health" >/dev/null
echo "OK: health"

###############################################################################
# Shared base payload (good defaults)
###############################################################################
payload_base='{
  "deal_id":"DTEST",
  "asof_date":"2024-06-01",
  "property_type":"Industrial",
  "city":"Tampa",
  "state":"FL",
  "zip":"33602",
  "year_built":2008,
  "gross_leasable_sqft":125000,
  "units":null,
  "purchase_price":25000000,
  "noi_t12":1500000,
  "occupancy_t12":0.95,
  "opex_t12":450000,
  "gross_rent_t12":2200000,
  "ltv":0.65,
  "interest_rate":0.062,
  "amort_years":25,
  "exit_cap_rate":0.065,
  "selling_cost_pct":0.02,

  "hold_years":5,
  "interest_only_years":1,
  "rent_growth":0.03,
  "opex_inflation":0.03,
  "occupancy_target":0.97,
  "occupancy_reversion_years":2,
  "taxes_year1":350000,
  "insurance_year1":60000,
  "reassess_taxes":true,
  "reassessed_tax_rate":0.02,
  "reassess_year":1,
  "capex_reserve_per_sqft":0.25,
  "replacement_capex_per_sqft":0.15,
  "value_add_capex":{"2":250000},
  "occupancy_shock_year":2,
  "occupancy_shock_drop":0.10,
  "occupancy_recovery_years":2,
  "rate_shock_year":3,
  "rate_shock_bps":150,
  "refi_year":3,
  "refi_ltv":0.65,
  "refi_rate":0.06,
  "refi_amort_years":25,
  "refi_cost_pct":0.01
}'

###############################################################################
# Helper: POST JSON, fail on non-2xx, return response
###############################################################################
_post_json() {
  local url="$1"
  local json="$2"
  curl -sf --connect-timeout 3 --max-time 60 \
    -X POST "$url" -H "Content-Type: application/json" -d "$json"
}

###############################################################################
# 1) Happy path + schema checks + sorted-by-IRR check
###############################################################################
echo "== whatif_inst (happy path) =="

payload_happy="$(jq -c '. + {
  "purchase_prices":[24000000,25000000],
  "exit_cap_rates":[0.0625,0.0675],
  "rent_growths":[0.02,0.03],
  "occupancy_shock_drops":[0.0,0.10],
  "rate_shock_bps_values":[0,150],
  "max_scenarios":40,
  "top_n":5,
  "sort_by":"irr"
}' <<<"$payload_base")"

resp="$(_post_json "$BASE/whatif_inst" "$payload_happy")"

# Basic schema checks
scenarios_len="$(jq '.scenarios | length' <<<"$resp")"
[[ "$scenarios_len" -ge 1 ]] || { echo "FAIL: scenarios array empty"; exit 2; }

jq -e '
  .scenarios[0] and
  (.scenarios[0] | has("inputs")) and
  (.scenarios[0] | has("predicted_noi_next12")) and
  (.scenarios[0] | has("summary")) and
  (.scenarios[0].summary | has("irr"))
' >/dev/null <<<"$resp" || { echo "FAIL: response schema missing keys"; exit 3; }

# Ensure scenarios are sorted desc by IRR (best first) when sort_by="irr"
jq -e '
  ( [ .scenarios[].summary.irr ] as $xs |
    ($xs | all(. != null)) and
    ($xs == ($xs | sort | reverse))
  )
' >/dev/null <<<"$resp" || { echo "FAIL: scenarios not sorted by irr or irr missing"; exit 4; }

best_irr="$(jq -r '.scenarios[0].summary.irr' <<<"$resp")"
best_price="$(jq -r '.scenarios[0].inputs.purchase_price' <<<"$resp")"
echo "OK: whatif_inst scenarios=$scenarios_len best_irr=$best_irr best_price=$best_price"

echo "Top IRRs:"
jq -r '.scenarios[:3] | to_entries[] |
  "\(.key+1)) irr=\(.value.summary.irr) price=\(.value.inputs.purchase_price) ltv=\(.value.inputs.ltv) rate=\(.value.inputs.interest_rate) exit_cap=\(.value.inputs.exit_cap_rate)"' \
  <<<"$resp"

###############################################################################
# 2) Invalid sort_by should be rejected with 422 (validation hardening)
###############################################################################
echo "== whatif_inst (invalid sort_by -> 422) =="

payload_bad_sort="$(jq -c '. + {
  "purchase_prices":[24000000,25000000],
  "exit_cap_rates":[0.0625,0.0675],
  "max_scenarios":10,
  "top_n":3,
  "sort_by":"this_is_not_valid"
}' <<<"$payload_base")"

tmp2="$(mktemp)"
code2="$(curl -sS --connect-timeout 3 --max-time 60 \
  -o "$tmp2" -w "%{http_code}" \
  -X POST "$BASE/whatif_inst" \
  -H "Content-Type: application/json" \
  -d "$payload_bad_sort" || true)"

if [[ "$code2" -ne 422 ]]; then
  echo "FAIL: expected HTTP 422 for invalid sort_by, got HTTP $code2"
  head -c 400 "$tmp2" || true
  echo
  rm -f "$tmp2"
  exit 5
fi

# Assert error message mentions sort_by and allowed values
jq -e '
  (.detail != null) and
  ((.detail | tostring | test("sort_by")) and (.detail | tostring | test("must be one of")))
' >/dev/null <"$tmp2" || {
  echo "FAIL: 422 body did not include expected sort_by validation detail"
  head -c 400 "$tmp2" || true
  echo
  rm -f "$tmp2"
  exit 5
}

rm -f "$tmp2"
echo "OK: invalid sort_by rejected with 422 (expected)"

###############################################################################
# 3) Oversized grid should trigger guardrail (expects 4xx)
###############################################################################
echo "== whatif_inst (guardrail oversized grid) =="

payload_big="$(jq -c '. + {
  "purchase_prices":[22000000,23000000,24000000,25000000,26000000,27000000,28000000,29000000,30000000],
  "exit_cap_rates":[0.055,0.0575,0.06,0.0625,0.065,0.0675,0.07,0.0725,0.075],
  "rent_growths":[0.01,0.02,0.03,0.04,0.05],
  "rate_shock_bps_values":[0,50,100,150,200],
  "max_scenarios":20,
  "top_n":5,
  "sort_by":"irr"
}' <<<"$payload_base")"

tmp3="$(mktemp)"
code3="$(curl -sS --connect-timeout 3 --max-time 60 \
  -o "$tmp3" -w "%{http_code}" \
  -X POST "$BASE/whatif_inst" \
  -H "Content-Type: application/json" \
  -d "$payload_big" || true)"

echo "HTTP $code3"
head -c 400 "$tmp3" || true
echo
rm -f "$tmp3"

if [[ "$code3" -ge 400 && "$code3" -lt 500 ]]; then
  echo "OK: guardrail triggered (expected 4xx)"
else
  echo "FAIL: expected guardrail 4xx, got HTTP $code3"
  exit 6
fi

echo "OK: all whatif_inst tests passed"
