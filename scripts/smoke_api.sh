#!/usr/bin/env zsh
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"

# Require jq (used to merge JSON payloads)
command -v jq >/dev/null || { echo "jq is required (brew install jq)"; exit 1; }

echo "== health =="
curl -sf "$BASE/health" >/dev/null

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
  "selling_cost_pct":0.02
}'

echo "== predict =="
curl -sf -X POST "$BASE/predict" \
  -H "Content-Type: application/json" \
  -d "$payload_base" >/dev/null

echo "== whatif =="
curl -sf -X POST "$BASE/whatif" \
  -H "Content-Type: application/json" \
  -d "$(jq -c '. + {"purchase_prices":[24000000,25000000],"exit_cap_rates":[0.0625,0.0675],"max_scenarios":50,"sort_by":"cash_on_cash"}' <<<"$payload_base")" \
  >/dev/null

echo "== underwrite =="
curl -sf -X POST "$BASE/underwrite" \
  -H "Content-Type: application/json" \
  -d "$(jq -c '. + {"hold_years":5,"noi_growth":0.03}' <<<"$payload_base")" \
  >/dev/null

echo "== underwrite_inst =="
curl -sf -X POST "$BASE/underwrite_inst" \
  -H "Content-Type: application/json" \
  -d "$(jq -c '. + {
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
  }' <<<"$payload_base")" \
  >/dev/null

echo "== whatif_inst =="
curl -sf -X POST "$BASE/whatif_inst" \
  -H "Content-Type: application/json" \
  -d "$(jq -c '. + {
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
    "refi_cost_pct":0.01,

    "purchase_prices":[24000000,25000000],
    "exit_cap_rates":[0.0625,0.0675],
    "rent_growths":[0.02,0.03],
    "occupancy_shock_drops":[0.0,0.10],
    "rate_shock_bps_values":[0,150],
    "max_scenarios":100,
    "top_n":5,
    "sort_by":"irr"
  }' <<<"$payload_base")" \
  >/dev/null

echo "OK: all smoke tests passed"