from __future__ import annotations

import importlib
import json
from pathlib import Path

import joblib
import pytest
from fastapi import HTTPException


class DummyModel:
    def predict(self, X):
        return [123456.0 for _ in range(len(X))]


def _load_api(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    explain_path = tmp_path / "feature_importance.json"
    joblib.dump(DummyModel(), model_path)
    explain_path.write_text(
        json.dumps(
            {
                "method": "permutation_importance",
                "n_repeats": 10,
                "scoring": "neg_mean_absolute_error",
                "rows": [{"feature": "purchase_price", "importance_mean": 1.0, "importance_std": 0.1}],
                "model_version": "v1",
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "dataset_hash": "abc123",
            }
        ),
        encoding="utf-8",
    )

    import os

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["EXPLAINABILITY_JSON_PATH"] = str(explain_path)
    os.environ["AUTH_MODE"] = "none"

    import src.config
    import src.api

    importlib.reload(src.config)
    return importlib.reload(src.api)


def _predict_payload() -> dict:
    return {
        "deal_id": "D1",
        "asof_date": "2024-06-01",
        "property_type": "Industrial",
        "city": "Tampa",
        "state": "FL",
        "zip": "33602",
        "year_built": 2008,
        "gross_leasable_sqft": 125000,
        "units": None,
        "purchase_price": 25000000,
        "noi_t12": 1500000,
        "occupancy_t12": 0.95,
        "opex_t12": 450000,
        "gross_rent_t12": 2200000,
        "ltv": 0.65,
        "interest_rate": 0.062,
        "amort_years": 25,
        "exit_cap_rate": 0.065,
        "selling_cost_pct": 0.02,
    }


def _inst_payload() -> dict:
    out = _predict_payload()
    out.update(
        {
            "hold_years": 5,
            "interest_only_years": 1,
            "rent_growth": 0.03,
            "opex_inflation": 0.03,
            "occupancy_target": 0.97,
            "occupancy_reversion_years": 2,
            "taxes_year1": 350000,
            "insurance_year1": 60000,
            "reassess_taxes": True,
            "reassessed_tax_rate": 0.02,
            "reassess_year": 1,
            "capex_reserve_per_sqft": 0.25,
            "replacement_capex_per_sqft": 0.15,
            "value_add_capex": {"2": 250000},
            "occupancy_shock_year": 2,
            "occupancy_shock_drop": 0.10,
            "occupancy_recovery_years": 2,
            "rate_shock_year": 3,
            "rate_shock_bps": 150,
            "refi_year": 3,
            "refi_ltv": 0.65,
            "refi_rate": 0.06,
            "refi_amort_years": 25,
            "refi_cost_pct": 0.01,
        }
    )
    return out


def test_predict_happy_path(tmp_path):
    mod = _load_api(tmp_path)
    req = mod.PredictRequest(**_predict_payload())
    resp = mod.predict(req)
    assert resp.predicted_noi_next12 == 123456.0
    assert resp.roi["loan_amount"] > 0


def test_explainability_includes_traceability_fields(tmp_path):
    mod = _load_api(tmp_path)
    payload = mod.explainability()
    assert payload["model_version"] == "v1"
    assert payload["trained_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert payload["dataset_hash"] == "abc123"


def test_whatif_inst_invalid_sort_key_raises_422(tmp_path):
    mod = _load_api(tmp_path)
    bad = _inst_payload()
    bad["sort_by"] = "not-valid"
    req = mod.WhatIfInstRequest(**bad)
    with pytest.raises(HTTPException) as exc:
        mod.whatif_inst(req)
    assert exc.value.status_code == 422
