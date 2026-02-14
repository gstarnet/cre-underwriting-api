# src/api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.roi import RoiInputs, compute_roi
from src.whatif import run_whatif
from src.underwrite import UnderwriteInputs, underwrite

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="CRE NOI + ROI API", version="0.4")

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------
# Typed prediction schema
# -----------------------
class PredictRequest(BaseModel):
    # Identifiers / time (kept for compatibility; model drops these)
    deal_id: Optional[str] = None
    asof_date: Optional[str] = None  # YYYY-MM-DD

    # Categorical
    property_type: str
    city: str
    state: str
    zip: str

    # Numeric
    year_built: int
    gross_leasable_sqft: float
    units: Optional[float] = None

    purchase_price: float
    noi_t12: float
    occupancy_t12: float
    opex_t12: float
    gross_rent_t12: float

    ltv: float = Field(..., ge=0.0, le=1.0)
    interest_rate: float = Field(..., ge=0.0, le=1.0)
    amort_years: int = Field(..., ge=1, le=40)

    # ROI assumptions (not used by ML model)
    exit_cap_rate: float = Field(0.065, ge=0.0001, le=1.0)
    selling_cost_pct: float = Field(0.02, ge=0.0, le=0.2)


class PredictResponse(BaseModel):
    predicted_noi_next12: float
    roi: Dict[str, Any]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()

    payload = req.model_dump()
    exit_cap_rate = payload.pop("exit_cap_rate")
    selling_cost_pct = payload.pop("selling_cost_pct")

    X = pd.DataFrame([payload])
    pred_noi = float(model.predict(X)[0])

    roi_out = compute_roi(
        RoiInputs(
            purchase_price=req.purchase_price,
            noi_next12=pred_noi,
            ltv=req.ltv,
            interest_rate=req.interest_rate,
            amort_years=req.amort_years,
            exit_cap_rate=exit_cap_rate,
            selling_cost_pct=selling_cost_pct,
        )
    )

    return PredictResponse(predicted_noi_next12=pred_noi, roi=roi_out.__dict__)


# -----------------------------------------
# Generic prediction schema (future-proof)
# -----------------------------------------
class PredictFeaturesRequest(BaseModel):
    features: Dict[str, Any]
    exit_cap_rate: float = Field(0.065, ge=0.0001, le=1.0)
    selling_cost_pct: float = Field(0.02, ge=0.0, le=0.2)


class PredictFeaturesResponse(BaseModel):
    predicted_noi_next12: float
    roi: Optional[Dict[str, Any]] = None


@app.post("/predict_features", response_model=PredictFeaturesResponse)
def predict_features(req: PredictFeaturesRequest):
    model = get_model()

    X = pd.DataFrame([req.features])
    pred_noi = float(model.predict(X)[0])

    roi = None
    f = req.features
    required = ["purchase_price", "ltv", "interest_rate", "amort_years"]
    if all(k in f and f[k] is not None for k in required):
        roi_out = compute_roi(
            RoiInputs(
                purchase_price=float(f["purchase_price"]),
                noi_next12=pred_noi,
                ltv=float(f["ltv"]),
                interest_rate=float(f["interest_rate"]),
                amort_years=int(f["amort_years"]),
                exit_cap_rate=float(req.exit_cap_rate),
                selling_cost_pct=float(req.selling_cost_pct),
            )
        )
        roi = roi_out.__dict__

    return PredictFeaturesResponse(predicted_noi_next12=pred_noi, roi=roi)


# -----------------------
# What-if sensitivity API
# -----------------------
class WhatIfRequest(PredictRequest):
    purchase_prices: Optional[List[float]] = None
    ltvs: Optional[List[float]] = None
    interest_rates: Optional[List[float]] = None
    exit_cap_rates: Optional[List[float]] = None

    max_scenarios: int = Field(200, ge=1, le=500)
    sort_by: str = Field("cash_on_cash")  # cash_on_cash | annual_cash_flow | estimated_exit_proceeds


class WhatIfScenario(BaseModel):
    purchase_price: float
    ltv: float
    interest_rate: float
    amort_years: int
    exit_cap_rate: float
    selling_cost_pct: float
    predicted_noi_next12: float
    roi: Dict[str, Any]


class WhatIfResponse(BaseModel):
    scenarios: List[WhatIfScenario]


@app.post("/whatif", response_model=WhatIfResponse)
def whatif(req: WhatIfRequest):
    model = get_model()

    payload = req.model_dump()

    purchase_prices = payload.pop("purchase_prices", None)
    ltvs = payload.pop("ltvs", None)
    interest_rates = payload.pop("interest_rates", None)
    exit_cap_rates = payload.pop("exit_cap_rates", None)
    max_scenarios = payload.pop("max_scenarios", 200)
    sort_by = payload.pop("sort_by", "cash_on_cash")

    selling_cost_pct = payload.pop("selling_cost_pct")
    payload.pop("exit_cap_rate", None)  # ROI-only; not a model feature

    results = run_whatif(
        model=model,
        base_features=payload,
        purchase_prices=purchase_prices,
        ltvs=ltvs,
        interest_rates=interest_rates,
        exit_cap_rates=exit_cap_rates,
        selling_cost_pct=selling_cost_pct,
        amort_years=req.amort_years,
        max_scenarios=max_scenarios,
        sort_by=sort_by,
    )

    return WhatIfResponse(
        scenarios=[
            WhatIfScenario(
                purchase_price=r.purchase_price,
                ltv=r.ltv,
                interest_rate=r.interest_rate,
                amort_years=r.amort_years,
                exit_cap_rate=r.exit_cap_rate,
                selling_cost_pct=r.selling_cost_pct,
                predicted_noi_next12=r.predicted_noi_next12,
                roi=r.roi,
            )
            for r in results
        ]
    )


# -----------------------
# Multi-year underwriting
# -----------------------
class UnderwriteRequest(PredictRequest):
    hold_years: int = Field(5, ge=1, le=30)
    noi_growth: float = Field(0.03, ge=-0.20, le=0.30)


class UnderwriteResponse(BaseModel):
    predicted_noi_next12: float
    underwriting: Dict[str, Any]


@app.post("/underwrite", response_model=UnderwriteResponse)
def underwrite_endpoint(req: UnderwriteRequest):
    model = get_model()

    payload = req.model_dump()
    exit_cap_rate = payload.pop("exit_cap_rate")
    selling_cost_pct = payload.pop("selling_cost_pct")
    hold_years = payload.pop("hold_years")
    noi_growth = payload.pop("noi_growth")

    X = pd.DataFrame([payload])
    pred_noi = float(model.predict(X)[0])

    uw = underwrite(
        UnderwriteInputs(
            purchase_price=req.purchase_price,
            noi_year1=pred_noi,
            ltv=req.ltv,
            interest_rate=req.interest_rate,
            amort_years=req.amort_years,
            hold_years=hold_years,
            noi_growth=noi_growth,
            exit_cap_rate=exit_cap_rate,
            selling_cost_pct=selling_cost_pct,
        )
    )

    return UnderwriteResponse(predicted_noi_next12=pred_noi, underwriting=uw)