# src/api.py  (additions/replacements)
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.roi import RoiInputs, compute_roi
from src.whatif import run_whatif

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="CRE NOI + ROI API", version="0.2")


class PredictRequest(BaseModel):
    deal_id: Optional[str] = None
    asof_date: Optional[str] = None  # YYYY-MM-DD

    property_type: str
    city: str
    state: str
    zip: str

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

    exit_cap_rate: float = Field(0.065, ge=0.0001, le=1.0)
    selling_cost_pct: float = Field(0.02, ge=0.0, le=0.2)


class PredictResponse(BaseModel):
    predicted_noi_next12: float
    roi: Dict[str, Any]


class WhatIfRequest(PredictRequest):
    # Scenario lists. If omitted, the base value from PredictRequest is used.
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


@app.post("/whatif", response_model=WhatIfResponse)
def whatif(req: WhatIfRequest):
    model = get_model()

    payload = req.model_dump()

    # Pull scenario lists + controls out first
    purchase_prices = payload.pop("purchase_prices", None)
    ltvs = payload.pop("ltvs", None)
    interest_rates = payload.pop("interest_rates", None)
    exit_cap_rates = payload.pop("exit_cap_rates", None)
    max_scenarios = payload.pop("max_scenarios", 200)
    sort_by = payload.pop("sort_by", "cash_on_cash")

    # ROI-only fields should not be features for prediction
    selling_cost_pct = payload.pop("selling_cost_pct")
    # exit_cap_rate is used only for ROI math; the base value is not needed for prediction
    payload.pop("exit_cap_rate", None)

    # run grid
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