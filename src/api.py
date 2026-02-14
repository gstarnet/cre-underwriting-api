# src/api.py
from __future__ import annotations

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Local modules
from src.roi import RoiInputs, compute_roi
from src.whatif import run_whatif
from src.underwrite import UnderwriteInputs, underwrite
from src.underwrite_inst import InstUnderwriteInputs, underwrite_institutional
from src.whatif_inst import run_whatif_inst

# Path to the trained sklearn Pipeline saved by src.train
MODEL_PATH = Path("models/model.joblib")

# FastAPI application object
app = FastAPI(title="CRE NOI + ROI API", version="0.7")

# Cached model instance (loaded once per process)
_model = None


def get_model():
    """
    Lazy-load and cache the trained model pipeline.

    - Loads models/model.joblib only once per process.
    - Raises a clear error if the model artifact is missing.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    """
    Lightweight health check endpoint.
    Useful for:
    - local smoke tests
    - container/orchestrator health probes
    """
    return {"status": "ok"}


# =============================================================================
# 1) /predict — typed request schema that matches your CRE deal snapshot
# =============================================================================
class PredictRequest(BaseModel):
    """
    Canonical typed input for the project.

    Notes:
    - deal_id and asof_date are accepted for usability, but are dropped by the
      model preprocessor (train pipeline explicitly drops deal_id/time column).
    - exit_cap_rate and selling_cost_pct are ROI assumptions, not ML features;
      they are removed before calling model.predict().
    """
    # Optional identifiers / time fields
    deal_id: Optional[str] = None
    asof_date: Optional[str] = None  # YYYY-MM-DD

    # Categorical features
    property_type: str
    city: str
    state: str
    zip: str

    # Numeric features
    year_built: int
    gross_leasable_sqft: float
    units: Optional[float] = None

    # Operating snapshot / pricing
    purchase_price: float
    noi_t12: float
    occupancy_t12: float
    opex_t12: float
    gross_rent_t12: float

    # Financing terms
    ltv: float = Field(..., ge=0.0, le=1.0)
    interest_rate: float = Field(..., ge=0.0, le=1.0)
    amort_years: int = Field(..., ge=1, le=40)

    # ROI assumptions (not used by the ML model)
    exit_cap_rate: float = Field(0.065, ge=0.0001, le=1.0)
    selling_cost_pct: float = Field(0.02, ge=0.0, le=0.2)


class PredictResponse(BaseModel):
    """
    Standard prediction response:
    - predicted_noi_next12: ML predicted NOI for the next 12 months
    - roi: simple 1-year cash flow + exit proceeds style ROI snapshot
    """
    predicted_noi_next12: float
    roi: Dict[str, Any]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict NOI_next12 and compute a simple ROI snapshot.

    Steps:
    1) Convert typed request into a DataFrame for the sklearn pipeline.
    2) Remove ROI-only fields (exit assumptions) from the model input.
    3) Predict NOI_next12.
    4) Compute ROI metrics from predicted NOI and financing + exit assumptions.
    """
    model = get_model()

    payload = req.model_dump()
    exit_cap_rate = payload.pop("exit_cap_rate")
    selling_cost_pct = payload.pop("selling_cost_pct")

    # Build a 1-row feature frame for sklearn Pipeline
    X = pd.DataFrame([payload])
    pred_noi = float(model.predict(X)[0])

    # Simple ROI math (year-1 cash flow + exit proceeds approximation)
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


# =============================================================================
# 2) /predict_features — generic dict-based input (future-proof / integrations)
# =============================================================================
class PredictFeaturesRequest(BaseModel):
    """
    Generic prediction request that accepts a free-form 'features' dict.

    This is useful for:
    - integrating with other pipelines where your feature keys already exist
    - partial payloads (you can omit ROI fields)
    - quick experiments without changing the typed schema
    """
    features: Dict[str, Any]
    exit_cap_rate: float = Field(0.065, ge=0.0001, le=1.0)
    selling_cost_pct: float = Field(0.02, ge=0.0, le=0.2)


class PredictFeaturesResponse(BaseModel):
    """
    - predicted_noi_next12 is always returned.
    - roi is returned only if required finance fields are present in features.
    """
    predicted_noi_next12: float
    roi: Optional[Dict[str, Any]] = None


@app.post("/predict_features", response_model=PredictFeaturesResponse)
def predict_features(req: PredictFeaturesRequest):
    """
    Predict NOI_next12 from a generic features dict.

    If the input includes:
      purchase_price, ltv, interest_rate, amort_years
    then ROI is computed too.
    """
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


# =============================================================================
# 3) /whatif — scenario grid for sensitivity analysis (simple ROI model)
# =============================================================================
class WhatIfRequest(PredictRequest):
    """
    Extends PredictRequest with scenario vectors.

    Any of the lists (purchase_prices, ltvs, interest_rates, exit_cap_rates)
    can be supplied to generate a Cartesian product scenario grid (bounded by
    max_scenarios). Scenarios are then sorted by 'sort_by'.
    """
    purchase_prices: Optional[List[float]] = None
    ltvs: Optional[List[float]] = None
    interest_rates: Optional[List[float]] = None
    exit_cap_rates: Optional[List[float]] = None

    max_scenarios: int = Field(200, ge=1, le=500)
    sort_by: str = Field("cash_on_cash")  # cash_on_cash | annual_cash_flow | estimated_exit_proceeds


class WhatIfScenario(BaseModel):
    """
    One what-if scenario result:
    - scenario inputs (price, ltv, rate, etc.)
    - predicted NOI
    - computed ROI metrics for that scenario
    """
    purchase_price: float
    ltv: float
    interest_rate: float
    amort_years: int
    exit_cap_rate: float
    selling_cost_pct: float
    predicted_noi_next12: float
    roi: Dict[str, Any]


class WhatIfResponse(BaseModel):
    """List of scenario results."""
    scenarios: List[WhatIfScenario]


@app.post("/whatif", response_model=WhatIfResponse)
def whatif(req: WhatIfRequest):
    """
    Generate what-if scenarios around a base deal snapshot.

    Implementation detail:
    - exit_cap_rate is ROI-only, not an ML feature, so it is removed before
      passing features into the model for prediction.
    - run_whatif() handles scenario generation, prediction, ROI math, sorting.
    """
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


# =============================================================================
# 4) /underwrite — simple multi-year underwriting (NOI grows at X%)
# =============================================================================
class UnderwriteRequest(PredictRequest):
    """
    Extends PredictRequest with a simple pro forma assumption:
    - noi_growth: constant annual NOI growth rate applied to predicted NOI.
    """
    hold_years: int = Field(5, ge=1, le=30)
    noi_growth: float = Field(0.03, ge=-0.20, le=0.30)


class UnderwriteResponse(BaseModel):
    """
    - predicted_noi_next12: ML predicted NOI for year 1
    - underwriting: multi-year schedule + exit + IRR based on simple NOI growth
    """
    predicted_noi_next12: float
    underwriting: Dict[str, Any]


@app.post("/underwrite", response_model=UnderwriteResponse)
def underwrite_endpoint(req: UnderwriteRequest):
    """
    Simple multi-year underwriting using:
    - ML-predicted year-1 NOI
    - a single NOI growth rate for years 2..N
    - amortizing debt and sale at year N
    """
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


# =============================================================================
# 5) /underwrite_inst — institutional-style underwriting (v2)
# =============================================================================
class UnderwriteInstRequest(PredictRequest):
    """
    Institutional underwriting request (v2).

    This endpoint runs:
    - a line-item pro forma (rent, occupancy path, opex, taxes, insurance)
    - capex buckets (recurring reserve + replacements + value-add schedule)
    - debt mechanics (IO, optional rate shock, optional refi)
    - exit proceeds + IRR

    The ML model is still called to return predicted NOI_next12 as an anchor,
    but the institutional pro forma is driven by the provided line items.
    """
    # Pro forma drivers
    rent_growth: float = Field(0.03, ge=-0.20, le=0.30)
    opex_inflation: float = Field(0.03, ge=-0.20, le=0.30)

    occupancy_target: float = Field(0.95, ge=0.0, le=1.0)
    occupancy_reversion_years: int = Field(2, ge=1, le=10)

    # Line-item taxes/insurance
    taxes_year1: float = Field(0.0, ge=0.0)
    insurance_year1: float = Field(0.0, ge=0.0)
    taxes_inflation: float = Field(0.03, ge=-0.10, le=0.30)
    insurance_inflation: float = Field(0.04, ge=-0.10, le=0.30)

    # Optional tax reassessment after purchase
    reassess_taxes: bool = False
    reassessed_tax_rate: float = Field(0.02, ge=0.0, le=0.10)
    reassess_year: int = Field(1, ge=1, le=30)

    # Capex v2
    capex_reserve_per_sqft: float = Field(0.0, ge=0.0)
    replacement_capex_per_sqft: float = Field(0.0, ge=0.0)
    value_add_capex: Optional[Dict[int, float]] = None  # year -> spend

    # Hold / debt
    hold_years: int = Field(5, ge=1, le=30)
    interest_only_years: int = Field(0, ge=0, le=10)

    # Occupancy shock v2 (optional)
    occupancy_shock_year: Optional[int] = Field(None, ge=1, le=30)
    occupancy_shock_drop: float = Field(0.0, ge=0.0, le=1.0)
    occupancy_recovery_years: int = Field(2, ge=1, le=10)

    # Rate shock v2 (optional)
    rate_shock_year: Optional[int] = Field(None, ge=1, le=30)
    rate_shock_bps: float = Field(0.0, ge=0.0, le=2000.0)

    # Refi v2 (optional)
    refi_year: Optional[int] = Field(None, ge=1, le=30)
    refi_ltv: float = Field(0.0, ge=0.0, le=1.0)
    refi_rate: float = Field(0.0, ge=0.0, le=1.0)
    refi_amort_years: int = Field(30, ge=1, le=40)
    refi_cost_pct: float = Field(0.01, ge=0.0, le=0.1)


class UnderwriteInstResponse(BaseModel):
    """
    - predicted_noi_next12: ML output (anchor / comparison)
    - institutional_underwriting: schedule + cash flows + summary (IRR, DSCR, credit metrics)
    """
    predicted_noi_next12: float
    institutional_underwriting: Dict[str, Any]


@app.post("/underwrite_inst", response_model=UnderwriteInstResponse)
def underwrite_inst_endpoint(req: UnderwriteInstRequest):
    """
    Institutional underwriting (v2).

    Flow:
    1) Extract/strip ROI-only fields (exit_cap_rate, selling_cost_pct) from model features.
    2) Extract all institutional knobs.
    3) Predict NOI_next12 (returned for reference).
    4) Run institutional underwriting using line-item pro forma + debt mechanics.
    """
    model = get_model()

    payload = req.model_dump()

    # ROI-only fields (not ML features)
    exit_cap_rate = payload.pop("exit_cap_rate")
    selling_cost_pct = payload.pop("selling_cost_pct")

    # Institutional knobs (remove before ML prediction)
    hold_years = payload.pop("hold_years")
    interest_only_years = payload.pop("interest_only_years")

    rent_growth = payload.pop("rent_growth")
    opex_inflation = payload.pop("opex_inflation")
    occupancy_target = payload.pop("occupancy_target")
    occupancy_reversion_years = payload.pop("occupancy_reversion_years")

    taxes_year1 = payload.pop("taxes_year1")
    insurance_year1 = payload.pop("insurance_year1")
    taxes_inflation = payload.pop("taxes_inflation")
    insurance_inflation = payload.pop("insurance_inflation")
    reassess_taxes = payload.pop("reassess_taxes")
    reassessed_tax_rate = payload.pop("reassessed_tax_rate")
    reassess_year = payload.pop("reassess_year")

    capex_reserve_per_sqft = payload.pop("capex_reserve_per_sqft")
    replacement_capex_per_sqft = payload.pop("replacement_capex_per_sqft")
    value_add_capex = payload.pop("value_add_capex")

    occupancy_shock_year = payload.pop("occupancy_shock_year")
    occupancy_shock_drop = payload.pop("occupancy_shock_drop")
    occupancy_recovery_years = payload.pop("occupancy_recovery_years")

    rate_shock_year = payload.pop("rate_shock_year")
    rate_shock_bps = payload.pop("rate_shock_bps")

    refi_year = payload.pop("refi_year")
    refi_ltv = payload.pop("refi_ltv")
    refi_rate = payload.pop("refi_rate")
    refi_amort_years = payload.pop("refi_amort_years")
    refi_cost_pct = payload.pop("refi_cost_pct")

    # ML anchor prediction (only uses the base deal snapshot features)
    X = pd.DataFrame([payload])
    pred_noi = float(model.predict(X)[0])

    # Institutional pro forma underwriting
    uw = underwrite_institutional(
        InstUnderwriteInputs(
            purchase_price=req.purchase_price,
            ltv=req.ltv,
            interest_rate=req.interest_rate,
            amort_years=req.amort_years,
            gross_rent_year1=req.gross_rent_t12,
            opex_year1=req.opex_t12,
            occupancy_year1=req.occupancy_t12,
            gross_leasable_sqft=req.gross_leasable_sqft,
            hold_years=hold_years,
            exit_cap_rate=exit_cap_rate,
            selling_cost_pct=selling_cost_pct,
            interest_only_years=interest_only_years,
            rent_growth=rent_growth,
            opex_inflation=opex_inflation,
            occupancy_target=occupancy_target,
            occupancy_reversion_years=occupancy_reversion_years,
            taxes_year1=taxes_year1,
            insurance_year1=insurance_year1,
            taxes_inflation=taxes_inflation,
            insurance_inflation=insurance_inflation,
            reassess_taxes=reassess_taxes,
            reassessed_tax_rate=reassessed_tax_rate,
            reassess_year=reassess_year,
            capex_reserve_per_sqft=capex_reserve_per_sqft,
            replacement_capex_per_sqft=replacement_capex_per_sqft,
            value_add_capex=value_add_capex,
            occupancy_shock_year=occupancy_shock_year,
            occupancy_shock_drop=occupancy_shock_drop,
            occupancy_recovery_years=occupancy_recovery_years,
            rate_shock_year=rate_shock_year,
            rate_shock_bps=rate_shock_bps,
            refi_year=refi_year,
            refi_ltv=refi_ltv,
            refi_rate=refi_rate,
            refi_amort_years=refi_amort_years,
            refi_cost_pct=refi_cost_pct,
        )
    )

    return UnderwriteInstResponse(predicted_noi_next12=pred_noi, institutional_underwriting=uw)


# =============================================================================
# 6) /whatif_inst — institutional what-if (v2) + credit metrics sorting
# =============================================================================
class WhatIfInstRequest(UnderwriteInstRequest):
    """
    Institutional what-if request (v2).

    Adds scenario vectors on top of UnderwriteInstRequest.

    The endpoint:
    - creates scenario combinations (bounded by max_scenarios)
    - runs institutional underwriting per scenario
    - sorts results by sort_by
    - returns the top N
    """
    # Primary axes
    purchase_prices: Optional[List[float]] = None
    ltvs: Optional[List[float]] = None
    interest_rates: Optional[List[float]] = None
    exit_cap_rates: Optional[List[float]] = None

    # Institutional axes
    rent_growths: Optional[List[float]] = None
    capex_reserve_per_sqft_values: Optional[List[float]] = None
    replacement_capex_per_sqft_values: Optional[List[float]] = None
    occupancy_shock_drops: Optional[List[float]] = None
    rate_shock_bps_values: Optional[List[float]] = None

    # Controls
    max_scenarios: int = Field(200, ge=1, le=500)
    top_n: int = Field(50, ge=1, le=200)
    sort_by: str = Field("irr")  # irr | min_dscr | debt_yield_year1 | net_sale_proceeds | cash_on_cash_year1


class WhatIfInstScenario(BaseModel):
    """
    One institutional what-if scenario result.

    - inputs: scenario inputs (what changed)
    - predicted_noi_next12: ML anchor output
    - summary: institutional summary (includes DSCR/IRR/credit metrics)
    """
    inputs: Dict[str, Any]
    predicted_noi_next12: float
    summary: Dict[str, Any]


class WhatIfInstResponse(BaseModel):
    """
    Institutional what-if response:
    - scenarios: top ranked scenarios by sort_by
    """
    scenarios: List[WhatIfInstScenario]


@app.post("/whatif_inst", response_model=WhatIfInstResponse)
def whatif_inst(req: WhatIfInstRequest):
    """
    Institutional what-if scenarios (v2).

    Key idea:
    - The institutional underwriting is driven by provided line items and pro forma knobs.
    - The ML NOI prediction is returned only as an anchor/comparison.
    """
    model = get_model()

    # Start from request data; then peel off vectors and build ML + underwriting payloads.
    payload = req.model_dump()

    # Scenario vectors
    purchase_prices = payload.pop("purchase_prices", None)
    ltvs = payload.pop("ltvs", None)
    interest_rates = payload.pop("interest_rates", None)
    exit_cap_rates = payload.pop("exit_cap_rates", None)

    rent_growths = payload.pop("rent_growths", None)
    capex_reserve_per_sqft_values = payload.pop("capex_reserve_per_sqft_values", None)
    replacement_capex_per_sqft_values = payload.pop("replacement_capex_per_sqft_values", None)
    occupancy_shock_drops = payload.pop("occupancy_shock_drops", None)
    rate_shock_bps_values = payload.pop("rate_shock_bps_values", None)

    max_scenarios = int(payload.pop("max_scenarios", 200))
    top_n = int(payload.pop("top_n", 50))
    sort_by = str(payload.pop("sort_by", "irr"))

    # ROI-only fields (not ML features)
    exit_cap_rate = payload.pop("exit_cap_rate")
    selling_cost_pct = payload.pop("selling_cost_pct")

    # Build ML features:
    # - Remove institutional knobs that the model does not need.
    # - Keep the base deal snapshot keys consistent with training.
    #
    # Note: These keys must match what your training pipeline expects.
    inst_keys = {
        "hold_years", "interest_only_years",
        "rent_growth", "opex_inflation", "occupancy_target", "occupancy_reversion_years",
        "taxes_year1", "insurance_year1", "taxes_inflation", "insurance_inflation",
        "reassess_taxes", "reassessed_tax_rate", "reassess_year",
        "capex_reserve_per_sqft", "replacement_capex_per_sqft", "value_add_capex",
        "occupancy_shock_year", "occupancy_shock_drop", "occupancy_recovery_years",
        "rate_shock_year", "rate_shock_bps",
        "refi_year", "refi_ltv", "refi_rate", "refi_amort_years", "refi_cost_pct",
    }

    base_ml_features = {k: v for k, v in payload.items() if k not in inst_keys}

    # Build base institutional inputs for InstUnderwriteInputs:
    base_inst_inputs = {
        "purchase_price": float(req.purchase_price),
        "ltv": float(req.ltv),
        "interest_rate": float(req.interest_rate),
        "amort_years": int(req.amort_years),

        "gross_rent_year1": float(req.gross_rent_t12),
        "opex_year1": float(req.opex_t12),
        "occupancy_year1": float(req.occupancy_t12),
        "gross_leasable_sqft": float(req.gross_leasable_sqft),

        "hold_years": int(req.hold_years),
        "exit_cap_rate": float(exit_cap_rate),
        "selling_cost_pct": float(selling_cost_pct),

        "interest_only_years": int(req.interest_only_years),

        "rent_growth": float(req.rent_growth),
        "opex_inflation": float(req.opex_inflation),
        "occupancy_target": float(req.occupancy_target),
        "occupancy_reversion_years": int(req.occupancy_reversion_years),

        "taxes_year1": float(req.taxes_year1),
        "insurance_year1": float(req.insurance_year1),
        "taxes_inflation": float(req.taxes_inflation),
        "insurance_inflation": float(req.insurance_inflation),
        "reassess_taxes": bool(req.reassess_taxes),
        "reassessed_tax_rate": float(req.reassessed_tax_rate),
        "reassess_year": int(req.reassess_year),

        "capex_reserve_per_sqft": float(req.capex_reserve_per_sqft),
        "replacement_capex_per_sqft": float(req.replacement_capex_per_sqft),
        "value_add_capex": req.value_add_capex,

        "occupancy_shock_year": req.occupancy_shock_year,
        "occupancy_shock_drop": float(req.occupancy_shock_drop),
        "occupancy_recovery_years": int(req.occupancy_recovery_years),

        "rate_shock_year": req.rate_shock_year,
        "rate_shock_bps": float(req.rate_shock_bps),

        "refi_year": req.refi_year,
        "refi_ltv": float(req.refi_ltv),
        "refi_rate": float(req.refi_rate),
        "refi_amort_years": int(req.refi_amort_years),
        "refi_cost_pct": float(req.refi_cost_pct),
    }

    results = run_whatif_inst(
        model=model,
        base_ml_features=base_ml_features,
        base_inst_inputs=base_inst_inputs,
        purchase_prices=purchase_prices,
        ltvs=ltvs,
        interest_rates=interest_rates,
        exit_cap_rates=exit_cap_rates,
        rent_growths=rent_growths,
        capex_reserve_per_sqft_values=capex_reserve_per_sqft_values,
        replacement_capex_per_sqft_values=replacement_capex_per_sqft_values,
        occupancy_shock_drops=occupancy_shock_drops,
        rate_shock_bps_values=rate_shock_bps_values,
        max_scenarios=max_scenarios,
        top_n=top_n,
        sort_by=sort_by,
    )

    return WhatIfInstResponse(
        scenarios=[
            WhatIfInstScenario(
                inputs=r.inputs,
                predicted_noi_next12=r.predicted_noi_next12,
                summary=r.institutional["summary"],
            )
            for r in results
        ]
    )