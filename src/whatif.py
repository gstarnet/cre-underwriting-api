# src/whatif.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.roi import RoiInputs, compute_roi


@dataclass(frozen=True)
class ScenarioResult:
    purchase_price: float
    ltv: float
    interest_rate: float
    amort_years: int
    exit_cap_rate: float
    selling_cost_pct: float
    predicted_noi_next12: float
    roi: Dict[str, Any]


def _dedupe_sorted(values: Optional[Iterable[float]]) -> List[float]:
    if values is None:
        return []
    cleaned = []
    for v in values:
        if v is None:
            continue
        cleaned.append(float(v))
    return sorted(set(cleaned))


def run_whatif(
        *,
        model,
        base_features: Dict[str, Any],
        purchase_prices: Optional[Iterable[float]] = None,
        ltvs: Optional[Iterable[float]] = None,
        interest_rates: Optional[Iterable[float]] = None,
        exit_cap_rates: Optional[Iterable[float]] = None,
        selling_cost_pct: float = 0.02,
        amort_years: int,
        max_scenarios: int = 300,
        sort_by: str = "cash_on_cash",  # or "annual_cash_flow"
) -> List[ScenarioResult]:
    """
    Generates a scenario grid, predicts NOI for each scenario, and computes ROI.

    base_features: the full feature dict used for prediction (same keys as training),
                   including purchase_price/ltv/interest_rate/amort_years.
    """

    pp_list = _dedupe_sorted(purchase_prices) or [float(base_features["purchase_price"])]
    ltv_list = _dedupe_sorted(ltvs) or [float(base_features["ltv"])]
    ir_list = _dedupe_sorted(interest_rates) or [float(base_features["interest_rate"])]
    exit_caps = _dedupe_sorted(exit_cap_rates) or [0.065]

    results: List[ScenarioResult] = []

    for pp in pp_list:
        for ltv in ltv_list:
            for ir in ir_list:
                for exit_cap in exit_caps:
                    if len(results) >= max_scenarios:
                        break

                    # Update features for this scenario
                    features = dict(base_features)
                    features["purchase_price"] = pp
                    features["ltv"] = ltv
                    features["interest_rate"] = ir
                    features["amort_years"] = amort_years

                    X = pd.DataFrame([features])
                    pred_noi = float(model.predict(X)[0])

                    roi_out = compute_roi(
                        RoiInputs(
                            purchase_price=pp,
                            noi_next12=pred_noi,
                            ltv=ltv,
                            interest_rate=ir,
                            amort_years=amort_years,
                            exit_cap_rate=exit_cap,
                            selling_cost_pct=selling_cost_pct,
                        )
                    )

                    results.append(
                        ScenarioResult(
                            purchase_price=pp,
                            ltv=ltv,
                            interest_rate=ir,
                            amort_years=amort_years,
                            exit_cap_rate=exit_cap,
                            selling_cost_pct=selling_cost_pct,
                            predicted_noi_next12=pred_noi,
                            roi=roi_out.__dict__,
                        )
                    )

    key_map = {
        "cash_on_cash": lambda r: r.roi.get("cash_on_cash", float("-inf")),
        "annual_cash_flow": lambda r: r.roi.get("annual_cash_flow", float("-inf")),
        "estimated_exit_proceeds": lambda r: r.roi.get(
            "estimated_exit_proceeds_after_debt_and_costs", float("-inf")
        ),
    }
    sort_key = key_map.get(sort_by, key_map["cash_on_cash"])
    results.sort(key=sort_key, reverse=True)
    return results