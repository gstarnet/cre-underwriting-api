# src/whatif_inst.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class WhatIfInstScenarioResult:
    """
    Result for one institutional what-if scenario.

    - inputs: scenario inputs (price/ltv/rate/etc.)
    - predicted_noi_next12: ML anchor output (for comparison only)
    - institutional: underwriting output from underwrite_institutional()
    """
    inputs: Dict[str, Any]
    predicted_noi_next12: float
    institutional: Dict[str, Any]


def _as_list_or_default(values: Optional[Sequence[Any]], default_value: Any) -> List[Any]:
    """
    Normalize scenario vectors:
    - If values is provided and non-empty, use it.
    - Otherwise return [default_value] to keep scenario cartesian product working.
    """
    if values is None:
        return [default_value]
    values = list(values)
    return values if len(values) > 0 else [default_value]


def _cartesian_bounded(grid_axes: List[Tuple[str, List[Any]]], max_scenarios: int) -> Iterable[Dict[str, Any]]:
    """
    Generate a bounded Cartesian product over named axes.

    This yields dicts like {"purchase_price": ..., "ltv": ..., "rent_growth": ...}
    up to max_scenarios items.
    """
    keys = [k for k, _ in grid_axes]
    values = [v for _, v in grid_axes]

    count = 0
    for combo in product(*values):
        yield dict(zip(keys, combo))
        count += 1
        if count >= max_scenarios:
            break


def _sort_key(result: WhatIfInstScenarioResult, sort_by: str) -> float:
    """
    Sorting helper for scenario results.

    Supported:
    - irr
    - min_dscr
    - debt_yield_year1
    - net_sale_proceeds
    - cash_on_cash_year1
    """
    s = result.institutional["summary"]

    if sort_by == "irr":
        return float(s.get("irr", float("-inf")))
    if sort_by == "min_dscr":
        return float(s.get("min_dscr", float("-inf")))
    if sort_by == "debt_yield_year1":
        return float(s.get("debt_yield_year1", float("-inf")))
    if sort_by == "net_sale_proceeds":
        return float(s.get("net_sale_proceeds", float("-inf")))
    if sort_by == "cash_on_cash_year1":
        return float(s.get("cash_on_cash_year1", float("-inf")))

    # Default
    return float(s.get("irr", float("-inf")))


# Keep allowed values in one place (engine-level guardrails).
_ALLOWED_SORT_KEYS = {
    "irr",
    "min_dscr",
    "debt_yield_year1",
    "net_sale_proceeds",
    "cash_on_cash_year1",
}


def _estimate_grid_size(grid_axes: List[Tuple[str, List[Any]]]) -> int:
    """
    Estimate total scenario grid size (Cartesian product).

    Used for hard guardrails so callers don't accidentally request massive work.
    """
    est = 1
    for _, v in grid_axes:
        est *= max(1, len(v))
    return est


def run_whatif_inst(
        *,
        model: Any,
        base_ml_features: Dict[str, Any],
        base_inst_inputs: Dict[str, Any],
        # Primary financial axes
        purchase_prices: Optional[Sequence[float]] = None,
        ltvs: Optional[Sequence[float]] = None,
        interest_rates: Optional[Sequence[float]] = None,
        exit_cap_rates: Optional[Sequence[float]] = None,
        # Institutional axes (v2)
        rent_growths: Optional[Sequence[float]] = None,
        capex_reserve_per_sqft_values: Optional[Sequence[float]] = None,
        replacement_capex_per_sqft_values: Optional[Sequence[float]] = None,
        occupancy_shock_drops: Optional[Sequence[float]] = None,
        rate_shock_bps_values: Optional[Sequence[float]] = None,
        # Controls
        max_scenarios: int = 200,
        top_n: int = 50,
        sort_by: str = "irr",
) -> List[WhatIfInstScenarioResult]:
    """
    Run institutional what-if scenarios.

    Notes:
    - ML prediction is used only as an anchor output; institutional underwriting
      is driven by line items (gross rent, opex, occupancy, etc.).
    - base_ml_features should include ONLY the fields the model expects.
    - base_inst_inputs should include the InstUnderwriteInputs fields.

    Hardening behavior:
    - Reject invalid sort_by values.
    - Guard against exploding scenario grids.
    - Per-scenario error capture: one bad scenario should not crash the whole run.
      (If all scenarios fail, we raise a ValueError with a summary.)
    """
    from src.underwrite_inst import InstUnderwriteInputs, underwrite_institutional

    if sort_by not in _ALLOWED_SORT_KEYS:
        raise ValueError(f"sort_by must be one of {sorted(_ALLOWED_SORT_KEYS)}")

    max_scenarios = int(max_scenarios)
    top_n = int(top_n)
    if max_scenarios < 1:
        raise ValueError("max_scenarios must be >= 1")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    # Normalize axis vectors to at least one value (the baseline)
    purchase_prices_l = _as_list_or_default(purchase_prices, float(base_inst_inputs["purchase_price"]))
    ltvs_l = _as_list_or_default(ltvs, float(base_inst_inputs["ltv"]))
    rates_l = _as_list_or_default(interest_rates, float(base_inst_inputs["interest_rate"]))
    exit_caps_l = _as_list_or_default(exit_cap_rates, float(base_inst_inputs["exit_cap_rate"]))

    rent_growths_l = _as_list_or_default(rent_growths, float(base_inst_inputs.get("rent_growth", 0.03)))
    capex_res_l = _as_list_or_default(
        capex_reserve_per_sqft_values, float(base_inst_inputs.get("capex_reserve_per_sqft", 0.0))
    )
    repl_capex_l = _as_list_or_default(
        replacement_capex_per_sqft_values, float(base_inst_inputs.get("replacement_capex_per_sqft", 0.0))
    )
    occ_shock_l = _as_list_or_default(occupancy_shock_drops, float(base_inst_inputs.get("occupancy_shock_drop", 0.0)))
    rate_shock_l = _as_list_or_default(rate_shock_bps_values, float(base_inst_inputs.get("rate_shock_bps", 0.0)))

    grid_axes = [
        ("purchase_price", purchase_prices_l),
        ("ltv", ltvs_l),
        ("interest_rate", rates_l),
        ("exit_cap_rate", exit_caps_l),
        ("rent_growth", rent_growths_l),
        ("capex_reserve_per_sqft", capex_res_l),
        ("replacement_capex_per_sqft", repl_capex_l),
        ("occupancy_shock_drop", occ_shock_l),
        ("rate_shock_bps", rate_shock_l),
    ]

    # Hard guardrail against huge Cartesian products.
    est = _estimate_grid_size(grid_axes)
    # Allow some overhead vs max_scenarios (because we cap generation anyway),
    # but still block pathological requests.
    if est > max_scenarios * 5:
        raise ValueError(
            f"Scenario grid too large: estimated {est} combinations. "
            f"Reduce vector sizes or lower max_scenarios."
        )

    results: List[WhatIfInstScenarioResult] = []

    # Capture per-scenario exceptions; keep a short sample for diagnostics.
    errors: List[Dict[str, Any]] = []
    error_sample_limit = 5

    for overrides in _cartesian_bounded(grid_axes, max_scenarios=max_scenarios):
        try:
            # Build scenario ML features (only fields model expects)
            ml_features = dict(base_ml_features)

            # Build scenario underwriting inputs
            inst_kwargs = dict(base_inst_inputs)

            # Apply overrides
            inst_kwargs["purchase_price"] = float(overrides["purchase_price"])
            inst_kwargs["ltv"] = float(overrides["ltv"])
            inst_kwargs["interest_rate"] = float(overrides["interest_rate"])
            inst_kwargs["exit_cap_rate"] = float(overrides["exit_cap_rate"])
            inst_kwargs["rent_growth"] = float(overrides["rent_growth"])
            inst_kwargs["capex_reserve_per_sqft"] = float(overrides["capex_reserve_per_sqft"])
            inst_kwargs["replacement_capex_per_sqft"] = float(overrides["replacement_capex_per_sqft"])
            inst_kwargs["occupancy_shock_drop"] = float(overrides["occupancy_shock_drop"])
            inst_kwargs["rate_shock_bps"] = float(overrides["rate_shock_bps"])

            # ML feature set includes purchase_price/ltv/interest_rate (model might use these)
            ml_features["purchase_price"] = float(overrides["purchase_price"])
            ml_features["ltv"] = float(overrides["ltv"])
            ml_features["interest_rate"] = float(overrides["interest_rate"])

            # Predict NOI anchor
            X = pd.DataFrame([ml_features])
            pred_noi = float(model.predict(X)[0])

            # Run institutional underwriting
            uw = underwrite_institutional(InstUnderwriteInputs(**inst_kwargs))

            # Record scenario input summary (what changed)
            inputs_out = {
                "purchase_price": inst_kwargs["purchase_price"],
                "ltv": inst_kwargs["ltv"],
                "interest_rate": inst_kwargs["interest_rate"],
                "exit_cap_rate": inst_kwargs["exit_cap_rate"],
                "rent_growth": inst_kwargs["rent_growth"],
                "capex_reserve_per_sqft": inst_kwargs["capex_reserve_per_sqft"],
                "replacement_capex_per_sqft": inst_kwargs["replacement_capex_per_sqft"],
                "occupancy_shock_drop": inst_kwargs["occupancy_shock_drop"],
                "rate_shock_bps": inst_kwargs["rate_shock_bps"],
            }

            results.append(
                WhatIfInstScenarioResult(
                    inputs=inputs_out,
                    predicted_noi_next12=pred_noi,
                    institutional=uw,
                )
            )

        except Exception as e:
            # Per-scenario failure should not nuke the whole what-if run.
            if len(errors) < error_sample_limit:
                errors.append({"overrides": overrides, "error": f"{type(e).__name__}: {e}"})
            continue

    if not results:
        # Make failure actionable so callers can debug inputs quickly.
        msg = "All institutional what-if scenarios failed."
        if errors:
            msg += f" Sample errors: {errors}"
        raise ValueError(msg)

    # Sort and return top N
    results.sort(key=lambda r: _sort_key(r, sort_by), reverse=True)
    return results[: max(1, int(top_n))]
