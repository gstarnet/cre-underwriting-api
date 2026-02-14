# src/underwrite_inst.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import math


@dataclass(frozen=True)
class InstUnderwriteInputs:
    # REQUIRED (no defaults) — must come first
    purchase_price: float
    ltv: float
    interest_rate: float          # annual
    amort_years: int

    gross_rent_year1: float       # effective gross rent year 1 (can be t12 baseline)
    opex_year1: float             # operating expenses year 1
    occupancy_year1: float        # 0..1

    gross_leasable_sqft: float

    # DEFAULTED (safe to follow)
    hold_years: int = 5
    interest_only_years: int = 0  # 0 = fully amortizing from day 1

    rent_growth: float = 0.03     # annual growth on market rents
    opex_inflation: float = 0.03  # annual growth on opex
    occupancy_target: float = 0.95
    occupancy_reversion_years: int = 2  # years to reach target linearly

    taxes_year1: float = 0.0
    insurance_year1: float = 0.0
    taxes_inflation: float = 0.03
    insurance_inflation: float = 0.04

    reassess_taxes: bool = False
    reassessed_tax_rate: float = 0.02  # e.g., 2% of purchase price
    reassess_year: int = 1             # year index to apply reassessment (usually year 1 or 2)

    capex_reserve_per_sqft: float = 0.0
    capex_one_time: Optional[Dict[int, float]] = None

    exit_cap_rate: float = 0.065
    selling_cost_pct: float = 0.02


def annual_debt_service_amortizing(loan_amount: float, annual_rate: float, amort_years: int) -> float:
    if loan_amount <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = amort_years * 12
    if r == 0:
        return float(loan_amount / amort_years)
    pmt_m = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return float(pmt_m * 12.0)


def remaining_balance_amortizing(loan_amount: float, annual_rate: float, amort_years: int, months_paid: int) -> float:
    if loan_amount <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = amort_years * 12
    k = min(max(int(months_paid), 0), n)
    if r == 0:
        paid = loan_amount * (k / n)
        return float(max(loan_amount - paid, 0.0))
    pmt = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    bal = loan_amount * (1 + r) ** k - pmt * (((1 + r) ** k - 1) / r)
    return float(max(bal, 0.0))


def irr(cash_flows: List[float], guess: float = 0.12) -> float:
    def npv(rate: float) -> float:
        return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))

    def d_npv(rate: float) -> float:
        return sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cash_flows) if t > 0)

    r = guess
    for _ in range(100):
        f = npv(r)
        df = d_npv(r)
        if abs(df) < 1e-12:
            break
        new_r = r - f / df
        if not math.isfinite(new_r):
            break
        if abs(new_r - r) < 1e-10:
            r = new_r
            break
        r = new_r
    return float(r)


def _occupancy_for_year(y: int, occ1: float, target: float, reversion_years: int) -> float:
    if reversion_years <= 0:
        return float(target)
    if y <= 1:
        return float(occ1)
    if y >= reversion_years:
        return float(target)
    step = (target - occ1) / float(reversion_years - 1)
    return float(occ1 + step * (y - 1))


def underwrite_institutional(x: InstUnderwriteInputs) -> Dict[str, Any]:
    loan = x.purchase_price * x.ltv
    equity = x.purchase_price - loan

    io_years = max(0, min(x.interest_only_years, x.hold_years))
    amort_ads = annual_debt_service_amortizing(loan, x.interest_rate, x.amort_years)
    io_ads = float(loan * x.interest_rate)

    schedule: List[Dict[str, Any]] = []
    cash_flows: List[float] = [-float(equity)]

    for year in range(1, x.hold_years + 1):
        occ = _occupancy_for_year(year, x.occupancy_year1, x.occupancy_target, x.occupancy_reversion_years)
        occ = max(0.0, min(1.0, occ))

        gross_rent = x.gross_rent_year1 * ((1 + x.rent_growth) ** (year - 1))
        effective_rent = gross_rent * occ

        opex = x.opex_year1 * ((1 + x.opex_inflation) ** (year - 1))

        if x.reassess_taxes and year >= x.reassess_year:
            taxes_base = x.purchase_price * x.reassessed_tax_rate
        else:
            taxes_base = x.taxes_year1
        taxes = taxes_base * ((1 + x.taxes_inflation) ** (year - 1))

        insurance = x.insurance_year1 * ((1 + x.insurance_inflation) ** (year - 1))

        capex_reserve = x.capex_reserve_per_sqft * x.gross_leasable_sqft
        capex_one_time = 0.0
        if x.capex_one_time and year in x.capex_one_time:
            capex_one_time = float(x.capex_one_time[year])

        noi = effective_rent - opex - taxes - insurance - capex_reserve
        debt_service = io_ads if year <= io_years else amort_ads
        cf = noi - debt_service - capex_one_time
        dscr = (noi / debt_service) if debt_service > 0 else math.inf

        schedule.append(
            {
                "year": year,
                "occupancy": occ,
                "gross_rent": float(gross_rent),
                "effective_rent": float(effective_rent),
                "opex": float(opex),
                "taxes": float(taxes),
                "insurance": float(insurance),
                "capex_reserve": float(capex_reserve),
                "capex_one_time": float(capex_one_time),
                "noi": float(noi),
                "debt_service": float(debt_service),
                "cash_flow": float(cf),
                "dscr": float(dscr),
            }
        )

        cash_flows.append(float(cf))

    noi_exit = schedule[-1]["noi"]
    exit_value = noi_exit / x.exit_cap_rate if x.exit_cap_rate > 0 else math.inf
    selling_costs = exit_value * x.selling_cost_pct

    amort_years_paid = max(0, x.hold_years - io_years)
    months_paid = amort_years_paid * 12
    bal = (
        remaining_balance_amortizing(loan, x.interest_rate, x.amort_years, months_paid)
        if amort_years_paid > 0
        else float(loan)
    )

    net_sale_proceeds = exit_value - selling_costs - bal
    cash_flows[-1] = cash_flows[-1] + float(net_sale_proceeds)

    irr_val = irr(cash_flows, guess=0.12)

    summary = {
        "loan_amount": float(loan),
        "equity_required": float(equity),
        "interest_only_years": int(io_years),
        "amort_annual_debt_service": float(amort_ads),
        "io_annual_debt_service": float(io_ads),
        "exit_value": float(exit_value),
        "selling_costs": float(selling_costs),
        "remaining_loan_balance_at_sale": float(bal),
        "net_sale_proceeds": float(net_sale_proceeds),
        "irr": float(irr_val),
        "cash_on_cash_year1": float(schedule[0]["cash_flow"] / equity) if equity > 0 else math.inf,
        "min_dscr": float(min(s["dscr"] for s in schedule)),
    }

    return {"summary": summary, "schedule": schedule, "cash_flows": cash_flows}