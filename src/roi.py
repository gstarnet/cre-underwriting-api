# src/roi.py
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RoiInputs:
    purchase_price: float
    noi_next12: float  # predicted NOI next 12 months
    ltv: float
    interest_rate: float  # annual, e.g. 0.065
    amort_years: int
    exit_cap_rate: float  # e.g. 0.065
    selling_cost_pct: float = 0.02  # default 2%


@dataclass(frozen=True)
class RoiOutputs:
    loan_amount: float
    equity_required: float
    annual_debt_service: float
    annual_cash_flow: float
    cash_on_cash: float
    estimated_exit_value: float
    estimated_exit_proceeds_after_debt_and_costs: float


def annual_debt_service(loan_amount: float, annual_rate: float, amort_years: int) -> float:
    """
    Standard amortizing loan payment (annualized).
    """
    if loan_amount <= 0:
        return 0.0

    r = annual_rate / 12.0
    n = amort_years * 12

    if r == 0:
        # Simple straight-line paydown approximation (rare in practice)
        return float(loan_amount / amort_years)

    payment_m = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return float(payment_m * 12.0)


def compute_roi(x: RoiInputs) -> RoiOutputs:
    # Basic equity/debt
    loan = x.purchase_price * x.ltv
    equity = x.purchase_price - loan

    # 1-year cash flow
    ads = annual_debt_service(loan, x.interest_rate, x.amort_years)
    cash_flow = x.noi_next12 - ads
    coc = cash_flow / equity if equity > 0 else math.inf

    # Simple exit value estimate (cap valuation)
    exit_value = x.noi_next12 / x.exit_cap_rate if x.exit_cap_rate > 0 else math.inf
    selling_costs = exit_value * x.selling_cost_pct

    # Simplified proceeds (does not model principal paydown over time)
    exit_proceeds = exit_value - selling_costs - loan

    return RoiOutputs(
        loan_amount=float(loan),
        equity_required=float(equity),
        annual_debt_service=float(ads),
        annual_cash_flow=float(cash_flow),
        cash_on_cash=float(coc),
        estimated_exit_value=float(exit_value),
        estimated_exit_proceeds_after_debt_and_costs=float(exit_proceeds),
    )