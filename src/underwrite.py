# src/underwrite.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import math


@dataclass(frozen=True)
class UnderwriteInputs:
    purchase_price: float
    noi_year1: float              # predicted NOI next 12 months
    ltv: float
    interest_rate: float          # annual, e.g. 0.062
    amort_years: int

    hold_years: int = 5
    noi_growth: float = 0.03      # annual NOI growth assumption
    exit_cap_rate: float = 0.065
    selling_cost_pct: float = 0.02


@dataclass(frozen=True)
class UnderwriteSummary:
    loan_amount: float
    equity_required: float
    annual_debt_service: float
    exit_value: float
    selling_costs: float
    remaining_loan_balance_at_sale: float
    net_sale_proceeds: float
    irr_5yr: float
    cash_on_cash_year1: float


def annual_debt_service(loan_amount: float, annual_rate: float, amort_years: int) -> float:
    if loan_amount <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = amort_years * 12
    if r == 0:
        return float(loan_amount / amort_years)
    pmt_m = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return float(pmt_m * 12.0)


def remaining_balance(loan_amount: float, annual_rate: float, amort_years: int, months_paid: int) -> float:
    """
    Remaining principal after 'months_paid' payments on a fully-amortizing loan.
    """
    if loan_amount <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = amort_years * 12
    k = min(max(int(months_paid), 0), n)

    if r == 0:
        # straight-line principal paydown
        paid = loan_amount * (k / n)
        return float(max(loan_amount - paid, 0.0))

    pmt = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    bal = loan_amount * (1 + r) ** k - pmt * (((1 + r) ** k - 1) / r)
    return float(max(bal, 0.0))


def irr(cash_flows: List[float], guess: float = 0.12) -> float:
    """
    Newton-Raphson IRR solver.
    cash_flows[0] should be negative (equity outlay).
    Returns IRR as decimal (0.15 = 15%).
    """
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


def underwrite(x: UnderwriteInputs) -> Dict[str, Any]:
    loan = x.purchase_price * x.ltv
    equity = x.purchase_price - loan
    ads = annual_debt_service(loan, x.interest_rate, x.amort_years)

    # Year-by-year NOI forecast
    noi_by_year = []
    for yr in range(1, x.hold_years + 1):
        noi = x.noi_year1 * ((1 + x.noi_growth) ** (yr - 1))
        noi_by_year.append(float(noi))

    # Annual cash flows before sale
    cash_flows = [-float(equity)]
    dscr_by_year = []
    for yr in range(1, x.hold_years + 1):
        noi = noi_by_year[yr - 1]
        cf = noi - ads
        cash_flows.append(float(cf))
        dscr = noi / ads if ads > 0 else math.inf
        dscr_by_year.append(float(dscr))

    # Sale at end of hold
    # Use hold-year NOI for exit valuation (simple convention)
    noi_exit = noi_by_year[-1]
    exit_value = noi_exit / x.exit_cap_rate if x.exit_cap_rate > 0 else math.inf
    selling_costs = exit_value * x.selling_cost_pct

    months_paid = x.hold_years * 12
    bal = remaining_balance(loan, x.interest_rate, x.amort_years, months_paid)

    net_sale_proceeds = exit_value - selling_costs - bal

    # Add sale proceeds to final year's cash flow
    cash_flows[-1] = cash_flows[-1] + float(net_sale_proceeds)

    irr_val = irr(cash_flows, guess=0.12)

    coc1 = (noi_by_year[0] - ads) / equity if equity > 0 else math.inf

    summary = UnderwriteSummary(
        loan_amount=float(loan),
        equity_required=float(equity),
        annual_debt_service=float(ads),
        exit_value=float(exit_value),
        selling_costs=float(selling_costs),
        remaining_loan_balance_at_sale=float(bal),
        net_sale_proceeds=float(net_sale_proceeds),
        irr_5yr=float(irr_val),
        cash_on_cash_year1=float(coc1),
    )

    schedule = []
    for i in range(x.hold_years):
        schedule.append(
            {
                "year": i + 1,
                "noi": noi_by_year[i],
                "debt_service": float(ads),
                "cash_flow": float(noi_by_year[i] - ads),
                "dscr": dscr_by_year[i],
            }
        )

    return {
        "summary": summary.__dict__,
        "cash_flows": cash_flows,   # year0..yearN
        "schedule": schedule,       # year1..yearN
    }