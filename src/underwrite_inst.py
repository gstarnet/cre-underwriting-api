# src/underwrite_inst.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math


@dataclass(frozen=True)
class InstUnderwriteInputs:
    """
    Institutional underwriting inputs (v2).

    Design goals:
    - Keep it asset-agnostic (works for MF/Ind/Retail/Office at a high level).
    - Model the underwriting mechanics that drive decisions:
      * line-item pro forma (rent/opex/taxes/insurance)
      * capex buckets (recurring reserve + replacements + value-add schedule)
      * debt mechanics (IO, optional rate shock, optional refi)
      * exit and IRR
    """

    # REQUIRED (no defaults)
    purchase_price: float
    ltv: float
    interest_rate: float
    amort_years: int

    gross_rent_year1: float
    opex_year1: float
    occupancy_year1: float
    gross_leasable_sqft: float

    # HOLD / EXIT
    hold_years: int = 5
    exit_cap_rate: float = 0.065
    selling_cost_pct: float = 0.02

    # DEBT (v1)
    interest_only_years: int = 0

    # PRO FORMA DRIVERS (v1)
    rent_growth: float = 0.03
    opex_inflation: float = 0.03
    occupancy_target: float = 0.95
    occupancy_reversion_years: int = 2

    # TAX/INSURANCE (v1)
    taxes_year1: float = 0.0
    insurance_year1: float = 0.0
    taxes_inflation: float = 0.03
    insurance_inflation: float = 0.04
    reassess_taxes: bool = False
    reassessed_tax_rate: float = 0.02
    reassess_year: int = 1

    # CAPEX (v2)
    capex_reserve_per_sqft: float = 0.0
    replacement_capex_per_sqft: float = 0.0
    value_add_capex: Optional[Dict[int, float]] = None  # year -> spend

    # OCCUPANCY SHOCK (v2)
    occupancy_shock_year: Optional[int] = None
    occupancy_shock_drop: float = 0.0
    occupancy_recovery_years: int = 2

    # RATE SHOCK (v2)
    rate_shock_year: Optional[int] = None
    rate_shock_bps: float = 0.0

    # REFI (v2)
    refi_year: Optional[int] = None
    refi_ltv: float = 0.0
    refi_rate: float = 0.0
    refi_amort_years: int = 30
    refi_cost_pct: float = 0.01


def annual_debt_service_amortizing(loan_amount: float, annual_rate: float, amort_years: int) -> float:
    """
    Compute annual debt service for a standard amortizing loan.

    Note:
    - This assumes monthly payments, returned as an annualized figure.
    """
    if loan_amount <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = amort_years * 12
    if r == 0:
        return float(loan_amount / amort_years)
    pmt_m = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return float(pmt_m * 12.0)


def remaining_balance_amortizing(loan_amount: float, annual_rate: float, amort_years: int, months_paid: int) -> float:
    """
    Remaining principal balance after 'months_paid' for an amortizing loan.
    """
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
    """
    Compute IRR via Newton-Raphson.

    cash_flows[0] should typically be negative (equity outflow).
    """
    def npv(rate: float) -> float:
        return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))

    def d_npv(rate: float) -> float:
        return sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cash_flows) if t > 0)

    r = float(guess)
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


def _base_occupancy_path(x: InstUnderwriteInputs, year: int) -> float:
    """
    Baseline occupancy path:
    - year 1 = occupancy_year1
    - then linearly reverts to occupancy_target over occupancy_reversion_years
    """
    if x.occupancy_reversion_years <= 1:
        return float(x.occupancy_target)
    if year <= 1:
        return float(x.occupancy_year1)
    if year >= x.occupancy_reversion_years:
        return float(x.occupancy_target)
    step = (x.occupancy_target - x.occupancy_year1) / float(x.occupancy_reversion_years - 1)
    return float(x.occupancy_year1 + step * (year - 1))


def _apply_occupancy_shock(x: InstUnderwriteInputs, occ: float, year: int) -> float:
    """
    Optional occupancy shock + recovery:
    - At occupancy_shock_year, occupancy drops by occupancy_shock_drop (absolute).
    - Then it recovers linearly back to occupancy_target over occupancy_recovery_years.
    """
    if not x.occupancy_shock_year or x.occupancy_shock_drop <= 0:
        return occ

    shock_year = int(x.occupancy_shock_year)
    if year < shock_year:
        return occ

    if year == shock_year:
        return max(0.0, min(1.0, occ - float(x.occupancy_shock_drop)))

    rec_years = max(1, int(x.occupancy_recovery_years))
    years_since = year - shock_year
    if years_since >= rec_years:
        return float(x.occupancy_target)

    occ_shocked = max(0.0, min(1.0, _base_occupancy_path(x, shock_year) - float(x.occupancy_shock_drop)))
    step = (x.occupancy_target - occ_shocked) / float(rec_years)
    return max(0.0, min(1.0, occ_shocked + step * years_since))


def _rate_for_year(x: InstUnderwriteInputs, year: int, base_rate: float) -> float:
    """
    Optional rate shock:
    - Starting at rate_shock_year, add rate_shock_bps to the interest rate.
    """
    if x.rate_shock_year and year >= int(x.rate_shock_year) and x.rate_shock_bps:
        return float(base_rate + (x.rate_shock_bps / 10000.0))
    return float(base_rate)


def underwrite_institutional(x: InstUnderwriteInputs) -> Dict[str, Any]:
    """
    Run the institutional pro forma and return:
    - schedule: per-year line items (NOI, debt service, DSCR, balances, etc.)
    - cash_flows: levered equity cash flows with sale proceeds included in final year
    - summary: headline metrics (IRR, DSCR stats) + institutional credit metrics (v2)

    Conventions:
    - NOI includes recurring reserves and replacement capex buckets (common for credit views).
    - value_add_capex is treated as below-the-line cash spend (reduces cash flow, not NOI).
    - Exit value = final-year NOI / exit_cap_rate.
    """
    # Initial debt / equity
    loan0 = float(x.purchase_price * x.ltv)
    equity0 = float(x.purchase_price - loan0)

    schedule: List[Dict[str, Any]] = []
    cash_flows: List[float] = [-equity0]

    # Track the active loan (supports refi)
    current_loan = float(loan0)
    current_rate = float(x.interest_rate)
    current_amort_years = int(x.amort_years)

    # IO applies only to the initial loan in this v2 implementation.
    io_years = max(0, min(int(x.interest_only_years), int(x.hold_years)))

    # For amort balance tracking on the active loan
    amort_months_paid = 0

    dscr_by_year: List[float] = []

    for year in range(1, int(x.hold_years) + 1):
        # Occupancy path with optional shock/recovery
        occ = _base_occupancy_path(x, year)
        occ = _apply_occupancy_shock(x, occ, year)
        occ = max(0.0, min(1.0, occ))

        # Revenue: grow gross rent, then apply occupancy
        gross_rent = float(x.gross_rent_year1 * ((1 + x.rent_growth) ** (year - 1)))
        effective_rent = float(gross_rent * occ)

        # Opex grows by inflation
        opex = float(x.opex_year1 * ((1 + x.opex_inflation) ** (year - 1)))

        # Taxes: optional reassessment
        if x.reassess_taxes and year >= int(x.reassess_year):
            taxes_base = float(x.purchase_price * x.reassessed_tax_rate)
        else:
            taxes_base = float(x.taxes_year1)
        taxes = float(taxes_base * ((1 + x.taxes_inflation) ** (year - 1)))

        insurance = float(x.insurance_year1 * ((1 + x.insurance_inflation) ** (year - 1)))

        # Capex buckets (treated as below-NOI reserves)
        capex_reserve = float(x.capex_reserve_per_sqft * x.gross_leasable_sqft)
        replacement_capex = float(x.replacement_capex_per_sqft * x.gross_leasable_sqft)

        # Value-add capex (below-the-line)
        value_add = 0.0
        if x.value_add_capex and year in x.value_add_capex:
            value_add = float(x.value_add_capex[year])

        # NOI (institutional credit view)
        noi = float(effective_rent - opex - taxes - insurance - capex_reserve - replacement_capex)

        # Debt service (apply rate shock if configured)
        yr_rate = _rate_for_year(x, year, current_rate)
        amort_ads = annual_debt_service_amortizing(current_loan, yr_rate, current_amort_years)
        io_ads = float(current_loan * yr_rate)

        debt_service = float(io_ads if year <= io_years else amort_ads)
        if year > io_years:
            amort_months_paid += 12

        # Equity cash flow
        cash_flow = float(noi - debt_service - value_add)

        # DSCR
        dscr = float((noi / debt_service) if debt_service > 0 else math.inf)
        dscr_by_year.append(dscr)

        # Loan balance end of year
        bal_end = (
            remaining_balance_amortizing(current_loan, yr_rate, current_amort_years, amort_months_paid)
            if amort_months_paid > 0
            else float(current_loan)
        )

        row: Dict[str, Any] = {
            "year": int(year),
            "occupancy": float(occ),
            "gross_rent": float(gross_rent),
            "effective_rent": float(effective_rent),
            "opex": float(opex),
            "taxes": float(taxes),
            "insurance": float(insurance),
            "capex_reserve": float(capex_reserve),
            "replacement_capex": float(replacement_capex),
            "value_add_capex": float(value_add),
            "noi": float(noi),
            "debt_service": float(debt_service),
            "cash_flow": float(cash_flow),
            "dscr": float(dscr),
            "debt_rate": float(yr_rate),
            "loan_balance_end_year": float(bal_end),
        }

        schedule.append(row)
        cash_flows.append(cash_flow)

        # Refi (end of refi_year)
        if x.refi_year and x.refi_ltv and year == int(x.refi_year) and float(x.refi_ltv) > 0:
            # Simple refi value proxy: this year's NOI at exit cap
            refi_value = float(noi / x.exit_cap_rate) if x.exit_cap_rate > 0 else math.inf
            new_loan = float(refi_value * x.refi_ltv)

            payoff = float(bal_end)
            refi_cost = float(new_loan * x.refi_cost_pct)
            cash_out = float(new_loan - payoff - refi_cost)

            # Add cash-out (or subtract cash-in) to this year's equity CF
            cash_flows[-1] = float(cash_flows[-1] + cash_out)

            # Record refi details for auditability
            schedule[-1]["refi_value"] = refi_value
            schedule[-1]["refi_new_loan"] = new_loan
            schedule[-1]["refi_payoff"] = payoff
            schedule[-1]["refi_cost"] = refi_cost
            schedule[-1]["refi_cash_out"] = cash_out

            # Reset debt state to new loan
            current_loan = new_loan
            current_rate = float(x.refi_rate) if float(x.refi_rate) > 0 else float(yr_rate)
            current_amort_years = int(x.refi_amort_years)
            amort_months_paid = 0

            # v2 convention: refi loan is fully amortizing (no IO reset)
            io_years = 0

    # Exit (end of hold)
    noi_exit = float(schedule[-1]["noi"])
    exit_value = float(noi_exit / x.exit_cap_rate) if x.exit_cap_rate > 0 else math.inf
    selling_costs = float(exit_value * x.selling_cost_pct)

    bal_sale = float(schedule[-1]["loan_balance_end_year"])
    net_sale_proceeds = float(exit_value - selling_costs - bal_sale)

    # Add sale proceeds to final year cash flow
    cash_flows[-1] = float(cash_flows[-1] + net_sale_proceeds)

    irr_val = float(irr(cash_flows, guess=0.12))

    # Institutional credit metrics (v2)
    loan_amount = float(loan0)
    debt_yield_year1 = float(schedule[0]["noi"] / loan_amount) if loan_amount > 0 else math.inf

    # Break-even occupancy (year 1):
    # occupancy needed so that effective_rent covers (opex+tax+ins+capex+replacement+debt_service).
    # This is a simplified but useful screening metric.
    year1_fixed = float(
        schedule[0]["opex"]
        + schedule[0]["taxes"]
        + schedule[0]["insurance"]
        + schedule[0]["capex_reserve"]
        + schedule[0]["replacement_capex"]
        + schedule[0]["debt_service"]
    )
    breakeven_occupancy_year1 = float(year1_fixed / schedule[0]["gross_rent"]) if schedule[0]["gross_rent"] > 0 else math.inf
    breakeven_occupancy_year1 = float(max(0.0, min(1.0, breakeven_occupancy_year1)))

    ltv_at_exit = float(bal_sale / exit_value) if exit_value > 0 and math.isfinite(exit_value) else math.inf

    summary = {
        # Core underwriting headline metrics
        "loan_amount": float(loan_amount),
        "equity_required": float(equity0),
        "exit_value": float(exit_value),
        "selling_costs": float(selling_costs),
        "remaining_loan_balance_at_sale": float(bal_sale),
        "net_sale_proceeds": float(net_sale_proceeds),
        "irr": float(irr_val),
        "cash_on_cash_year1": float(schedule[0]["cash_flow"] / equity0) if equity0 > 0 else math.inf,
        "dscr_by_year": [float(v) for v in dscr_by_year],
        "min_dscr": float(min(dscr_by_year)) if dscr_by_year else math.inf,

        # Institutional credit metrics (v2)
        "debt_yield_year1": float(debt_yield_year1),
        "breakeven_occupancy_year1": float(breakeven_occupancy_year1),
        "ltv_at_exit": float(ltv_at_exit),
    }

    return {"summary": summary, "schedule": schedule, "cash_flows": cash_flows}