# src/underwrite_inst.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import math


@dataclass(frozen=True)
class InstUnderwriteInputs:
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
    capex_reserve_per_sqft: float = 0.0          # recurring reserve
    replacement_capex_per_sqft: float = 0.0      # additional replacements bucket
    value_add_capex: Optional[Dict[int, float]] = None  # year -> spend

    # OCCUPANCY SHOCK (v2)
    occupancy_shock_year: Optional[int] = None   # e.g., 2
    occupancy_shock_drop: float = 0.0            # e.g., 0.10 means -10% absolute occupancy
    occupancy_recovery_years: int = 2            # years to recover back to target

    # RATE SHOCK (v2)
    rate_shock_year: Optional[int] = None        # e.g., 3
    rate_shock_bps: float = 0.0                  # e.g., 150 = +1.50%

    # REFI (v2)
    refi_year: Optional[int] = None              # e.g., 3 (end of year 3)
    refi_ltv: float = 0.0                        # if 0, disable
    refi_rate: float = 0.0
    refi_amort_years: int = 30
    refi_cost_pct: float = 0.01                  # % of new loan amount as cost


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


def _base_occupancy_path(x: InstUnderwriteInputs, year: int) -> float:
    # baseline linear move from year1 to target
    if x.occupancy_reversion_years <= 1:
        return float(x.occupancy_target)
    if year <= 1:
        return float(x.occupancy_year1)
    if year >= x.occupancy_reversion_years:
        return float(x.occupancy_target)
    step = (x.occupancy_target - x.occupancy_year1) / float(x.occupancy_reversion_years - 1)
    return float(x.occupancy_year1 + step * (year - 1))


def _apply_occupancy_shock(x: InstUnderwriteInputs, occ: float, year: int) -> float:
    if not x.occupancy_shock_year or x.occupancy_shock_drop <= 0:
        return occ

    shock_year = int(x.occupancy_shock_year)
    if year < shock_year:
        return occ

    # Shock hits at shock_year
    if year == shock_year:
        return max(0.0, min(1.0, occ - float(x.occupancy_shock_drop)))

    # Recover linearly to target over recovery_years
    rec_years = max(1, int(x.occupancy_recovery_years))
    years_since = year - shock_year
    if years_since >= rec_years:
        return float(x.occupancy_target)

    # interpolation between (occ at shock) and target
    occ_shocked = max(0.0, min(1.0, _base_occupancy_path(x, shock_year) - float(x.occupancy_shock_drop)))
    step = (x.occupancy_target - occ_shocked) / float(rec_years)
    return max(0.0, min(1.0, occ_shocked + step * years_since))


def _rate_for_year(x: InstUnderwriteInputs, year: int, base_rate: float) -> float:
    if x.rate_shock_year and year >= int(x.rate_shock_year) and x.rate_shock_bps:
        return float(base_rate + (x.rate_shock_bps / 10000.0))
    return float(base_rate)


def underwrite_institutional(x: InstUnderwriteInputs) -> Dict[str, Any]:
    # initial debt
    loan = x.purchase_price * x.ltv
    equity = x.purchase_price - loan

    schedule: List[Dict[str, Any]] = []
    cash_flows: List[float] = [-float(equity)]

    # Track debt state (for refi)
    current_loan = float(loan)
    current_rate = float(x.interest_rate)
    current_amort_years = int(x.amort_years)
    io_years = max(0, min(int(x.interest_only_years), int(x.hold_years)))

    amort_ads = annual_debt_service_amortizing(current_loan, current_rate, current_amort_years)
    io_ads = float(current_loan * current_rate)

    # For amort balance tracking: count amort months actually paid on current loan
    amort_months_paid = 0

    for year in range(1, x.hold_years + 1):
        # occupancy path with optional shock/recovery
        occ = _base_occupancy_path(x, year)
        occ = _apply_occupancy_shock(x, occ, year)
        occ = max(0.0, min(1.0, occ))

        # revenue
        gross_rent = x.gross_rent_year1 * ((1 + x.rent_growth) ** (year - 1))
        effective_rent = gross_rent * occ

        # expenses
        opex = x.opex_year1 * ((1 + x.opex_inflation) ** (year - 1))

        # taxes reassessment option
        if x.reassess_taxes and year >= x.reassess_year:
            taxes_base = x.purchase_price * x.reassessed_tax_rate
        else:
            taxes_base = x.taxes_year1
        taxes = taxes_base * ((1 + x.taxes_inflation) ** (year - 1))

        insurance = x.insurance_year1 * ((1 + x.insurance_inflation) ** (year - 1))

        # capex buckets
        capex_reserve = x.capex_reserve_per_sqft * x.gross_leasable_sqft
        replacement_capex = x.replacement_capex_per_sqft * x.gross_leasable_sqft
        value_add = 0.0
        if x.value_add_capex and year in x.value_add_capex:
            value_add = float(x.value_add_capex[year])

        noi = effective_rent - opex - taxes - insurance - capex_reserve - replacement_capex

        # debt service: apply rate shock (simple)
        yr_rate = _rate_for_year(x, year, current_rate)
        # recompute ADS if rate changes (approx; assumes payment reset annually)
        amort_ads = annual_debt_service_amortizing(current_loan, yr_rate, current_amort_years)
        io_ads = float(current_loan * yr_rate)

        debt_service = io_ads if year <= io_years else amort_ads
        if year > io_years:
            amort_months_paid += 12

        cf = noi - debt_service - value_add
        dscr = (noi / debt_service) if debt_service > 0 else math.inf

        schedule.append(
            {
                "year": year,
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
                "cash_flow": float(cf),
                "dscr": float(dscr),
                "debt_rate": float(yr_rate),
                "loan_balance_end_year": None,  # filled below
            }
        )

        cash_flows.append(float(cf))

        # compute current loan balance end of year (only amort months paid count)
        bal_end = (
            remaining_balance_amortizing(current_loan, yr_rate, current_amort_years, amort_months_paid)
            if amort_months_paid > 0
            else float(current_loan)
        )
        schedule[-1]["loan_balance_end_year"] = float(bal_end)

        # Refi at end of refi_year
        if x.refi_year and x.refi_ltv and year == int(x.refi_year):
            # Refi value using that year's NOI at exit cap (simple convention)
            refi_value = float(noi / x.exit_cap_rate) if x.exit_cap_rate > 0 else math.inf
            new_loan = float(refi_value * x.refi_ltv)

            # payoff old loan + refi costs
            payoff = float(bal_end)
            refi_cost = float(new_loan * x.refi_cost_pct)
            cash_out = new_loan - payoff - refi_cost  # can be negative if under-water

            # add cash-out (or subtract if cash-in required) to that year's cash flow
            cash_flows[-1] = cash_flows[-1] + cash_out
            schedule[-1]["refi_value"] = refi_value
            schedule[-1]["refi_new_loan"] = new_loan
            schedule[-1]["refi_payoff"] = payoff
            schedule[-1]["refi_cost"] = refi_cost
            schedule[-1]["refi_cash_out"] = cash_out

            # reset debt state
            current_loan = new_loan
            current_rate = float(x.refi_rate) if x.refi_rate > 0 else yr_rate
            current_amort_years = int(x.refi_amort_years)
            amort_months_paid = 0  # reset on new loan
            io_years = 0  # assume refi loan is amortizing; keep it simple in v2

    # Exit at end of hold
    noi_exit = schedule[-1]["noi"]
    exit_value = float(noi_exit / x.exit_cap_rate) if x.exit_cap_rate > 0 else math.inf
    selling_costs = float(exit_value * x.selling_cost_pct)

    # remaining balance at sale is whatever last schedule computed
    bal_sale = float(schedule[-1]["loan_balance_end_year"])
    net_sale_proceeds = float(exit_value - selling_costs - bal_sale)
    cash_flows[-1] = cash_flows[-1] + net_sale_proceeds

    irr_val = irr(cash_flows, guess=0.12)

    summary = {
        "loan_amount": float(loan),
        "equity_required": float(equity),
        "exit_value": float(exit_value),
        "selling_costs": float(selling_costs),
        "remaining_loan_balance_at_sale": float(bal_sale),
        "net_sale_proceeds": float(net_sale_proceeds),
        "irr": float(irr_val),
        "cash_on_cash_year1": float(schedule[0]["cash_flow"] / equity) if equity > 0 else math.inf,
        "min_dscr": float(min(s["dscr"] for s in schedule)),
    }

    return {"summary": summary, "schedule": schedule, "cash_flows": cash_flows}