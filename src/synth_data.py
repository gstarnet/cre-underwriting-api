# src/synth_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import date, timedelta
import random
import math
import pandas as pd


@dataclass(frozen=True)
class Market:
    city: str
    state: str
    zip: str
    market_rent_psf_mo: float  # baseline $/sf/month (or MF proxy)
    opex_ratio: float          # baseline opex % of gross rent
    cap_base: float            # baseline market cap rate


MARKETS = [
    Market("Orlando", "FL", "32801", 2.10, 0.33, 0.062),
    Market("Tampa", "FL", "33602", 2.05, 0.32, 0.061),
    Market("Miami", "FL", "33101", 2.60, 0.36, 0.055),
    Market("Jacksonville", "FL", "32202", 1.75, 0.31, 0.067),
    Market("St Petersburg", "FL", "33701", 2.15, 0.34, 0.060),
    Market("Austin", "TX", "78701", 2.40, 0.34, 0.058),
    Market("Atlanta", "GA", "30303", 2.00, 0.33, 0.063),
    Market("Nashville", "TN", "37219", 2.10, 0.33, 0.062),
]

PROPERTY_TYPES = ["Multifamily", "Industrial", "Retail", "Office", "SelfStorage"]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rand_date(start: date, end: date) -> date:
    days = (end - start).days
    return start + timedelta(days=random.randint(0, days))


def amort_annual_debt_service(loan_amount: float, annual_rate: float, amort_years: int) -> float:
    if loan_amount <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = amort_years * 12
    if r == 0:
        return loan_amount / amort_years
    pmt_m = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return float(pmt_m * 12.0)


def make_row(i: int, start: date, end: date) -> dict:
    market = random.choice(MARKETS)
    ptype = random.choice(PROPERTY_TYPES)

    asof = rand_date(start, end)

    year_built = random.randint(1965, 2022)

    # Size / units
    if ptype == "Multifamily":
        units = random.randint(40, 320)
        sqft = units * random.uniform(650, 1050)  # proxy
        rent_psf_mo = market.market_rent_psf_mo * random.uniform(0.85, 1.20)
    elif ptype == "Industrial":
        units = None
        sqft = random.uniform(50_000, 400_000)
        rent_psf_mo = market.market_rent_psf_mo * random.uniform(0.60, 1.00)
    elif ptype == "Retail":
        units = None
        sqft = random.uniform(20_000, 180_000)
        rent_psf_mo = market.market_rent_psf_mo * random.uniform(0.90, 1.30)
    elif ptype == "Office":
        units = None
        sqft = random.uniform(40_000, 300_000)
        rent_psf_mo = market.market_rent_psf_mo * random.uniform(0.80, 1.20)
    else:  # SelfStorage
        units = None
        sqft = random.uniform(25_000, 140_000)
        rent_psf_mo = market.market_rent_psf_mo * random.uniform(0.70, 1.05)

    # Occupancy influenced by type and building age
    age = asof.year - year_built
    occ_base = {
        "Multifamily": 0.94,
        "Industrial": 0.96,
        "Retail": 0.90,
        "Office": 0.86,
        "SelfStorage": 0.92,
    }[ptype]
    occ_penalty_age = clamp((age - 25) * 0.001, 0.0, 0.06)  # older tends to slightly lower occ
    occupancy = clamp(random.gauss(occ_base - occ_penalty_age, 0.03), 0.65, 0.99)

    gross_rent_t12 = sqft * rent_psf_mo * 12.0 * occupancy

    # Opex ratio varies by type and market
    opex_type_adj = {
        "Multifamily": 0.08,
        "Industrial": -0.05,
        "Retail": 0.02,
        "Office": 0.10,
        "SelfStorage": -0.02,
    }[ptype]
    opex_ratio = clamp(random.gauss(market.opex_ratio + opex_type_adj, 0.04), 0.18, 0.55)
    opex_t12 = gross_rent_t12 * opex_ratio

    noi_t12 = gross_rent_t12 - opex_t12

    # Purchase price using cap rate logic + noise
    cap = clamp(random.gauss(market.cap_base + (0.01 if ptype == "Office" else 0.0), 0.008), 0.04, 0.10)
    price_from_noi = noi_t12 / cap
    purchase_price = price_from_noi * random.uniform(0.92, 1.10)  # deal premium/discount

    # Financing terms
    ltv = clamp(random.gauss(0.67, 0.06), 0.50, 0.80)
    interest_rate = clamp(random.gauss(0.062, 0.007), 0.045, 0.090)
    amort_years = random.choice([20, 25, 30])

    # Next-12 NOI: rent growth + occupancy drift + opex drift + noise
    # (kept mild so model can learn relationships)
    rent_growth = random.gauss(0.03, 0.03)  # -? to +?
    rent_growth = clamp(rent_growth, -0.06, 0.12)

    occ_drift = random.gauss(0.0, 0.02)
    next_occ = clamp(occupancy + occ_drift, 0.60, 0.99)

    opex_infl = clamp(random.gauss(0.03, 0.02), -0.03, 0.10)

    gross_rent_next12 = sqft * rent_psf_mo * (1.0 + rent_growth) * 12.0 * next_occ
    opex_next12 = gross_rent_next12 * clamp(opex_ratio * (1.0 + opex_infl), 0.15, 0.60)

    noi_next12 = gross_rent_next12 - opex_next12

    # Add a little idiosyncratic NOI noise
    noi_next12 *= random.uniform(0.97, 1.03)

    return {
        "deal_id": f"D{i:04d}",
        "asof_date": asof.isoformat(),
        "property_type": ptype,
        "city": market.city,
        "state": market.state,
        "zip": market.zip,
        "year_built": int(year_built),
        "gross_leasable_sqft": float(round(sqft, 2)),
        "units": (None if units is None else int(units)),
        "purchase_price": float(round(purchase_price, 2)),
        "noi_t12": float(round(noi_t12, 2)),
        "occupancy_t12": float(round(occupancy, 4)),
        "opex_t12": float(round(opex_t12, 2)),
        "gross_rent_t12": float(round(gross_rent_t12, 2)),
        "ltv": float(round(ltv, 4)),
        "interest_rate": float(round(interest_rate, 4)),
        "amort_years": int(amort_years),
        "noi_next12": float(round(noi_next12, 2)),
    }


def generate(n: int, out_path: str) -> None:
    random.seed(42)

    start = date(2021, 1, 1)
    end = date(2025, 12, 31)

    rows = [make_row(i + 1, start, end) for i in range(n)]
    df = pd.DataFrame(rows)

    # Ensure chronological sorting (helps your time_split logic)
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    df = df.sort_values("asof_date").reset_index(drop=True)
    df["asof_date"] = df["asof_date"].dt.date.astype(str)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/raw/cre_deals.csv")
    args = parser.parse_args()

    generate(args.n, args.out)