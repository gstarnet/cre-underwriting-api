# src/data_load.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

CRE_TARGET = "noi_next12"
TIME_COL = "asof_date"


@dataclass(frozen=True)
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_cre_csv(path: str = "data/raw/cre_deals.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required column: {TIME_COL}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        bad = df[df[TIME_COL].isna()].head(5)
        raise ValueError(f"{TIME_COL} has invalid dates. Example bad rows:\n{bad}")

    if CRE_TARGET not in df.columns:
        raise ValueError(f"Missing target column: {CRE_TARGET}")

    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Dataset:
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("Need at least 3 rows to split into train/test.")

    split_idx = int(len(df) * (1 - test_frac))
    split_idx = max(1, min(split_idx, len(df) - 1))  # ensure at least 1 row in each side

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    y_train = train_df[CRE_TARGET].astype(float)
    y_test = test_df[CRE_TARGET].astype(float)

    X_train = train_df.drop(columns=[CRE_TARGET])
    X_test = test_df.drop(columns=[CRE_TARGET])

    return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)