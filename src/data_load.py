# src/data_load.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

CRE_TARGET = "noi_next12"
TIME_COL = "asof_date"  # or close_date

def load_cre_csv(path: str = "data/raw/cre_deals.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        raise ValueError(f"{TIME_COL} has invalid dates.")
    if CRE_TARGET not in df.columns:
        raise ValueError(f"Missing target column: {CRE_TARGET}")
    return df

def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Dataset:
    # Sort by time and split last N% as holdout (prevents leakage)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    y_train = train_df[CRE_TARGET].astype(float)
    y_test  = test_df[CRE_TARGET].astype(float)

    drop_cols = [CRE_TARGET]
    X_train = train_df.drop(columns=drop_cols)
    X_test  = test_df.drop(columns=drop_cols)

    return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)