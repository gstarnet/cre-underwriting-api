# src/validate_ts.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_load import load_cre_csv, time_split, CRE_TARGET, TIME_COL

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    drop_cols = [c for c in ["deal_id", TIME_COL] if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]
    categorical_cols = [c for c in categorical_cols if c not in drop_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def eval_regression(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)  # old sklearn compatible
    rmse = float(mse ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))

    r2 = None
    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, y_pred))

    return {"mae": mae, "rmse": rmse, "r2": r2}


def walk_forward_cv(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
) -> pd.DataFrame:
    n = len(X)
    n_splits = max(2, min(n_splits, n - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pre = build_preprocessor(X)
    model = HistGradientBoostingRegressor(random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    rows = []
    split_idx = 0

    for train_idx, val_idx in tscv.split(X):
        split_idx += 1

        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)

        # guard (should be fine for HGB)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Non-finite predictions in split {split_idx}")

        m = eval_regression(y_va, pred)

        rows.append(
            {
                "model": "hgb",
                "split": split_idx,
                "train_n": len(train_idx),
                "val_n": len(val_idx),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
            }
        )

    return pd.DataFrame(rows)


def main():
    df = load_cre_csv("data/raw/cre_deals.csv")
    ds = time_split(df, test_frac=0.2)

    # CV runs on TRAIN ONLY, ordered by time
    X_train = ds.X_train.sort_values(TIME_COL).reset_index(drop=True)
    y_train = ds.y_train.reset_index(drop=True)

    cv_metrics = walk_forward_cv(X_train, y_train, n_splits=5)

    summary = (
        cv_metrics.groupby("model")[["mae", "rmse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std"]

    cv_path = REPORTS_DIR / "ts_cv_metrics.csv"
    summary_path = REPORTS_DIR / "ts_cv_summary.csv"
    cv_metrics.to_csv(cv_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Saved:", cv_path)
    print("Saved:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()