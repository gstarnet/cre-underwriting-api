# src/validate_ts.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import get_settings
from src.data_load import load_cre_csv, time_split, TIME_COL
from src.dataset_versioning import write_dataset_metadata

settings = get_settings()
REPORTS_DIR = settings.reports_dir
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
        n_splits: int = 8,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    n = len(X)
    n_splits = max(2, min(n_splits, n - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pre = build_preprocessor(X)
    model = HistGradientBoostingRegressor(random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    rows = []
    fold_artifacts: List[Dict[str, Any]] = []
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
        fold_artifacts.append(
            {
                "split": split_idx,
                "train_n": len(train_idx),
                "val_n": len(val_idx),
                "train_start": str(X_tr[TIME_COL].min()) if TIME_COL in X_tr else None,
                "train_end": str(X_tr[TIME_COL].max()) if TIME_COL in X_tr else None,
                "val_start": str(X_va[TIME_COL].min()) if TIME_COL in X_va else None,
                "val_end": str(X_va[TIME_COL].max()) if TIME_COL in X_va else None,
                "metrics": m,
            }
        )

    return pd.DataFrame(rows), fold_artifacts


def main():
    df = load_cre_csv(str(settings.data_path))
    dataset_meta = write_dataset_metadata(df, data_path=settings.data_path)
    ds = time_split(df, test_frac=0.2)

    # CV runs on TRAIN ONLY, ordered by time
    X_train = ds.X_train.sort_values(TIME_COL).reset_index(drop=True)
    y_train = ds.y_train.reset_index(drop=True)

    cv_metrics, fold_artifacts = walk_forward_cv(
        X_train, y_train, n_splits=settings.ts_cv_splits
    )

    summary = (
        cv_metrics.groupby("model")[["mae", "rmse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std"]

    cv_path = REPORTS_DIR / "ts_cv_metrics.csv"
    summary_path = REPORTS_DIR / "ts_cv_summary.csv"
    folds_path = REPORTS_DIR / "ts_cv_folds.json"
    cv_metrics.to_csv(cv_path, index=False)
    summary.to_csv(summary_path, index=False)
    fold_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_splits": int(settings.ts_cv_splits),
        "model": "hgb",
        "model_params": HistGradientBoostingRegressor(random_state=42).get_params(),
        "dataset_hash": dataset_meta.get("dataset_hash"),
        "folds": fold_artifacts,
    }
    folds_path.write_text(json.dumps(fold_payload, indent=2), encoding="utf-8")

    print("Saved:", cv_path)
    print("Saved:", summary_path)
    print("Saved:", folds_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
