"""
Train a baseline tabular model for CRE NOI forecasting.

Outputs:
- models/model.joblib              (trained sklearn Pipeline)
- reports/metrics.csv              (model metrics on holdout split)
- reports/feature_importance.csv   (permutation importance; model-agnostic)
- reports/feature_importance.json  (same content for API consumption)

Notes:
- This project uses a time-based split (see src/data_load.time_split()) to avoid leakage.
- Explainability here is intentionally simple and production-friendly:
  permutation importance on the test fold (model-agnostic, works for any sklearn estimator).
"""

from __future__ import annotations

# Standard library
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple

# Third-party
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Local
from src.config import get_settings
from src.data_load import load_cre_csv, time_split, CRE_TARGET, TIME_COL
from src.dataset_versioning import write_dataset_metadata
from src.unstructured import UNSTRUCTURED_TEXT_COL, ensure_unstructured_text_column


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
settings = get_settings()
MODELS_DIR = settings.model_path.parent
REPORTS_DIR = settings.reports_dir

MODEL_PATH = settings.model_path
MODEL_SNAPSHOT_JSON = settings.model_snapshot_path
METRICS_CSV = REPORTS_DIR / "metrics.csv"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_JSON = REPORTS_DIR / "feature_importance.json"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _infer_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Infer categorical vs numeric feature columns from the raw dataframe.

    We explicitly exclude:
    - target column (CRE_TARGET)
    - time column (TIME_COL)
    """
    feature_cols = [c for c in df.columns if c not in {CRE_TARGET, TIME_COL}]
    text_cols = [c for c in feature_cols if c == UNSTRUCTURED_TEXT_COL]
    non_text_cols = [c for c in feature_cols if c not in set(text_cols)]
    cat_cols = [c for c in non_text_cols if df[c].dtype == "object"]
    num_cols = [c for c in non_text_cols if c not in cat_cols]
    return cat_cols, num_cols, text_cols


class _TextCleaner(BaseEstimator, TransformerMixin):
    """Normalize optional text column into a 1D string array for TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(X)
        return s.fillna("").astype(str).map(lambda v: " ".join(v.split())).values


def _build_preprocessor(cat_cols: List[str], num_cols: List[str], text_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessor:
    - categorical: impute missing, one-hot encode (handle_unknown to avoid crashes)
    - numeric: impute missing, standardize

    NOTE:
    - We keep OneHotEncoder sparse output (default) since it's efficient for many categories.
    - Tree model (HGB) expects dense after preprocessing; we handle that via a small adapter in pipeline.
    """
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # with_mean=False supports sparse safely
        ]
    )

    transformers = [
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ]
    if text_cols:
        text_pipe = Pipeline(
            steps=[
                ("clean", _TextCleaner()),
                ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)),
            ]
        )
        transformers.append(("text", text_pipe, text_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _to_dense_if_needed(X):
    """
    HistGradientBoostingRegressor requires dense input.
    If preprocessing yields a sparse matrix, convert to dense.

    This is intentionally localized to keep the rest of the pipeline standard.
    """
    # scipy sparse matrices have .toarray()
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


class _DenseAdapter:
    """
    A tiny transformer used inside sklearn Pipeline to convert sparse -> dense
    only for models that require dense matrices.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _to_dense_if_needed(X)


def _eval_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # sklearn version-safe (no squared=)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def _save_metrics(rows: List[Dict[str, Any]]) -> None:
    """
    Save metrics table to reports/metrics.csv.
    """
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_CSV, index=False)


def _save_model_snapshot(
    *,
    best_name: str,
    metrics_rows: List[Dict[str, Any]],
    dataset_meta: Dict[str, Any],
) -> None:
    payload = {
        "model_version": "v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "best_model_name": best_name,
        "dataset_hash": dataset_meta.get("dataset_hash"),
        "dataset_metadata_path": str(settings.dataset_metadata_path),
        "holdout_metrics": metrics_rows,
    }
    MODEL_SNAPSHOT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_permutation_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Compute and store permutation importance.

    IMPORTANT FIX:
    - permutation_importance(model, X_test, ...) returns importances per *input column* of X_test
      (because it permutes columns of X_test before passing into the pipeline).
    - Therefore, feature names MUST match X_test.columns, NOT the expanded OHE feature names.
      Using get_feature_names_out() will often cause length mismatches.

    Outputs:
    - reports/feature_importance.csv
    - reports/feature_importance.json
    """
    if not isinstance(X_test, pd.DataFrame):
        # safety: ensure we have column names
        X_test = pd.DataFrame(X_test)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    # Use *original* input feature names (matches permutation_importance length)
    feature_names = list(X_test.columns)

    if len(feature_names) != len(importances_mean):
        raise RuntimeError(
            f"Permutation importance length mismatch: "
            f"features={len(feature_names)} importances={len(importances_mean)}. "
            f"Ensure X_test is a DataFrame with the same columns used for training."
        )

    fi = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances_mean,
            "importance_std": importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    # JSON artifact for API usage (top-to-bottom order)
    payload = {
        "scoring": "neg_mean_absolute_error",
        "n_repeats": 10,
        "rows": [
            {
                "feature": r["feature"],
                "importance_mean": float(r["importance_mean"]),
                "importance_std": float(r["importance_std"]),
            }
            for r in fi.to_dict(orient="records")
        ],
    }
    FEATURE_IMPORTANCE_JSON.write_text(pd.Series(payload).to_json(), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main training flow
# -----------------------------------------------------------------------------
def main() -> None:
    _ensure_dirs()

    df = load_cre_csv()
    dataset_meta = write_dataset_metadata(df, data_path=settings.data_path)
    df = ensure_unstructured_text_column(df, strict=False)
    for c in ("deal_notes", "document_paths"):
        if c in df.columns and c != UNSTRUCTURED_TEXT_COL:
            df = df.drop(columns=[c])
    ds = time_split(df)

    print(f"Target: {CRE_TARGET}")

    cat_cols, num_cols, text_cols = _infer_feature_cols(df)
    if text_cols:
        text_series = df[text_cols[0]].fillna("").astype(str).map(lambda v: v.strip())
        if not (text_series != "").any():
            text_cols = []

    pre = _build_preprocessor(cat_cols, num_cols, text_cols)

    # Candidate models:
    # - ridge: stable baseline, handles sparse well
    # - hgb: tree model; requires dense input -> DenseAdapter
    ridge_pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )

    hgb_pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("dense", _DenseAdapter()),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    )

    candidates = [
        ("ridge", ridge_pipe),
        ("hgb", hgb_pipe),
    ]

    metrics_rows: List[Dict[str, Any]] = []
    best_name = None
    best_pipe = None
    best_rmse = float("inf")

    for name, pipe in candidates:
        pipe.fit(ds.X_train, ds.y_train)
        pred = pipe.predict(ds.X_test)
        m = _eval_regression(ds.y_test, pred)
        m["model"] = name
        metrics_rows.append(m)

        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_name = name
            best_pipe = pipe

    _save_metrics(metrics_rows)
    _save_model_snapshot(
        best_name=str(best_name),
        metrics_rows=metrics_rows,
        dataset_meta=dataset_meta,
    )
    print(f"Saved metrics: {METRICS_CSV}")

    # Save best model
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH} best= {best_name}")

    # Explainability artifacts on the chosen best model (test fold)
    _save_permutation_importance(best_pipe, ds.X_test, ds.y_test)
    print(f"Saved feature importance: {FEATURE_IMPORTANCE_CSV} and {FEATURE_IMPORTANCE_JSON}")
    print(f"Saved model snapshot: {MODEL_SNAPSHOT_JSON}")


if __name__ == "__main__":
    main()
