"""
Generate explainability artifacts for the trained CRE model.

Outputs:
- reports/feature_importance.csv   (permutation importance; model-agnostic)
- reports/feature_importance.json  (same content for API consumption)

Notes:
- Uses the same time-based split as training (see src/data_load.time_split()).
- Permutation importance is computed over the *raw input columns* (X_test.columns),
  because sklearn.inspection.permutation_importance permutes columns of the input
  passed to the Pipeline (before preprocessing expands OHE features).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data_load import load_cre_csv, time_split, CRE_TARGET, TIME_COL

MODEL_PATH = Path("models/model.joblib")
REPORTS_DIR = Path("reports")
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_JSON = REPORTS_DIR / "feature_importance.json"


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_feature_importance(fi: pd.DataFrame) -> None:
    """
    Save explainability artifacts to:
      - reports/feature_importance.csv
      - reports/feature_importance.json
    """
    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    # JSON payload is intentionally simple and stable for API consumption.
    # Keep the keys aligned with scripts/test_explainability.sh expectations.
    rows = [
        {
            "feature": r["feature"],
            "importance_mean": float(r["importance_mean"]),
            "importance_std": float(r["importance_std"]),
        }
        for r in fi.to_dict(orient="records")
    ]

    payload = {
        # Required by test + useful for consumers
        "target": CRE_TARGET,
        "method": "permutation_importance",
        "scoring": "neg_mean_absolute_error",
        "n_repeats": 10,
        "rows_test": int(len(fi)),
        "feature_importance": rows,
        # Backward-compat alias (older scripts may look for "rows")
        "rows": rows,
    }

    # Write JSON deterministically
    import json

    FEATURE_IMPORTANCE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    _ensure_dirs()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model at {MODEL_PATH}. Run: python -m src.train")

    model = joblib.load(MODEL_PATH)

    # Load data and reuse the same leakage-safe time split as training
    df = load_cre_csv()
    ds = time_split(df)

    # Ensure X_test is a DataFrame with columns, so feature names match permutation output length
    X_test = ds.X_test
    y_test = ds.y_test

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Compute permutation importance against the Pipeline directly
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

    feature_names = list(X_test.columns)

    # Critical guard: prevent silent mismatches
    if len(feature_names) != len(importances_mean):
        raise RuntimeError(
            f"Permutation importance length mismatch: "
            f"features={len(feature_names)} importances={len(importances_mean)}. "
            f"This usually happens if you try to use expanded (OHE) feature names. "
            f"Use X_test.columns (raw inputs) instead."
        )

    fi = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances_mean,
            "importance_std": importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    _save_feature_importance(fi)

    print(f"Saved: {FEATURE_IMPORTANCE_CSV}")
    print(f"Saved: {FEATURE_IMPORTANCE_JSON}")
    print("Top 10:")
    print(fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()