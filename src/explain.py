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

from typing import Any, Dict, List
import json

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

from src.config import get_settings
from src.data_load import load_cre_csv, time_split, CRE_TARGET, TIME_COL
from src.dataset_versioning import write_dataset_metadata

settings = get_settings()
MODEL_PATH = settings.model_path
REPORTS_DIR = settings.reports_dir
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_JSON = settings.explainability_json_path


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_snapshot() -> Dict[str, Any]:
    if not settings.model_snapshot_path.exists():
        return {}
    try:
        return json.loads(settings.model_snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_ridge_coefficients(model: Any) -> List[Dict[str, float]]:
    if not hasattr(model, "named_steps"):
        return []
    fitted_model = model.named_steps.get("model")
    pre = model.named_steps.get("pre")
    if not isinstance(fitted_model, Ridge) or pre is None:
        return []
    if not hasattr(fitted_model, "coef_") or not hasattr(pre, "get_feature_names_out"):
        return []

    feature_names = list(pre.get_feature_names_out())
    coeffs = fitted_model.coef_
    if len(feature_names) != len(coeffs):
        return []

    pairs = [
        {"feature": str(feature_names[i]), "coefficient": float(coeffs[i])}
        for i in range(len(feature_names))
    ]
    pairs.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
    return pairs[:50]


def _save_feature_importance(
    fi: pd.DataFrame,
    *,
    model_snapshot: Dict[str, Any],
    ridge_coefficients: List[Dict[str, float]],
    dataset_hash: str,
) -> None:
    """
    Save explainability artifacts to:
      - reports/feature_importance.csv
      - reports/feature_importance.json
    """
    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    rows = [
        {
            "feature": r["feature"],
            "importance_mean": float(r["importance_mean"]),
            "importance_std": float(r["importance_std"]),
        }
        for r in fi.to_dict(orient="records")
    ]

    payload = {
        "target": CRE_TARGET,
        "method": "permutation_importance",
        "scoring": "neg_mean_absolute_error",
        "n_repeats": 10,
        "rows_test": int(len(fi)),
        "feature_importance": rows,
        "rows": rows,
        "dataset_hash": dataset_hash or model_snapshot.get("dataset_hash"),
        "model_version": model_snapshot.get("model_version"),
        "trained_at_utc": model_snapshot.get("trained_at_utc"),
    }
    if ridge_coefficients:
        payload["ridge_coefficients"] = ridge_coefficients

    FEATURE_IMPORTANCE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    _ensure_dirs()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model at {MODEL_PATH}. Run: python -m src.train")

    model = joblib.load(MODEL_PATH)

    # Load data and reuse the same leakage-safe time split as training
    df = load_cre_csv()
    dataset_meta = write_dataset_metadata(df, data_path=settings.data_path)
    model_snapshot = _load_model_snapshot()
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

    ridge_coefficients = _extract_ridge_coefficients(model)
    _save_feature_importance(
        fi,
        model_snapshot=model_snapshot,
        ridge_coefficients=ridge_coefficients,
        dataset_hash=str(dataset_meta.get("dataset_hash", "")),
    )

    print(f"Saved: {FEATURE_IMPORTANCE_CSV}")
    print(f"Saved: {FEATURE_IMPORTANCE_JSON}")
    print("Top 10:")
    print(fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
