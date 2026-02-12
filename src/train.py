# src/train.py
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_load import load_cre_csv, time_split, CRE_TARGET, TIME_COL

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Drop non-signal fields
    drop_cols = [c for c in ["deal_id", TIME_COL] if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]
    categorical_cols = [c for c in categorical_cols if c not in drop_cols]

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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

    # R² not defined for < 2 samples
    r2 = None
    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, y_pred))

    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    df = load_cre_csv("data/raw/cre_deals.csv")
    ds = time_split(df, test_frac=0.2)

    pre = build_preprocessor(ds.X_train)

    candidates = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
    }

    results = []
    best_name, best_rmse, best_pipe = None, None, None

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(ds.X_train, ds.y_train)
        pred = pipe.predict(ds.X_test)

        metrics = eval_regression(ds.y_test, pred)
        metrics["model"] = name
        results.append(metrics)

        if best_rmse is None or metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name
            best_pipe = pipe

    metrics_df = pd.DataFrame(results).sort_values("rmse")
    metrics_path = REPORTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(best_pipe, model_path)

    print(f"Target: {CRE_TARGET}")
    print("Saved metrics:", metrics_path)
    print(metrics_df.to_string(index=False))
    print("Saved model:", model_path, "best=", best_name)


if __name__ == "__main__":
    main()