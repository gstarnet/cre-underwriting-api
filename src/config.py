from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    log_level: str
    log_json: bool
    auth_mode: str
    auth_token: str
    data_path: Path
    dataset_metadata_path: Path
    model_path: Path
    model_snapshot_path: Path
    reports_dir: Path
    explainability_json_path: Path
    ts_cv_splits: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))

    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_json=_env_bool("LOG_JSON", True),
        auth_mode=os.getenv("AUTH_MODE", "none").strip().lower(),
        auth_token=os.getenv("AUTH_TOKEN", ""),
        data_path=Path(os.getenv("DATA_PATH", "data/raw/cre_deals.csv")),
        dataset_metadata_path=Path(
            os.getenv("DATASET_METADATA_PATH", "data/metadata/dataset_metadata.json")
        ),
        model_path=Path(os.getenv("MODEL_PATH", "models/model.joblib")),
        model_snapshot_path=Path(
            os.getenv("MODEL_SNAPSHOT_PATH", "reports/model_snapshot.json")
        ),
        reports_dir=reports_dir,
        explainability_json_path=Path(
            os.getenv(
                "EXPLAINABILITY_JSON_PATH",
                str(reports_dir / "feature_importance.json"),
            )
        ),
        ts_cv_splits=max(2, int(os.getenv("TS_CV_SPLITS", "8"))),
    )
