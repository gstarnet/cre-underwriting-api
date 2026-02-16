from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict
import json

import pandas as pd

from src.config import get_settings


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_metadata(df: pd.DataFrame, data_path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "data_path": str(data_path),
        "dataset_hash": file_sha256(data_path),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if "asof_date" in df.columns and len(df) > 0:
        ts = pd.to_datetime(df["asof_date"], errors="coerce")
        payload["asof_date_min"] = (
            ts.min().date().isoformat() if not ts.isna().all() else None
        )
        payload["asof_date_max"] = (
            ts.max().date().isoformat() if not ts.isna().all() else None
        )

    return payload


def write_dataset_metadata(
    df: pd.DataFrame,
    *,
    data_path: Path,
    metadata_path: Path | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    metadata_path = metadata_path or settings.dataset_metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_dataset_metadata(df, data_path=data_path)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
