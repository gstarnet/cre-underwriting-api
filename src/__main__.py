# src/__main__.py
from __future__ import annotations

import uvicorn

from src.config import get_settings
from src.logging_utils import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    uvicorn.run("src.api:app", host=settings.host, port=settings.port, reload=False)

if __name__ == "__main__":
    main()
