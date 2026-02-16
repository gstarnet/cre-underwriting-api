# CRE Underwriting API

Local, production-oriented framework for experimenting with CRE (commercial real estate) NOI forecasting and underwriting workflows:
- Train a baseline ML model (tabular) to forecast next-12-month NOI.
- Run underwriting / ROI calculations (simple + “institutional-style” v2).
- Run what-if scenario grids (simple + institutional) with guardrails.
- Serve everything via FastAPI for local testing and containerized runs.
- Generate explainability artifacts (permutation importance) for API consumption.

> Status: working end-to-end locally (train → explain → serve → smoke tests) and in Docker (mount-model + with-model targets).


## Quick start (local)

### 1) Create venv + install deps
```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use the helper script:
```zsh
./scripts/rebuild_env.sh
```

### 2) Train model + metrics
```zsh
python -m src.train
```

### 3) Generate explainability artifacts
```zsh
python -m src.explain
```

### 4) Start API
Serves on `0.0.0.0:8000`.
```zsh
python -m src
```

### 5) Run smoke tests
```zsh
./scripts/smoke_api.sh
```

Optional focused tests:
```zsh
BASE=http://127.0.0.1:8000 ./scripts/test_whatif_inst.sh
./scripts/test_explainability.sh
pytest -q
```

## Codex-friendly workflow

Use stable command targets so local and automated workflows stay aligned:

```zsh
make setup
make test
make run
make smoke
make rebuild
```

Verification levels:

- Fast verify: `make test`
- Full verify:
  1. Start API (`make run`)
  2. Run smoke tests (`make smoke`)
  3. If model/explainability logic changed, run:
     - `python -m src.train`
     - `python -m src.explain`

Change protocol:

- API contract changes: run `make test` and `make smoke`
- Training/data/explainability changes: run `make test`, `python -m src.train`, and `python -m src.explain`
- Keep generated artifacts (`models/`, `reports/`, `data/raw/`) out of source-control changes unless explicitly requested


## Docker

Two deployment modes are supported:

### A) Mount model at runtime (`mount-model`)
Build:
```zsh
docker build --target mount-model -t cre-underwriting-api:mount .
```

Run (mount local `models/` into the container):
```zsh
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models" cre-underwriting-api:mount
```

### B) Bake model into the image (`with-model`)
Build:
```zsh
docker build --target with-model -t cre-underwriting-api:with-model .
```

Run:
```zsh
docker run --rm -p 8000:8000 cre-underwriting-api:with-model
```

Or use the helper:
```zsh
./scripts/docker_build_run.sh build
./scripts/docker_build_run.sh run mount-model
./scripts/docker_build_run.sh run with-model
./scripts/docker_build_run.sh stop
```


## API endpoints (FastAPI)

Base URL: `http://127.0.0.1:8000`

- `GET  /health` — basic health check
- `POST /predict` — predict next-12 NOI
- `POST /predict_features` — return feature vector used for prediction (debug/visibility)
- `POST /whatif` — simple what-if scenario grid (price / exit cap variations)
- `POST /underwrite` — simple multi-year underwriting
- `POST /underwrite_inst` — institutional underwriting (v2 knobs)
- `POST /whatif_inst` — institutional what-if scenario grid with validation + guardrails
- `GET  /explainability` — exposes `reports/feature_importance.json` (permutation importance)

Smoke tests cover the full surface area.


## Generated artifacts (ignored by git)

These outputs are generated locally and should be excluded from version control:

- `data/raw/cre_deals.csv`
- `models/model.joblib`
- `reports/metrics.csv`
- `reports/ts_cv_metrics.csv`
- `reports/ts_cv_summary.csv`
- `reports/feature_importance.csv`
- `reports/feature_importance.json`

To rebuild everything from scratch:
```zsh
./scripts/rebuild_all.sh
```


## Scripts

### `scripts/rebuild_env.sh`
Rebuilds the local Python environment (creates `.venv`, installs dependencies).

```zsh
./scripts/rebuild_env.sh
```

### `scripts/rebuild_all.sh`
One-command rebuild of all generated artifacts (synthetic data, train, validation, explainability).

```zsh
./scripts/rebuild_all.sh
```

### `scripts/docker_build_run.sh`
Build and (optionally) run Docker images for both deployment modes:
- `mount-model`: container expects `models/` mounted at runtime
- `with-model`: model is baked into the image

Examples:
```zsh
# build both targets
./scripts/docker_build_run.sh build

# run mount-model (mounts local models/)
./scripts/docker_build_run.sh run mount-model

# run with-model (no volume mount)
./scripts/docker_build_run.sh run with-model

# stop running container (if supported by script)
./scripts/docker_build_run.sh stop
```

### `scripts/smoke_api.sh`
Runs end-to-end checks against a running API:
- health
- predict
- whatif
- underwrite
- underwrite_inst
- whatif_inst

```zsh
./scripts/smoke_api.sh
```

### `scripts/test_whatif_inst.sh`
Focused tests for `/whatif_inst` hardening (happy path + invalid inputs + guardrail).

```zsh
BASE=http://127.0.0.1:8000 ./scripts/test_whatif_inst.sh
```

### `scripts/test_explainability.sh`
Verifies:
- model exists
- `src.explain` runs
- CSV/JSON output schema
- `/explainability` endpoint schema (if present)

```zsh
./scripts/test_explainability.sh
```


## Sanity + package notes

### `src/sanity.py`
Quick verification of:
- dataset loads
- time-based split behaves as expected
- basic stats print

```zsh
python -m src.sanity
```

### `src/__init__.py`
Keeps `src/` as a proper Python package and may expose a minimal public surface.

Avoid `from src import ...` style imports for internal modules; prefer explicit module imports such as:
```python
from src.data_load import load_cre_csv
```


## Tests

### `tests/test_whatif_exit_cap_default.py`
Regression test: confirms `exit_cap_rate` default/behavior in what-if logic.

Run:
```zsh
pytest -q
```


## Repository layout (high level)

- `src/`
  - `__main__.py` — `python -m src` entrypoint (Uvicorn)
  - `api.py` — FastAPI app + endpoints
  - `data_load.py` — load + time-based split utilities
  - `train.py` — training pipeline (produces `models/model.joblib` + metrics)
  - `validate_ts.py` — time-series CV metrics (walk-forward style)
  - `explain.py` — permutation importance → `reports/feature_importance.*`
  - `roi.py` — ROI engine / finance helpers
  - `underwrite.py` — simple underwriting model
  - `underwrite_inst.py` — institutional underwriting v2
  - `whatif.py` — simple what-if scenario generation
  - `whatif_inst.py` — institutional what-if with validation + guardrails
  - `sanity.py` — basic sanity checks
- `scripts/`
  - `smoke_api.sh`
  - `test_whatif_inst.sh`
  - `test_explainability.sh`
  - `rebuild_env.sh`
  - `rebuild_all.sh`
  - `docker_build_run.sh`
- `tests/`
  - `test_whatif_exit_cap_default.py`
- `models/` (generated)
- `reports/` (generated)
- `data/raw/` (generated)


## License
Internal/proprietary (Cloud Breeze). See `LICENSE`.
