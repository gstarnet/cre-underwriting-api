# Release Notes

## v0.3.0 - 2026-02-16

### Summary
Adds operational hardening and traceability across runtime config, logging/auth, dataset/model metadata, backtesting artifacts, explainability output, and CI verification.

### Changes (since `v0.2.0`)
- Added centralized environment-driven runtime settings in `src/config.py`:
  - network: `HOST`, `PORT`
  - logging: `LOG_LEVEL`, `LOG_JSON`
  - paths: `DATA_PATH`, `DATASET_METADATA_PATH`, `MODEL_PATH`, `MODEL_SNAPSHOT_PATH`, `REPORTS_DIR`, `EXPLAINABILITY_JSON_PATH`
  - validation: `TS_CV_SPLITS`
  - auth: `AUTH_MODE`, `AUTH_TOKEN`
- Added structured logging utilities in `src/logging_utils.py` and wired startup logging in `src/__main__.py`.
- Added HTTP middleware in `src/api.py` for:
  - response `X-Request-ID`
  - structured request completion logs
  - optional token auth mode (`AUTH_MODE=token`, excludes `/health`)
- Added dataset versioning helpers in `src/dataset_versioning.py`:
  - SHA-256 dataset checksum
  - metadata artifact at `data/metadata/dataset_metadata.json`
- Updated training flow in `src/train.py`:
  - writes dataset metadata during training
  - writes model snapshot metadata (`reports/model_snapshot.json`) with model version/time/hash and holdout metrics
- Expanded explainability output in `src/explain.py`:
  - includes traceability fields (`model_version`, `trained_at_utc`, `dataset_hash`)
  - includes `ridge_coefficients` when the selected model is Ridge
- Improved explainability API compatibility in `src/api.py`:
  - `/explainability` now ensures traceability keys are present in response payload
- Strengthened backtesting in `src/validate_ts.py`:
  - configurable fold count via `TS_CV_SPLITS` (default 8)
  - persisted fold-level artifact `reports/ts_cv_folds.json` with per-fold metrics/windows/model params/hash
- Added `make validate-ts` target in `Makefile`.
- Updated `scripts/rebuild_all.sh` cleanup/build flow for new metadata outputs.
- Added minimal API behavior coverage in `tests/test_api_basic.py`.
- Hardened CI workflow in `.github/workflows/ci.yml`:
  - dataset fallback generation only when dataset is missing
  - Docker smoke step (build image + run container + `curl /health`)
- Updated docs (`README.md`, `.env.example`) and ignore rules (`.gitignore`) for new runtime and artifact behavior.

### Notes
- Real CRE data replacement is not bundled in this release; synthetic dataset generation remains as fallback in local/CI workflows.
- `make test` depends on the `PYTHON` setting/environment; in non-venv shells use `PYTHON=./.venv/bin/python make test` if needed.

## v0.2.0 - 2026-02-16

### Summary
Introduces hybrid tabular + unstructured model inputs, document extraction support, and local workflow hardening updates.

### Changes (since `v0.1.1`)
- Added hybrid training path in `src/train.py`:
  - combines tabular features with TF-IDF text features (`unstructured_text`)
  - auto-disables TF-IDF branch when all text is empty (prevents empty-vocabulary failures)
- Added unstructured ingestion utilities in `src/unstructured.py`:
  - supports text extraction from `.txt`, `.md`, `.csv`, `.json`, `.log`, `.pdf`, `.docx`, and OCR-capable images
  - includes helpers for document-path parsing and unified text composition
- Updated API request handling in `src/api.py` to accept unstructured inputs:
  - `unstructured_text`
  - `deal_notes`
  - `document_paths`
  - unified preprocessing is applied across prediction and underwriting endpoints
- Added tests in `tests/test_unstructured.py` for parsing and extraction helpers.
- Updated dependencies in `requirements.txt` for document/OCR support:
  - `pypdf`
  - `python-docx`
  - `Pillow`
  - `pytesseract`
- Switched project license to MIT in `LICENSE` and aligned `README.md` license section.
- Improved local scripts/tooling:
  - `Makefile` smoke target now runs a configurable script list via `SMOKE_SCRIPTS`
  - `scripts/rebuild_all.sh` now prefers `.venv/bin/python` and falls back to `python3`
  - `.gitignore` now excludes `.vscode/`
  - `scripts/docker_build_run.sh` default port mapping adjusted (`mount-model`/`with-model`)

### Notes
- This release is backward-compatible for existing tabular-only payloads.
- OCR requires system Tesseract installation in addition to Python packages.

## v0.1.1 - 2026-02-16

### Summary
Improves repository ergonomics for Codex-assisted development and local contributor workflows.

### Changes
- Added `AGENTS.md` with repository-specific agent guidance and verification protocol.
- Added `Makefile` with standard local commands:
  - `make setup`
  - `make test`
  - `make run`
  - `make smoke`
  - `make rebuild`
- Added `pyproject.toml` with baseline `pytest` and `ruff` configuration.
- Updated `README.md` with a Codex-friendly workflow and fast/full verification guidance.
- Updated `requirements.txt` to include `pytest`.

### Notes
- No API endpoint surface changes.
- No model training or inference logic changes.
