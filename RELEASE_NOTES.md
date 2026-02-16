# Release Notes

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
