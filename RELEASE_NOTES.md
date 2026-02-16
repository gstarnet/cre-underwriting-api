# Release Notes

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
