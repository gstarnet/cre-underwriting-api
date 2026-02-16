# AGENTS.md

This file defines the default operating protocol for Codex agents working in this repository.

## Scope

- Keep changes minimal and task-focused.
- Prefer editing files under `src/`, `tests/`, and `scripts/`.
- Do not edit generated artifacts under `models/`, `reports/`, or `data/raw/` unless the task explicitly asks for it.

## Environment Setup

Use one of these setup flows:

```zsh
make setup
```

or:

```zsh
./scripts/rebuild_env.sh
```

## Standard Commands

- Run API locally: `make run`
- Run unit tests: `make test`
- Run smoke tests (API must already be running): `make smoke`
- Full rebuild artifacts: `make rebuild`

## Verification Protocol

After code edits:

1. Run `make test`.
2. If API code changed (`src/api.py`, endpoint models, request/response behavior), run `make smoke`.
3. If training, feature engineering, or explainability changed (`src/train.py`, `src/data_load.py`, `src/explain.py`), run:
   - `python -m src.train`
   - `python -m src.explain`

## File Ownership Hints

- `src/api.py`: FastAPI routes and validation boundaries.
- `src/train.py`: training pipeline and model artifact creation.
- `src/explain.py`: feature importance artifacts for `/explainability`.
- `src/underwrite.py`: baseline underwriting logic.
- `src/underwrite_inst.py`: institutional underwriting logic.
- `src/whatif.py`: baseline scenario grid behavior.
- `src/whatif_inst.py`: institutional what-if behavior + guardrails.
- `tests/`: regression and behavior checks.
- `scripts/`: local operational helpers and smoke-test entrypoints.

## Coding Conventions

- Use explicit intra-package imports (example: `from src.data_load import load_cre_csv`).
- Keep endpoint behavior backward-compatible unless task explicitly requires breaking changes.
- Add or update tests for behavior changes.
- Keep comments brief and only where logic is not obvious.

