.PHONY: setup run test smoke rebuild train explain sanity

PYTHON ?= python3

setup:
	./scripts/rebuild_env.sh

run:
	$(PYTHON) -m src

test:
	$(PYTHON) -m pytest -q

smoke:
	./scripts/smoke_api.sh

rebuild:
	./scripts/rebuild_all.sh

train:
	$(PYTHON) -m src.train

explain:
	$(PYTHON) -m src.explain

sanity:
	$(PYTHON) -m src.sanity
