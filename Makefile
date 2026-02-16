.PHONY: setup run test smoke rebuild train explain sanity

PYTHON ?= python3
SMOKE_SCRIPTS ?= ./scripts/smoke_api.sh ./scripts/test_explainability.sh ./scripts/test_whatif_inst.sh

setup:
	./scripts/rebuild_env.sh

run:
	$(PYTHON) -m src

test:
	$(PYTHON) -m pytest -q

smoke:
	@for s in $(SMOKE_SCRIPTS); do \
		echo "== smoke: $$s =="; \
		$$s; \
	done

rebuild:
	./scripts/rebuild_all.sh

train:
	$(PYTHON) -m src.train

explain:
	$(PYTHON) -m src.explain

sanity:
	$(PYTHON) -m src.sanity
