PYTHON ?= python3

.PHONY: database-lineage-experiment database-lineage-fast

database-lineage-experiment:
	$(PYTHON) experiments/database_lineage/scripts/run_all.py --full

database-lineage-fast:
	$(PYTHON) experiments/database_lineage/scripts/run_all.py --fast
