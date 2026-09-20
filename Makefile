# Conductress developer tasks. Targets mirror .github/workflows/tests.yml.
# Everything here acts on the active virtualenv only. Provisioning a benchmark
# runner (system packages, sysctl, systemd) is `conductress setup`, which is
# deliberately not a target.
# pytest and mypy run with PYTHONPATH=src so other editable installs on the
# host cannot shadow this tree.

PYTHON ?= python3

.PHONY: install lint format test integration typecheck ci

install:
	pip install -e '.[dev,control,plots]'

lint:
	black --check --line-length 120 src/ tests/
	isort --check-only --profile black --line-length 120 src/ tests/
	pylint --errors-only --disable=import-error src/

format:
	black --line-length 120 src/ tests/
	isort --profile black --line-length 120 src/ tests/

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit tests/control --cov=conductress --cov-fail-under=72

integration:
	PYTHONPATH=src $(PYTHON) -m pytest tests/integration -m "not requires_server"

typecheck:
	PYTHONPATH=src mypy src/ --ignore-missing-imports

ci: lint test integration typecheck
