# Sentry-Clicks — convenience commands. All quality targets run inside the dev
# container so the toolchain is identical regardless of host setup.

.PHONY: help build format lint test check coverage clean

# Run a command inside the sentry container, mounting the repo live.
DC := docker compose run --rm sentry

# Coverage flags applied only when explicitly requested (`make check` and
# `make coverage`), not on every `pytest` invocation — keeps the inner loop fast.
COV_OPTS := --cov --cov-report=term-missing --cov-report=xml --cov-fail-under=70

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Build the docker image
	docker compose build

format:  ## Auto-fix lint and reformat (ruff --fix + black)
	$(DC) sh -c "ruff check --fix . && black ."

lint:  ## Run ruff (check) and mypy
	$(DC) sh -c "ruff check . && mypy"

test:  ## Run pytest
	$(DC) pytest

check:  ## Format-check + lint + test (with coverage gate), single container start
	$(DC) sh -c "ruff check . && black --check . && mypy && pytest $(COV_OPTS)"

coverage:  ## Run pytest with coverage report (no other checks)
	$(DC) pytest $(COV_OPTS)

clean:  ## Remove generated caches on the host
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
