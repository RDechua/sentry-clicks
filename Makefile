# Sentry-Clicks — convenience commands. All quality targets run inside the dev
# container so the toolchain is identical regardless of host setup.

.PHONY: help build format lint test check clean

# Run a command inside the sentry container, mounting the repo live.
DC := docker compose run --rm sentry

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

check:  ## Format-check + lint + test, single container start
	$(DC) sh -c "ruff check . && black --check . && mypy && pytest"

clean:  ## Remove generated caches on the host
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
