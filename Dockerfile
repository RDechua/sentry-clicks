# syntax=docker/dockerfile:1.7

# Forced linux/amd64 base — portable across local M-series, CI runners, and cloud x86 VMs.
# Choice recorded in docs/decisions.md (2026-05-25 — Base image and architecture).
FROM --platform=linux/amd64 python:3.11-slim-bookworm AS runtime

# - PYTHONDONTWRITEBYTECODE: no .pyc on disk
# - PYTHONUNBUFFERED: stdout/stderr flush immediately (matters for `docker compose logs`)
# - PIP_*: skip wheel cache and version-check chatter
# - UV_LINK_MODE=copy: don't hardlink across filesystems (safer in bind-mounted containers)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

# OS-level deps:
# - libgomp1: OpenMP runtime LightGBM requires at import time. Installed now so the
#   apt layer is fixed and stays cached when Python deps are added in Task 1.3.
# - ca-certificates: HTTPS roots for any future fetches (Kaggle, model registries).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv from its official image — no curl-pipe install, no version drift.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Non-root runtime user. UID 1000 matches a typical host user, which keeps bind-mount
# file permissions sane when the container writes back to the host.
ARG APP_UID=1000
RUN useradd --create-home --shell /bin/bash --uid ${APP_UID} sentry

WORKDIR /app

# Project metadata first (this layer caches as long as pyproject.toml/README don't change).
COPY --chown=sentry:sentry pyproject.toml README.md ./

# Source last (this layer rebuilds on any code change).
COPY --chown=sentry:sentry src/ ./src/

# Editable install. No runtime deps yet — those land in Task 1.3.
RUN uv pip install --system --no-cache -e .

USER sentry

# Default to a Python REPL; CLI entry points come in Task 1.10.
CMD ["python"]
