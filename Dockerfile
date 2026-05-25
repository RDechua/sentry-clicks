# syntax=docker/dockerfile:1.7

# Native architecture (auto-detected from the host). The forced linux/amd64 pin was
# tried first and reverted — every container start re-paid the Rosetta cold-start
# cost on M-series (5+ min to import the ML wheels). Cloud/CI deployment on a
# different arch should rebuild the image on the target. See docs/decisions.md.
FROM python:3.11-slim-bookworm AS runtime

# Project venv lives outside /app so the `.:/app` bind mount in docker-compose
# doesn't shadow it at runtime.
ENV VENV_PATH=/opt/venv

# - UV_LINK_MODE=copy: don't hardlink across filesystems (bind-mount safety)
# - UV_COMPILE_BYTECODE=1: precompile .pyc at install time for faster cold imports
# - UV_PYTHON_DOWNLOADS=never: use the base image's Python; don't let uv fetch another
# - UV_PROJECT_ENVIRONMENT: pins the venv location (see VENV_PATH above)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=${VENV_PATH}

# OS-level deps. libgomp1 is required at LightGBM import time. ca-certificates for
# HTTPS roots. Cleaned in the same RUN to avoid baking apt caches into the layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv binary copied from its official image. Pinned to a specific version
# (not `:latest`) so rebuilds are reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv

# Non-root user. UID 1000 matches a typical host user, keeping bind-mount file
# permissions sane when the container writes back to the host.
ARG APP_UID=1000
RUN useradd --create-home --shell /bin/bash --uid ${APP_UID} sentry \
 && mkdir -p ${VENV_PATH} \
 && chown sentry:sentry ${VENV_PATH}

USER sentry
WORKDIR /app

# Deps layer — caches until pyproject.toml or uv.lock change.
COPY --chown=sentry:sentry pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Project layer — rebuilds on source or README changes. README is here (not in the
# deps layer) because hatchling reads it only when building the project itself.
COPY --chown=sentry:sentry README.md ./
COPY --chown=sentry:sentry src/ ./src/
RUN uv sync --frozen

ENV PATH="${VENV_PATH}/bin:$PATH"

# Default to a Python REPL; CLI entry points come in Task 1.10.
CMD ["python"]
