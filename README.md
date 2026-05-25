# Sentry-Clicks

Mobile ad click fraud detection. Project in progress — the full README is written in Week 7 of the build.

## Quickstart

Requirements: Docker Desktop.

```sh
# 1. Point the container at your Kaggle TalkingData files (kept outside the repo).
cp .env.example .env
# Then edit .env and set DATA_DIR to your actual path.

# 2. Build the container image.
docker compose build

# 3. Smoke-test that the package is importable.
docker compose run --rm sentry python -c "import sentry; print('ok')"
# Expected output: ok
```

See `docs/PRD.md` for the project's problem statement and design.
