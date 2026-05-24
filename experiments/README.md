# experiments/

One-off scripts that don't belong in `src/sentry/` — quick benchmarks, scratch analyses, throwaway ablations, debugging scripts. Each file is self-contained.

Like `notebooks/`, this is not production code. It's checked in so the project's experimental work is visible and reproducible.

## Rules

- **No imports from `experiments/` into `src/sentry/`.** If something here is useful enough to depend on, promote it into `src/` first.
- **Self-contained:** each script reads its own data, defines its own helpers, and produces its own output. Don't share state between scripts.
- **Document at the top of each file:** what question it was trying to answer, what was found, and (if relevant) which decision in `docs/decisions.md` it informed.
