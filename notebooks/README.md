# notebooks/

This directory contains exploratory Jupyter notebooks — EDA, model debugging, feature inspection, ad-hoc plots.

Notebooks are **not** production code. They are deliberately allowed to be messy, partial, and contain in-progress thinking. They're checked in so the project's exploratory work is reproducible and visible, not because they're held to the same standard as `src/sentry/`.

## Rules

- **No imports into `src/sentry/`.** If logic in a notebook turns out to be useful, promote it into `src/` first, then import the function from there in the notebook.
- **Clear cell outputs before commit.** Large outputs bloat the repo and the diff. The exception is notebooks where the output is the artifact (a one-shot data-quality report, for example) — note that explicitly at the top of the notebook.
- **Naming:** `YYYY-MM-DD-short-description.ipynb`. Chronological order then matches discovery order.
