"""Generate the figures embedded in the PDF project report.

One-off reporting script (Week 8). Produces reports/figs/results.png: a
two-panel summary — detection lift (PR-AUC vs. baselines) and the modeled
enforcement-cost reduction. Numbers are the held-out-test / validation figures
documented in docs/PROJECT_REPORT.md and docs/decisions.md.

Run inside the container (matplotlib lives there):
    docker compose run --rm sentry python experiments/report_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless

import matplotlib.pyplot as plt

OUT = Path("reports/figs")

# Ink colors: Sentry highlighted, baselines muted.
SENTRY = "#1f5f3f"
MUTED = "#9aa0a6"


def main() -> None:
    """Render the two-panel results figure to reports/figs/results.png."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel 1 — PR-AUC lift over baselines (held-out test).
    labels = ["Sentry\n(LightGBM)", "Best single\nfeature", "Random\n(base rate)"]
    prauc = [0.559, 0.11, 0.0025]
    bars = ax1.bar(labels, prauc, color=[SENTRY, MUTED, MUTED], width=0.6)
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("Detection performance (held-out test)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 0.62)
    for b, v in zip(bars, prauc, strict=True):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2 — modeled enforcement cost, log scale (validation).
    clabels = ["Block nothing", "Sentry policy"]
    cost = [1_116_537, 1_854]
    cbars = ax2.bar(clabels, cost, color=[MUTED, SENTRY], width=0.55)
    ax2.set_yscale("log")
    ax2.set_ylabel("Modeled cost (USD, log scale)")
    ax2.set_title("Enforcement cost (validation)", fontsize=11, fontweight="bold")
    ax2.set_ylim(1e3, 5e6)
    for b, v in zip(cbars, cost, strict=True):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.3, f"${v:,.0f}", ha="center", fontsize=10)
    ax2.annotate(
        "99.8% reduction",
        xy=(1, 1854),
        xytext=(0.5, 6e4),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=SENTRY,
        arrowprops={"arrowstyle": "->", "color": SENTRY},
    )
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = OUT / "results.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
