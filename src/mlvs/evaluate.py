"""Figures and tables for model comparison.

Produces three publication-ready figures from a training run: an overlay of the
cross-validated ROC curves, a small-multiples grid with one panel per model, and
a ranked comparison of mean AUC. Colour slots are assigned in fixed order and
never cycled; past eight models the overlay is skipped in favour of the grid,
since more than eight lines cannot be told apart reliably.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Categorical slots, assigned in fixed order. Validated for adjacent-pair CVD
# separation against both surfaces.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2de",
        "chance": "#8d8c86",
        "series": [
            "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948",
        ],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "chance": "#77766f",
        "series": [
            "#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767",
        ],
    },
}

MAX_OVERLAY_SERIES = 8

#: Human-readable axis and title text for metric keys.
METRIC_LABELS = {
    "auc": "AUC",
    "average_precision": "average precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "F1 score",
    "mcc": "Matthews correlation",
}


def _style(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9)
    ax.xaxis.label.set_color(theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])


def plot_roc_overlay(
    results: dict,
    path: str | Path,
    theme: str = "light",
    title: str = "Cross-validated ROC by model",
) -> Path | None:
    """Overlay every model's mean ROC curve on one axis."""
    palette = THEMES[theme]
    ordered = sorted(results.values(), key=lambda r: -r.mean_auc)

    if len(ordered) > MAX_OVERLAY_SERIES:
        log.warning(
            "%d models exceeds the %d-series overlay cap; use the ROC grid instead",
            len(ordered), MAX_OVERLAY_SERIES,
        )
        return None

    fig, ax = plt.subplots(figsize=(6.2, 5.4), facecolor=palette["surface"])
    _style(ax, palette)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5,
            color=palette["chance"], label="Chance (0.500)", zorder=1)

    for slot, result in enumerate(ordered):
        ax.plot(
            result.mean_fpr, result.mean_tpr,
            color=palette["series"][slot], linewidth=2, zorder=2 + slot,
            label=f"{result.label} ({result.mean_auc:.3f})",
        )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, color=palette["text"], fontsize=12, pad=12, loc="left")

    legend = ax.legend(
        loc="lower right", frameon=True, fontsize=9,
        facecolor=palette["surface"], edgecolor=palette["grid"],
        title="Model (mean AUC)",
    )
    legend.get_title().set_color(palette["muted"])
    legend.get_title().set_fontsize(9)
    for text in legend.get_texts():
        text.set_color(palette["text"])

    return _save(fig, path, palette)


def plot_roc_grid(
    results: dict, path: str | Path, theme: str = "light", n_cols: int = 3
) -> Path:
    """One ROC panel per model, so every model is legible regardless of count."""
    palette = THEMES[theme]
    ordered = sorted(results.values(), key=lambda r: -r.mean_auc)
    n_rows = int(np.ceil(len(ordered) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.1 * n_rows),
        facecolor=palette["surface"], squeeze=False,
    )

    for idx, ax in enumerate(axes.flat):
        if idx >= len(ordered):
            ax.axis("off")
            continue
        result = ordered[idx]
        _style(ax, palette)
        ax.plot([0, 1], [0, 1], "--", linewidth=1.2, color=palette["chance"])
        ax.plot(
            result.mean_fpr, result.mean_tpr,
            color=palette["series"][idx % len(palette["series"])], linewidth=2,
        )
        ax.set_title(
            f"{result.label}\nAUC {result.mean_auc:.3f} ± {result.std_auc:.3f}",
            fontsize=10, color=palette["text"], pad=8,
        )
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        if idx % n_cols == 0:
            ax.set_ylabel("True positive rate")
        if idx >= len(ordered) - n_cols:
            ax.set_xlabel("False positive rate")

    return _save(fig, path, palette)


def plot_model_comparison(
    results: dict,
    path: str | Path,
    metric: str = "auc",
    theme: str = "light",
) -> Path:
    """Ranked horizontal bars of one metric, with cross-validation spread."""
    palette = THEMES[theme]
    rows = []
    for result in results.values():
        values = [getattr(f, metric) for f in result.fold_metrics]
        rows.append((result.label, float(np.mean(values)), float(np.std(values))))
    rows.sort(key=lambda r: r[1])

    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(
        figsize=(7.2, 0.52 * len(rows) + 1.6), facecolor=palette["surface"]
    )
    _style(ax, palette)
    ax.grid(axis="y", visible=False)

    y = np.arange(len(rows))
    ax.barh(
        y, means, height=0.62,
        color=palette["series"][0],
        error_kw={"ecolor": palette["muted"], "elinewidth": 1.2, "capsize": 3},
        xerr=stds, zorder=2,
    )
    ax.set_yticks(y, labels, color=palette["text"], fontsize=10)

    # Headroom for the value labels; AUC caps at 1.0 but its label must still fit.
    ax.set_xlim(0, means.max() + stds.max() + 0.12)
    for yi, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            mean + std + 0.015, yi, f"{mean:.3f}",
            va="center", fontsize=9, color=palette["text"],
        )

    pretty = METRIC_LABELS.get(metric, metric.replace("_", " "))
    ax.set_xlabel(f"Mean {pretty} across cross-validation folds (± SD)")
    ax.set_title(
        f"Model comparison by {pretty}", color=palette["text"],
        fontsize=12, pad=12, loc="left",
    )
    return _save(fig, path, palette)


def _save(fig, path: str | Path, palette: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=palette["surface"], bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)
    return path


def metrics_table(results: dict) -> pd.DataFrame:
    """Mean ± SD of every fold metric, one row per model, best AUC first."""
    rows = []
    for result in results.values():
        row = {"model": result.key, "label": result.label, "family": result.family}
        if result.fold_metrics:
            for name in asdict(result.fold_metrics[0]):
                values = [getattr(f, name) for f in result.fold_metrics]
                row[f"{name}_mean"] = round(float(np.mean(values)), 4)
                row[f"{name}_std"] = round(float(np.std(values)), 4)
        row["fit_seconds"] = round(result.fit_seconds, 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


def write_report(
    results: dict,
    outdir: str | Path,
    theme: str = "light",
    table_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write all comparison figures and the metrics table.

    Figures go to ``outdir``; the metrics CSV goes to ``table_path`` when given,
    so it can sit beside the run's other tables rather than among the images.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    overlay = plot_roc_overlay(results, outdir / "roc_overlay.png", theme=theme)
    if overlay:
        written["roc_overlay"] = overlay
    written["roc_grid"] = plot_roc_grid(results, outdir / "roc_grid.png", theme=theme)
    written["comparison"] = plot_model_comparison(
        results, outdir / "model_comparison.png", theme=theme
    )

    table = Path(table_path) if table_path else outdir / "model_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    metrics_table(results).to_csv(table, index=False)
    written["metrics"] = table
    return written
