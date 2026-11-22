"""Figure helpers. Every figure carries an insight computed from the data at
runtime, so a refreshed dataset never leaves a stale caption behind."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg") if not hasattr(matplotlib, "get_ipython") else None
import matplotlib.pyplot as plt  # noqa: E402

from . import config as cfg


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def stamp_and_save(fig, path: Path | str, insight: str = "", dpi: int = 150) -> Path:
    """Add a caption box under the figure and save it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if insight:
        fig_w, fig_h = fig.get_size_inches()
        text = textwrap.fill(" ".join(str(insight).split()), width=int(fig_w * 12))
        pad = (0.35 + 0.24 * (text.count("\n") + 1)) / fig_h
        fig.tight_layout(rect=[0, pad, 1, 1])
        fig.text(
            0.5,
            pad / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3CD", ec="#E0A800", alpha=0.95),
        )
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def eda_figure_path(question_no: int, slug: str) -> Path:
    return cfg.FIGURES_DIR / f"q{question_no:02d}_{slug}.png"


def plot_classification_report(y_true, y_proba, y_pred, importance=None, title="Model evaluation"):
    """2x2 panel: ROC, PR, confusion matrix, top features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="grey", label="random")
    axes[0, 0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    base = float(np.mean(y_true))
    axes[0, 1].plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}")
    axes[0, 1].axhline(base, ls="--", color="grey", label=f"baseline = {base:.3f}")
    axes[0, 1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    axes[1, 0].imshow(cm, cmap="Blues")
    axes[1, 0].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["pred on-time", "pred late"],
        yticklabels=["actual on-time", "actual late"],
        title="Confusion matrix at chosen threshold",
    )
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    if importance is not None and len(importance):
        top = importance.head(10).iloc[::-1]
        axes[1, 1].barh(top["feature"], top["importance"], color="#2a9d8f")
        axes[1, 1].set(xlabel="Permutation importance", title="Top features (held-out)")
        axes[1, 1].grid(axis="x", alpha=0.3)
    else:
        axes[1, 1].axis("off")

    fig.suptitle(title, fontsize=15)
    return fig


def plot_cost_curve(policy_table: pd.DataFrame, cost_fp: float, cost_fn: float):
    best = policy_table.loc[policy_table["expected_cost"].idxmin()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(policy_table["threshold"], policy_table["expected_cost"], color="#264653")
    ax.axvline(best["threshold"], ls="--", color="#e76f51",
               label=f"min cost @ {best['threshold']:.3f}")
    ax.set(
        xlabel="Decision threshold",
        ylabel="Expected cost (assumed units)",
        title=f"Expected cost vs threshold (FP={cost_fp}, FN={cost_fn})",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_calibration(calib: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, calib["mean_predicted"].max()], [0, calib["mean_predicted"].max()],
            "--", color="grey", label="perfect calibration")
    ax.plot(calib["mean_predicted"], calib["observed_rate"], "o-", color="#2a9d8f", label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed rate",
           title="Calibration by decile (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_regression_diagnostics(y_true, y_pred, unit="days", by: pd.Series | None = None):
    """Predicted vs actual, residual distribution, and error by segment."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    n_panels = 3 if by is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    lim = np.nanpercentile(np.concatenate([y_true, y_pred]), 99)
    axes[0].scatter(y_true, y_pred, s=4, alpha=0.15, color="#264653")
    axes[0].plot([0, lim], [0, lim], "--", color="#e76f51")
    axes[0].set(xlim=(0, lim), ylim=(0, lim), xlabel=f"Actual ({unit})",
                ylabel=f"Predicted ({unit})", title="Predicted vs actual")
    axes[0].grid(alpha=0.3)

    axes[1].hist(np.clip(residual, np.nanpercentile(residual, 1), np.nanpercentile(residual, 99)),
                 bins=60, color="#2a9d8f")
    axes[1].axvline(0, color="black", ls="--")
    axes[1].set(xlabel=f"Residual ({unit}, predicted - actual)", ylabel="Orders",
                title=f"Residuals (median abs err {np.median(np.abs(residual)):.2f} {unit})")
    axes[1].grid(alpha=0.3)

    if by is not None:
        err = pd.DataFrame({"segment": np.asarray(by), "abs_err": np.abs(residual), "bias": residual})
        agg = err.groupby("segment", observed=True).agg(mae=("abs_err", "mean"), bias=("bias", "mean"),
                                                        n=("abs_err", "size"))
        agg = agg[agg["n"] >= 30]
        axes[2].bar(agg.index.astype(str), agg["mae"], color="#e9c46a", label="MAE")
        axes[2].plot(agg.index.astype(str), agg["bias"], "o-", color="#e76f51", label="mean bias")
        axes[2].axhline(0, color="black", lw=0.5)
        axes[2].set(ylabel=unit, title="Error by segment (n>=30)")
        axes[2].tick_params(axis="x", rotation=30)
        axes[2].legend()
        axes[2].grid(axis="y", alpha=0.3)
    return fig
