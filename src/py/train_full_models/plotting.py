# plotting.py

"""
Shared ROC curve plotting for this stage's training scripts.

Pipeline context: used by train_time_specific.py and train_time_agnostic.py.

Inputs: none (operates on an in-memory CVResult).
Outputs: writes the figure to each path in out_paths.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from cv import CVResult


def plot_roc(
    result: CVResult,
    title: str,
    out_paths: list[Path],
    color: str | None = None,
    line_alpha: float = 1.0,
    line_label: str | None = None,
    fill_label: str | None = None,
    dpi: int = 300,
) -> None:
    """
    Plot the mean ROC curve with a +/- 1 std. dev. band and save it.

    Args:
        result: aggregated CV metrics from cv.cross_validate().
        title: full plot title (caller builds the exact string).
        out_paths: file paths to save to; format is inferred per path.
        color: line/band color, or None for the default color cycle.
        line_alpha: opacity of the mean ROC line.
        line_label: legend label for the mean ROC line; defaults to
            "AUC = {mean_auc:.3f} +/- {std_auc:.3f}" if not given.
        fill_label: legend label for the std. dev. band, or None to leave
            the band out of the legend.
        dpi: resolution to save at.
    """
    if line_label is None:
        line_label = f"AUC = {result.mean_auc:.3f} ± {result.std_auc:.3f}"

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        result.mean_fpr, result.mean_tpr,
        label=line_label, lw=1, alpha=line_alpha, color=color,
    )
    ax.fill_between(
        result.mean_fpr, result.tprs_lower, result.tprs_upper,
        alpha=0.2, label=fill_label, color=color,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.grid(axis="both")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
