# compare_bar_plots.py

"""
Builds one grouped bar-chart comparison of mean CV AUCROC across tissues,
for one model architecture: either time-specific vs. time-agnostic, or
time-agnostic curr/prev/expanded feature modes against each other.

Pipeline context: reads the per-tissue CV summary CSVs written by
train_time_specific.py and train_time_agnostic.py.

Inputs:
  - results/train_full_models/train_time_specific/cv_aucroc_summary/<model>/hrs<window>/<tissue>.csv
  - results/train_full_models/train_time_agnostic/cv_aucroc_summary/<model>/<feature_mode>/<tissue>.csv

Outputs:
  - results/train_full_models/compare_bar_plots/figures/<model>/time_specific_vs_agnostic.{png,pdf}
    (-comparison time_specific_vs_agnostic)
  - results/train_full_models/compare_bar_plots/figures/<model>/curr_prev_expanded.{png,pdf}
    (-comparison curr_prev_expanded)
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.config import load_config
from core.constants import MODELS, TISSUES, WINDOWS
from core.log import configure_logging
from core.paths import compare_bar_plots_figure_dir
from train_full_models.io import load_time_specific, load_time_agnostic

logger = logging.getLogger(__name__)


def plot_time_specific_vs_agnostic(short, full):
    """Bar chart: time-specific (per window) vs. time-agnostic AUC, per tissue."""
    ts_data = load_time_specific(short)
    curr_data = load_time_agnostic(short, "curr")

    config = load_config()
    colors = config["plot_colors"]["time_specific_vs_agnostic"]
    style = config["plot_style"]
    bar_width = style["bar_width"]
    grouped_alpha = style["time_specific_vs_agnostic"]["grouped_alpha"]
    highlight_alpha = style["time_specific_vs_agnostic"]["highlight_alpha"]

    x = np.arange(len(TISSUES))
    fig, ax = plt.subplots(figsize=(10, 6))

    offsets = [-bar_width, 0, bar_width]

    # time-specific bars
    for i, (w, off) in enumerate(zip(WINDOWS, offsets)):
        means = []
        stds = []

        for t in TISSUES:
            m, s = ts_data[t][i]
            means.append(m)
            stds.append(s)

        ax.bar(x + off, means, bar_width,
               color=colors[f"time_specific_{i+1}"],
               alpha=grouped_alpha,
               label=f"hrs{w}")

        for j in range(len(x)):
            ax.errorbar(x[j] + off, means[j], yerr=stds[j],
                        fmt='none', color="black", capsize=4, lw=1)

    # time-agnostic
    gap = bar_width * 1.5
    curr_pos = x + bar_width + gap

    means = [curr_data[t][0] for t in TISSUES]
    stds = [curr_data[t][1] for t in TISSUES]

    ax.bar(curr_pos, means, bar_width,
           color=colors["curr"], alpha=highlight_alpha,
           label="time-agnostic")

    for j in range(len(x)):
        ax.errorbar(curr_pos[j], means[j], yerr=stds[j],
                    fmt='none', color="black", capsize=4, lw=1)

    ax.set_xticks(x + gap / 2)
    ax.set_xticklabels(TISSUES)

    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"{full}: Time-specific vs time-agnostic mean AUCROC ± 1 std.")

    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    out_dir = compare_bar_plots_figure_dir(short)
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{out_dir}/time_specific_vs_agnostic.{fmt}",
                    dpi=300, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved {out_dir}/time_specific_vs_agnostic.{{png,pdf}}")


def plot_time_agnostic_comparison(short, full):
    """Bar chart: time-agnostic prev vs. curr vs. expanded AUC, per tissue."""
    data_prev = load_time_agnostic(short, "prev")
    data_curr = load_time_agnostic(short, "curr")
    data_expanded = load_time_agnostic(short, "expanded")

    config = load_config()
    colors = config["plot_colors"]["curr_prev_expanded"]
    style = config["plot_style"]
    bar_width = style["bar_width"]
    grouped_alpha = style["curr_prev_expanded"]["grouped_alpha"]
    highlight_alpha = style["curr_prev_expanded"]["highlight_alpha"]

    x = np.arange(len(TISSUES))
    fig, ax = plt.subplots(figsize=(10, 6))

    offsets = [-bar_width / 2, bar_width / 2]

    # prev & curr cluster
    for i, (mode, data) in enumerate([("prev", data_prev), ("curr", data_curr)]):
        off = offsets[i]

        means = [data[t][0] for t in TISSUES]
        stds = [data[t][1] for t in TISSUES]

        ax.bar(x + off, means, bar_width,
               color=colors[mode],
               alpha=grouped_alpha,
               label=mode)

        for j in range(len(x)):
            ax.errorbar(x[j] + off, means[j], yerr=stds[j],
                        fmt='none', color="black", capsize=4, lw=1)

    # expanded (separate)
    gap = bar_width * 1
    pos = x + bar_width + gap

    means = [data_expanded[t][0] for t in TISSUES]
    stds = [data_expanded[t][1] for t in TISSUES]

    ax.bar(pos, means, bar_width,
           color=colors["expanded"],
           alpha=highlight_alpha,
           label="expanded")

    for j in range(len(x)):
        ax.errorbar(pos[j], means[j], yerr=stds[j],
                    fmt='none', color="black", capsize=4, lw=1)

    ax.set_xticks(x + gap / 2)
    ax.set_xticklabels(TISSUES)

    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"{full}: Time-agnostic vs expanded mean AUCROC ± 1 std.")

    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    out_dir = compare_bar_plots_figure_dir(short)
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{out_dir}/curr_prev_expanded.{fmt}",
                    dpi=300, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved {out_dir}/curr_prev_expanded.{{png,pdf}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one model AUC comparison")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument(
        "--comparison",
        required=True,
        choices=["time_specific_vs_agnostic", "curr_prev_expanded"],
    )
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)

    short = args.model
    full = MODELS[short]["full"]

    if args.comparison == "time_specific_vs_agnostic":
        plot_time_specific_vs_agnostic(short, full)
    elif args.comparison == "curr_prev_expanded":
        plot_time_agnostic_comparison(short, full)

    logger.info("Done.")


if __name__ == "__main__":
    main()
