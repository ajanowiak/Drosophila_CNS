# train_full_models/second_bar_plots.py

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import make_names_dict

WINDOWS = ["06-08", "10-12", "14-16"]
TISSUES = ["Neuroblasts", "Neurons", "Glia"]

BAR_WIDTH = 0.15
ALPHA1 = 0.6
ALPHA2 = 0.85

# Model mapping
def resolve_model(model_key):
    mapping = {
        "rf": "RandomForestClassifier",
        "svm": "SVC",
        "lr": "LogisticRegression",
        "xgb": "XGBClassifier",
    }

    model_full_name = mapping[model_key]
    names = make_names_dict()

    short = names[model_full_name]["short"]
    full = names[model_full_name]["full"]

    return short, full


# Data loaders
def load_time_specific(short):
    data = {t: [] for t in TISSUES}

    for w in WINDOWS:
        path = f"results/time_specific/{short}/hrs{w}/cv_aucroc_summary_{short}_hrs{w}.csv"

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        for t in TISSUES:
            row = df[df["tissue"] == t]

            if not row.empty:
                data[t].append((row["mean_auc"].values[0],
                                row["std_auc"].values[0]))
            else:
                data[t].append((np.nan, np.nan))

    return data


def load_time_agnostic(short, mode_tag):
    path = f"results/time_agnostic/{short}/{mode_tag}/cv_aucroc_summary_{short}_{mode_tag}.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    data = {}
    for t in TISSUES:
        row = df[df["tissue"] == t]
        data[t] = (
            row["mean_auc"].values[0],
            row["std_auc"].values[0]
        )

    return data


# Plotting
def plot_time_specific_vs_agnostic(short, full):
    ts_data = load_time_specific(short)
    curr_data = load_time_agnostic(short, "curr")

    cmap = cm.get_cmap("Blues")

    COLORS = {
        "time_specific_1": cmap(0.45),
        "time_specific_2": cmap(0.55),
        "time_specific_3": cmap(0.65),
        "curr": cmap(0.85),
    }

    x = np.arange(len(TISSUES))
    fig, ax = plt.subplots(figsize=(10, 6))

    offsets = [-BAR_WIDTH, 0, BAR_WIDTH]

    # time-specific bars
    for i, (w, off) in enumerate(zip(WINDOWS, offsets)):
        means = []
        stds = []

        for t in TISSUES:
            m, s = ts_data[t][i]
            means.append(m)
            stds.append(s)

        ax.bar(x + off, means, BAR_WIDTH,
               color=COLORS[f"time_specific_{i+1}"],
               alpha=ALPHA1,
               label=f"hrs{w}")

        for j in range(len(x)):
            ax.errorbar(x[j] + off, means[j], yerr=stds[j],
                        fmt='none', color="black", capsize=4, lw=1)

    # time-agnostic
    gap = BAR_WIDTH * 1.5
    curr_pos = x + BAR_WIDTH + gap

    means = [curr_data[t][0] for t in TISSUES]
    stds = [curr_data[t][1] for t in TISSUES]

    ax.bar(curr_pos, means, BAR_WIDTH,
           color=COLORS["curr"], alpha=ALPHA2,
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

    out_dir = f"results/figures/expanded_bar_plots/{short}"
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{out_dir}/time_specific_vs_agnostic_{short}.{fmt}",
                    dpi=300, bbox_inches="tight")

    plt.close(fig)

def plot_time_agnostic_comparison(short, full):
    data_prev = load_time_agnostic(short, "prev")
    data_curr = load_time_agnostic(short, "curr")
    data_expanded = load_time_agnostic(short, "expanded")

    cmap = cm.get_cmap("Blues")
    COLORS = {
        "prev": cmap(0.7),
        "curr": cmap(0.85),
        "expanded": "firebrick"
    }

    x = np.arange(len(TISSUES))
    fig, ax = plt.subplots(figsize=(10, 6))

    offsets = [-BAR_WIDTH / 2, BAR_WIDTH / 2]

    # prev & curr cluster
    for i, (mode, data) in enumerate([("prev", data_prev), ("curr", data_curr)]):
        off = offsets[i]

        means = [data[t][0] for t in TISSUES]
        stds = [data[t][1] for t in TISSUES]

        ax.bar(x + off, means, BAR_WIDTH,
               color=COLORS[mode],
               alpha=ALPHA1,
               label=mode)

        for j in range(len(x)):
            ax.errorbar(x[j] + off, means[j], yerr=stds[j],
                        fmt='none', color="black", capsize=4, lw=1)

    # expanded (separate)
    gap = BAR_WIDTH * 1
    pos = x + BAR_WIDTH + gap

    means = [data_expanded[t][0] for t in TISSUES]
    stds = [data_expanded[t][1] for t in TISSUES]

    ax.bar(pos, means, BAR_WIDTH,
           color=COLORS["expanded"],
           alpha=0.7,
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

    out_dir = f"results/figures/expanded_bar_plots/{short}"
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{out_dir}/curr_prev_expanded_{short}.{fmt}",
                    dpi=300, bbox_inches="tight")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot model AUC comparisons")

    parser.add_argument(
        "--model",
        required=True,
        choices=["rf", "svm", "xgb", "lr"],
        help="Model type"
    )

    args = parser.parse_args()

    short, full = resolve_model(args.model)

    plot_time_specific_vs_agnostic(short, full)
    plot_time_agnostic_comparison(short, full)


if __name__ == "__main__":
    main()