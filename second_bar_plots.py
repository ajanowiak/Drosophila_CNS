import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import make_names_dict

WINDOWS = ["06-08", "10-12", "14-16"]
TISSUES = ["Neuroblasts", "Neurons", "Glia"]

COLORS = {
    "time_specific": "grey",
    "curr": "steelblue",
    "prev": "grey",
    "curr+prev": "firebrick"
}


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
                data[t].append((row["mean_auc"].values[0], row["std_auc"].values[0]))
            else:
                data[t].append((np.nan, np.nan))

    return data


def load_time_agnostic(short, mode):
    mode_tag = mode.replace("+", "plus")
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


# ---------------------------------------------------------------------
# Plot 1: Time-specific vs curr
# ---------------------------------------------------------------------

def plot_time_specific_vs_curr(short, full):

    ts_data = load_time_specific(short)
    curr_data = load_time_agnostic(short, "curr")

    x = np.arange(len(TISSUES))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, w in enumerate(WINDOWS):
        means = []
        stds = []

        for t in TISSUES:
            m, s = ts_data[t][i]
            means.append(m)
            stds.append(s)

        ax.bar(x + (i - 1) * width, means, width,
               color=COLORS["time_specific"], alpha=0.8,
               label=f"hrs{w}" if i == 0 else None)

        for j in range(len(x)):
            ax.errorbar(x[j] + (i - 1) * width, means[j], yerr=stds[j],
                        fmt='.', color="black", capsize=5)

    # curr bars
    means = [curr_data[t][0] for t in TISSUES]
    stds  = [curr_data[t][1] for t in TISSUES]

    ax.bar(x + width*1.5, means, width,
           color=COLORS["curr"], label="curr")

    for j in range(len(x)):
        ax.errorbar(x[j] + width*1.5, means[j], yerr=stds[j],
                    fmt='.', color="black", capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(TISSUES)
    ax.set_ylim(0, 1)

    ax.set_ylabel("ROC AUC")
    ax.set_title(f"{full}: Time-specific vs current window")
    ax.legend()

    out_dir = f"results/figures/time_comparison/{short}"
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{out_dir}/ts_vs_curr_{short}.{fmt}", dpi=300, bbox_inches="tight")

    plt.close()


# ---------------------------------------------------------------------
# Plot 2: Time-agnostic feature comparison
# ---------------------------------------------------------------------

def plot_time_agnostic_modes(short, full):

    data_prev = load_time_agnostic(short, "prev")
    data_curr = load_time_agnostic(short, "curr")
    data_both = load_time_agnostic(short, "curr+prev")

    x = np.arange(len(TISSUES))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = [
        ("prev", data_prev),
        ("curr", data_curr),
        ("curr+prev", data_both),
    ]

    for i, (mode, data) in enumerate(modes):
        means = [data[t][0] for t in TISSUES]
        stds  = [data[t][1] for t in TISSUES]

        ax.bar(x + (i - 1) * width, means, width,
               color=COLORS[mode], label=mode)

        for j in range(len(x)):
            ax.errorbar(x[j] + (i - 1) * width, means[j], yerr=stds[j],
                        fmt='.', color="black", capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(TISSUES)
    ax.set_ylim(0, 1)

    ax.set_ylabel("ROC AUC")
    ax.set_title(f"{full}: Time-agnostic feature comparison")
    ax.legend()

    out_dir = f"results/figures/time_comparison/{short}"
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ["png", "pdf"]:
        plt.savefig(f"{out_dir}/ta_modes_{short}.{fmt}", dpi=300, bbox_inches="tight")

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    names = make_names_dict()
    short = names[args.model]["short"]
    full  = names[args.model]["full"]

    plot_time_specific_vs_curr(short, full)
    plot_time_agnostic_modes(short, full)


if __name__ == "__main__":
    main()