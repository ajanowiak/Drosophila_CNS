# logit_plots.py

"""
Coefficient forest plot and volcano plot for a fitted logit summary table.

Pipeline context: shared between first_logit_model (regression_coefs.py)
and second_logit_model (logit_regression_significant_features.py), which
both plot the output of core.logit_analysis.fit_logit_summary().

Inputs: none (operates on an in-memory summary DataFrame).
Outputs: writes the figure to each path in out_paths.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_failed_placeholder(message: str, out_paths: list[str]) -> None:
    """
    Save a placeholder figure carrying a short failure message, in place of
    a coefficient/volcano plot that couldn't be produced (e.g. the Logit fit
    itself raised a singular-matrix error). Keeps the file present so a
    failed fit is still an inspectable artifact rather than a missing file.
    """
    plt.figure(figsize=(8, 6))
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)

    for out_path in out_paths:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
    plt.close()


def plot_coeffs(summary_df, title: str, out_paths: list[str], top_n: int) -> None:
    """Plot logistic regression coefficients with 95% confidence intervals."""
    df_plot = summary_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        df_plot["coef"],
        df_plot.index,
        xerr=[
            df_plot["coef"] - df_plot["ci_lower"],
            df_plot["ci_upper"] - df_plot["coef"],
        ],
        fmt="o",
    )

    plt.axvline(0, linestyle="--", color="r")
    plt.xlabel("Coefficient (log-odds)")
    plt.title(title)
    plt.tight_layout()

    for out_path in out_paths:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
    plt.close()


def plot_volcano(summary_df, title: str, out_paths: list[str], p_thresh: float, effect_thresh: float) -> None:
    """Create a volcano plot of logistic regression coefficients."""
    df = summary_df.copy()
    p_column_name = "p_adjusted_bh"

    conditions = [
        (df[p_column_name] < p_thresh) & (df["coef"] > effect_thresh),
        (df[p_column_name] < p_thresh) & (df["coef"] < -effect_thresh),
        (df[p_column_name] < p_thresh),
    ]
    choices = ["positive_strong", "negative_strong", "significant_only"]
    df["category"] = np.select(conditions, choices, default="other")

    plt.figure(figsize=(7, 6))

    for cat, group in df.groupby("category"):
        if cat == "positive_strong":
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), label="Strong positive", c="firebrick")
        elif cat == "negative_strong":
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), label="Strong negative", c="navy")
        elif cat == "significant_only":
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), label="Significant small effect")
        else:
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), alpha=0.3, c="grey")

    plt.axhline(-np.log10(p_thresh), c="black", linestyle="dotted")
    plt.axvline(effect_thresh, c="black", linestyle="dotted")
    plt.axvline(-effect_thresh, c="black", linestyle="dotted")

    plt.xlabel("Effect size (coef)")
    plt.ylabel("-log10(adjusted p-value (Benjamini-Hochberg))")
    plt.title(title)

    plt.legend()
    plt.tight_layout()

    for out_path in out_paths:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
    plt.close()
