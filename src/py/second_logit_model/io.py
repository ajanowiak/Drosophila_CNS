# io.py

"""
I/O helpers for logit_regression_significant_features.py: extracting the
significant feature list from regression_coefs.py's output, and saving
this stage's own summary and CV-result tables.

Inputs: results/regression_coefs/<shap_model>/<selection_tag>/<tissue>_summary.csv
Outputs: none (returns the significant feature list; save_* persist results).
"""

import os

import pandas as pd


def extract_significant_features(
    tissue: str, shap_model: str, selection_tag: str, p_value_threshold: float
) -> pd.Series:
    """
    Extract motif IDs with BH-adjusted p-value < p_value_threshold from
    regression_coefs.py's output for the same (tissue, shap_model, selection_tag).
    """
    summary_path = f"results/regression_coefs/{shap_model}/{selection_tag}/{tissue}_summary.csv"

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary_df = pd.read_csv(summary_path, index_col=0)
    significant_df = summary_df[summary_df["p_adjusted_bh"] < p_value_threshold]

    return pd.Series(significant_df.index.tolist())


def save_summary_table(df: pd.DataFrame, path: str) -> None:
    """Save a summary table to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def save_cv_result_row(row: dict, path: str) -> None:
    """Save a one-row CV result to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
