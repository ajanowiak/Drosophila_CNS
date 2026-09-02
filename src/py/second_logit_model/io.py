# io.py

"""
I/O helpers for logit_regression_significant_features.py: extracting the
significant feature list from regression_coefs.py's output, and saving
this stage's own summary and CV-result tables.

Inputs: results/first_logit_model/regression_coefs/tables/<shap_model>/<selection_tag>/<tissue>.csv
Outputs: none (returns the significant feature list; save_* persist results).
"""

import os

import pandas as pd

from core.paths import regression_coefs_table_path


def extract_significant_features(
    tissue: str, shap_model: str, selection_tag: str, p_value_threshold: float
) -> pd.Series:
    """
    Extract motif IDs with BH-adjusted p-value < p_value_threshold from
    regression_coefs.py's output for the same (tissue, shap_model, selection_tag).
    """
    summary_path = regression_coefs_table_path(shap_model, selection_tag, tissue)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary_df = pd.read_csv(summary_path, index_col=0)
    significant_df = summary_df[summary_df["p_adjusted_bh"] < p_value_threshold]

    # "const" is the fitted intercept, not a real motif column in X - drop
    # it even when it's itself BH-significant (see first_logit_model/io.py's
    # extract_used_features, which excludes it the same way).
    motif_ids = [motif_id for motif_id in significant_df.index if motif_id != "const"]

    return pd.Series(motif_ids)


def save_summary_table(df: pd.DataFrame, path: str) -> None:
    """Save a summary table to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def save_cv_result_row(row: dict, path: str) -> None:
    """Save a one-row CV result to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
