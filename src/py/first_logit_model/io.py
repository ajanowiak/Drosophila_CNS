# io.py

"""
I/O helpers for the first logit model stage: saving its own summary table,
and reading back an already-fit summary to recover the exact feature set
regression_coefs.py used (for logit_regression_aucroc.py).

Inputs: results/first_logit_model/regression_coefs/tables/<shap_model>/<selection_tag>/<tissue>.csv
Outputs: none (returns the feature list; save_summary_table() persists results).
"""

import os

import pandas as pd

from core.paths import regression_coefs_table_path


def save_summary_table(df: pd.DataFrame, path: str) -> None:
    """Save a summary table to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def extract_used_features(tissue: str, shap_model: str, selection_tag: str) -> pd.Series:
    """
    Return every motif ID regression_coefs.py actually fit for one
    (tissue, shap_model, selection_tag), read back from its saved summary
    table rather than recomputed - so this can never disagree with what
    was actually fit, and never repeats an expensive binary search.
    """
    summary_path = regression_coefs_table_path(shap_model, selection_tag, tissue)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary_df = pd.read_csv(summary_path, index_col=0)
    motif_ids = [motif_id for motif_id in summary_df.index if motif_id != "const"]

    return pd.Series(motif_ids)
