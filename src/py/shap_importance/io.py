# io.py

"""
I/O helpers for shap_analysis.py: loading a trained classifier and saving
the resulting SHAP importance table.

Inputs: a pickled classifier.
Outputs: none (returns the loaded classifier; save_shap_table() persists results).
"""

import os
import pickle

import pandas as pd


def load_model(path: str):
    """Unpickle a trained classifier."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_shap_table(df: pd.DataFrame, path: str) -> None:
    """Save the SHAP importance table to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
