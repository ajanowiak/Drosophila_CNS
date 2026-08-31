# io.py

import os

import pandas as pd


def save_summary_table(df: pd.DataFrame, path: str) -> None:
    """Save a summary table to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
