# io.py

import os
import pickle

import numpy as np
import pandas as pd

from core.constants import TISSUES, WINDOWS
from core.paths import train_time_specific_summary_path, train_time_agnostic_summary_path

def save_model(model, path: str) -> None:
    """Pickle a fitted model, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(model, open(path, "wb"))

def save_cv_result(result, path: str) -> None:
    """Pickle a CVResult, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(result, open(path, "wb"))


def load_cv_result(path: str):
    """Load a CVResult pickled by save_cv_result()."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_summary_row(row: dict, path: str) -> None:
    """Write a one-row CV summary to CSV, creating its parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def load_time_specific(short: str) -> dict:
    """Load mean/std AUC per (window, tissue) for a time-specific model, for compare_bar_plots.py."""
    data = {t: [] for t in TISSUES}

    for w in WINDOWS:
        for t in TISSUES:
            path = train_time_specific_summary_path(short, w, t)

            if not os.path.exists(path):
                data[t].append((np.nan, np.nan))
                continue

            df = pd.read_csv(path)
            data[t].append((df["mean_auc"].values[0], df["std_auc"].values[0]))

    return data


def load_time_agnostic(short: str, mode_tag: str) -> dict:
    """Load mean/std AUC per tissue for a time-agnostic model and feature mode, for compare_bar_plots.py."""
    data = {}

    for t in TISSUES:
        path = train_time_agnostic_summary_path(short, mode_tag, t)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        data[t] = (df["mean_auc"].values[0], df["std_auc"].values[0])

    return data
