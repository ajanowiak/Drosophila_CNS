# core/features.py

"""
Builds time-agnostic feature matrices by stacking per-window motif
enrichment scores.

Pipeline context: shared across stages that need a time-agnostic dataset
(train_full_models, shap_importance) - lives in core/ specifically so it's
importable from any stage directory via PYTHONPATH=src/py.

Inputs:
  - <training_dir>/hrs<window>/motif_enrichment.csv
  - <training_dir>/hrs<window>/y_<tissue>.csv

Outputs: none (returns X, y, composite in memory; callers persist results).
"""

import logging
import os

import numpy as np
import pandas as pd

from core.constants import WINDOWS, PREV_WINDOW, FeatureMode
from core.paths import prepare_data_dir

logger = logging.getLogger(__name__)


def stack_windows(
    tissue: str,
    feature_mode: FeatureMode = FeatureMode.CURRENT,
    windows: list[str] = WINDOWS,
    training_dir: str = prepare_data_dir("unfiltered"),
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Stack per-window motif enrichment matrices into one time-agnostic dataset.

    feature_mode selects which window's enrichment scores become the feature
    columns for each main window w:
        CURRENT  - only w's own enrichment scores
        PREVIOUS - only the enrichment scores of the window preceding w
        EXPANDED - both, concatenated horizontally

    Only prev columns are suffixed (_prev), to disambiguate them from curr
    columns when both are concatenated (EXPANDED). Curr columns always keep
    raw motif-ID column names, in every mode - callers that join column
    names against the motif annotation table (e.g. shap_analysis.py) only
    ever see raw IDs for curr features.

    Returns:
        X (pd.DataFrame), y (pd.Series), composite (np.ndarray) - composite
        encodes (window_index, label) pairs for StratifiedKFold, so folds
        stay balanced across both developmental window and class.
    """
    use_curr = feature_mode in (FeatureMode.CURRENT, FeatureMode.EXPANDED)
    use_prev = feature_mode in (FeatureMode.PREVIOUS, FeatureMode.EXPANDED)

    Xs, ys = [], []

    for idx, w in enumerate(windows):
        y_w = pd.read_csv(
            os.path.join(training_dir, f"hrs{w}/y_{tissue}.csv"), index_col=0
        ).iloc[:, 0]

        sources = {}
        if use_curr:
            sources["curr"] = pd.read_csv(
                os.path.join(training_dir, f"hrs{w}/motif_enrichment.csv"),
                index_col=0,
            )
        if use_prev:
            prev_w = PREV_WINDOW[w]
            sources["prev"] = pd.read_csv(
                os.path.join(training_dir, f"hrs{prev_w}/motif_enrichment.csv"),
                index_col=0,
            )

        shared = y_w.index
        for src in sources.values():
            shared = shared.intersection(src.index)

        # some loop/tissue/window combinations have no presence annotation
        # (NaN in y_w itself), not just in the enrichment sources - both
        # have to be dropped, or a NaN label reaches classifier.fit() later.
        nan_mask = y_w.loc[shared].isna()
        for src in sources.values():
            nan_mask |= src.loc[shared].isna().any(axis=1)

        n_dropped = int(nan_mask.sum())
        keep = shared[~nan_mask]
        if n_dropped > 0:
            logger.info(f"[{tissue}] hrs{w}: dropped {n_dropped} loops with NaN ({len(keep)} remaining)")

        y_w = y_w.loc[keep]

        parts = []
        for name, src in sources.items():
            block = src.loc[keep]
            if name == "prev":
                block = block.add_suffix("_prev")
            parts.append(block)

        X_w = pd.concat(parts, axis=1)
        X_w["_window"] = idx

        Xs.append(X_w)
        ys.append(y_w)

    X = pd.concat(Xs, axis=0)
    y = pd.concat(ys, axis=0)

    composite = pd.Categorical(list(zip(X["_window"], y))).codes
    X = X.drop(columns=["_window"])

    logger.info(
        f"[{tissue}] Feature matrix ({feature_mode.value}): {X.shape} | "
        f"positives: {int(y.sum())} / {len(y)}"
    )

    return X, y, composite


def load_single_window(
    tissue: str,
    window: str,
    training_dir: str = prepare_data_dir("unfiltered"),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load one window's motif enrichment features and presence labels for the
    time-specific training mode - no stacking across windows, just
    stack_windows()'s per-window loading and NaN handling for a single window.
    """
    X, y, _ = stack_windows(tissue, feature_mode=FeatureMode.CURRENT, windows=[window], training_dir=training_dir)
    return X, y
