# explain.py

"""
Computes SHAP values for a trained classifier and summarizes them into a
per-motif importance table.

Inputs: none (operates on an in-memory classifier and feature matrix).
Outputs: none (returns SHAP values / a summary DataFrame in memory).
"""

import numpy as np
import pandas as pd
import shap

TREE_MODELS = {"RF", "XGB"}


def compute_shap_values(classifier, model: str, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for X, explaining P(y=1) only.

    Args:
        classifier: a trained, sklearn-style classifier.
        model: short model code (e.g. "RF"), used to pick the explainer.
        X: feature matrix the classifier was trained on.
    """
    if model in TREE_MODELS:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X)

        # normalize TreeExplainer output across shap versions
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values

    def predict_proba_pos(classifier, X):
        """Return P(y=1) only."""
        return classifier.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(lambda X: predict_proba_pos(classifier, X), X)
    return explainer.shap_values(X)


def summarize_shap_values(
    shap_values: np.ndarray, X: pd.DataFrame, id_to_name: dict[str, str]
) -> pd.DataFrame:
    """
    Aggregate per-motif SHAP statistics: mean, absolute mean, mean absolute,
    and standard deviation, sorted by absolute mean importance.

    motif_id/motif_name are derived from X's own column order, so they are
    guaranteed to line up with shap_values regardless of the annotation
    table's row order.
    """
    mean_vals = shap_values.mean(axis=0)
    std_vals = shap_values.std(axis=0)
    abs_mean_vals = np.abs(mean_vals)
    mean_abs_vals = np.abs(shap_values).mean(axis=0)

    return pd.DataFrame({
        "motif_id": X.columns,
        "motif_name": [id_to_name.get(motif_id, motif_id) for motif_id in X.columns],
        "mean_importance": mean_vals,
        "abs_mean_importance": abs_mean_vals,
        "mean_abs_importance": mean_abs_vals,
        "std_importance": std_vals,
    }).sort_values("abs_mean_importance", ascending=False)
