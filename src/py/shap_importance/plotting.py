# plotting.py

"""
Beeswarm SHAP summary plot for shap_analysis.py.

Inputs: none (operates on in-memory SHAP values).
Outputs: writes the figure to each path in out_paths.
"""

import os

import matplotlib.pyplot as plt
import shap

def plot_beeswarm(shap_values, X, feature_names: list[str], title: str, out_paths: list[str]) -> None:
    """Render and save a SHAP beeswarm summary plot."""
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()

    for out_path in out_paths:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
    plt.close()
