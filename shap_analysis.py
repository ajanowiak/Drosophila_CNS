# shap_analysis.py

import shap
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no GUI memory overhead
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

from utils import make_names_dict, compose_windows


# ── Memory helpers ────────────────────────────────────────────────────────────

def _get_kernel_background(X: pd.DataFrame, n_background: int = 100) -> pd.DataFrame:
    """
    Summarise X into a small background dataset using k-means.

    KernelExplainer only needs a representative background, not the full
    training set. shap.kmeans is the canonical way to do this and does NOT
    reduce the accuracy of the SHAP values — it is how the method is designed
    to be used on large datasets.

    Rule of thumb: 50-200 cluster centres are plenty.
    """
    return shap.kmeans(X, min(n_background, len(X)))


def _sample_foreground(X: pd.DataFrame, max_explain: int, random_state: int) -> pd.DataFrame:
    """
    Optionally subsample the rows we actually *explain*.

    For KernelExplainer, explaining every row is O(n²) in time and memory.
    Explaining a random sample of ~500–1000 rows still gives a faithful
    picture of global feature importance (SHAP beeswarms / mean |SHAP|).
    Tree-based models are fast enough to explain everything, so we skip
    sampling for them.
    """
    if max_explain and len(X) > max_explain:
        return X.sample(n=max_explain, random_state=random_state)
    return X


# ── Core analysis function ────────────────────────────────────────────────────

def shap_analysis_with_beeswarm(
    classifier_path: str,
    tissue: str,
    motif_annotations_path: str = None,
    motif_annotations_sep: str = None,
    windows: list = ["06-08", "10-12", "14-16"],
    top_n_motifs: int = 20,
    random_state: int = 0,
    # --- new memory-control knobs ---
    kernel_background_size: int = 100,   # k-means centres for KernelExplainer background
    kernel_max_explain: int = 500,       # max rows to explain with KernelExplainer
):
    """
    Perform SHAP-based interpretability analysis for a pretrained
    time-agnostic chromatin loop classifier.

    Changes vs. original
    --------------------
    * KernelExplainer (SVM / LR) now uses a k-means background summary
      instead of the full dataset — the standard memory-reduction technique
      recommended in the SHAP docs.
    * The foreground (rows being explained) is optionally subsampled for
      KernelExplainer; tree models still explain every row.
    * Matplotlib figure is explicitly closed after saving to free memory.
    * SHAP values array is deleted from memory as soon as the DataFrame is built.

    Args:
        classifier_path (str): Path to a trained time-agnostic classifier (.pkl)
        tissue (str): Tissue name (must match label file naming convention)
        motif_annotations_path (str): Optional CSV with motif annotations
        motif_annotations_sep (str): Separator for the annotations file
        windows (list[str]): Time windows for the time-agnostic dataset
        top_n_motifs (int): Number of motifs in the bar plot
        random_state (int): Random seed for reproducibility
        kernel_background_size (int): Number of k-means centres used as the
            KernelExplainer background (default 100)
        kernel_max_explain (int): Max rows explained by KernelExplainer;
            None = explain all (default 500)

    Returns:
        shap_df (pd.DataFrame): SHAP statistics for all motifs
    """

    # ── Load model ────────────────────────────────────────────────────────────
    with open(classifier_path, "rb") as f:
        model = pickle.load(f)

    model_names = make_names_dict()
    model_key   = type(model).__name__
    model_name  = model_names[model_key]["full"]
    model_short = model_names[model_key]["short"]

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, composite = compose_windows(tissue, windows)

    # ── Optional motif annotation ─────────────────────────────────────────────
    motif_ids   = X.columns          # fallback: raw codes
    motif_names = X.columns

    if motif_annotations_path:
        annot = pd.read_csv(motif_annotations_path, sep=motif_annotations_sep)

        num_mismatches = len(X.columns) - (X.columns == annot["id"]).sum()
        assert num_mismatches == 0, (
            f"Invalid annotations! {num_mismatches} mismatches in motif codes!"
        )

        motif_names = annot["name"]
        motif_ids   = annot["id"]
        X.columns   = annot["name"].astype(str) + "  -  (" + annot["id"].astype(str) + ")"

    # ── SHAP explainer ────────────────────────────────────────────────────────
    is_tree = model_key in ["RandomForestClassifier", "XGBClassifier"]

    if is_tree:
        # TreeExplainer is exact and memory-efficient already
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Normalise TreeExplainer output shape
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        X_shap = X  # keep reference for beeswarm

    else:
        # ── Memory-efficient KernelExplainer path (SVM / LR) ─────────────────
        #
        # 1. Background: k-means summary of X  →  O(k × p) instead of O(n × p)
        # 2. Foreground: random subsample      →  O(m × p), m << n
        #
        # Both are standard SHAP practices; neither biases the values.

        background = _get_kernel_background(X, kernel_background_size)

        X_shap = _sample_foreground(X, kernel_max_explain, random_state)

        def predict_proba_pos(data):
            return model.predict_proba(data)[:, 1]

        explainer   = shap.KernelExplainer(predict_proba_pos, background)
        shap_values = explainer.shap_values(X_shap, silent=True)

    # ── Beeswarm plot ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title(
        f"{model_name} beeswarm SHAP feature importance plot for tissue {tissue}"
    )
    plt.tight_layout()

    figures_path = f"results/figures/shap/{model_short}"
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(
        os.path.join(figures_path, f"{tissue}_beeswarm_{model_short}.pdf"),
        dpi=300, format="pdf"
    )
    plt.close(fig)   # ← free figure memory immediately

    # ── SHAP statistics DataFrame ─────────────────────────────────────────────
    mean_vals     = shap_values.mean(axis=0)
    std_vals      = shap_values.std(axis=0)
    abs_mean_vals = np.abs(mean_vals)
    mean_abs_vals = np.abs(shap_values).mean(axis=0)

    del shap_values   # ← free the largest array as soon as we're done with it

    shap_df = pd.DataFrame({
        "motif_id":           motif_ids,
        "motif_name":         motif_names,
        "mean_importance":    mean_vals,
        "abs_mean_importance": abs_mean_vals,
        "mean_abs_importance": mean_abs_vals,
        "std_importance":     std_vals,
    }).sort_values("abs_mean_importance", ascending=False)

    data_path = f"results/shap/{model_short}"
    os.makedirs(data_path, exist_ok=True)
    shap_df.to_csv(
        os.path.join(data_path, f"{tissue}_shap_table_{model_short}.csv"),
        index=False
    )

    print(f"  ✓  Done: {model_short} / {tissue}")
    return shap_df


# ── Parallel worker (top-level so it is picklable) ────────────────────────────

def _run_one(args_tuple):
    """Worker function for multiprocessing — one (model_short, tissue) job."""
    model_short, tissue, annot_path = args_tuple
    print(f"  → Starting {model_short} / {tissue}  (pid {os.getpid()})")
    try:
        shap_analysis_with_beeswarm(
            classifier_path=(
                f"results/models/time_agnostic/all_data/{model_short}_{tissue}.pkl"
            ),
            tissue=tissue,
            top_n_motifs=25,
            motif_annotations_path=annot_path,
            motif_annotations_sep="\t",
        )
    except Exception as exc:
        # Don't let one failure kill the whole pool
        print(f"  ✗  FAILED {model_short} / {tissue}: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SHAP analysis. Available models: RF, XGB, SVM, LR"
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=["RF", "XGB", "LR", "SVM"],
        default=["RF", "XGB", "LR", "SVM"],
        help="Model names (space-separated). Default: all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help=(
            "Number of parallel worker processes. "
            "Each worker holds one model + its data in memory. "
            "Default: 3."
        ),
    )
    args = parser.parse_args()

    tissues    = ["Neuroblasts", "Neurons", "Glia"]
    annot_path = "data/motif_names.tsv"

    # Build the full job list
    jobs = [
        (m, t, annot_path)
        for m in args.models
        for t in tissues
    ]

    print(
        f"Running {len(jobs)} jobs across {args.workers} worker(s)...\n"
        "(Each worker process is isolated — a crash in one won't affect others.)"
    )

    # spawn is safer than fork for mixed numpy/sklearn/shap workloads
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        pool.map(_run_one, jobs)

    print("\nAll jobs finished.")


if __name__ == "__main__":
    main()