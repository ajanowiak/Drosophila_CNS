# compute_permutation_importance.py
# alternative for mdi_analysis.py

# Tutaj importance to score(baseline) - score(permuted)
# default scorer for RandomForestClassifier is accuracy

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import compose_windows, print_timestamp
from sklearn.inspection import permutation_importance



def pemutation_analysis(
    classifier_path: str,
    tissue: str,
    n_repeats: int = 5,
    motif_annotations_path: str = None,
    motif_annotations_sep: str = None,
    windows: list[str] = ["06-08", "10-12", "14-16"],
    top_n_motifs: int = 20
):
    """
    Analyse permutation feature importance scores for a pretrained
    time-agnostic Random Forest classifier.

    This function:
    1. Loads a pretrained RF classifier
    2. Reconstructs the time-agnostic dataset
    3. Extracts feature importance scores
    4. Saves a bar plot of the top motifs
    5. Saves a full importance table

    Returns:
        importance_df (pd.DataFrame)
    """

    # Load model
    with open(classifier_path, "rb") as f:
        model = pickle.load(f)

    model_name = "RandomForest"

    # Load data
    X, y, composite = compose_windows(tissue, windows)

    motif_ids = X.columns
    motif_names = X.columns

    # Replace motif IDs with readable names if annotations are provided
    if motif_annotations_path:
        annot = pd.read_csv(
            motif_annotations_path,
            sep=motif_annotations_sep
        )

        num_mismatches = len(X.columns) - (X.columns == annot["id"]).sum()
        assert num_mismatches == 0, (
            f"Invalid annotations! "
            f"There are {num_mismatches} mismatches in motif codes."
        )

        motif_ids = annot["id"]
        motif_names = annot["name"]

    r = permutation_importance(model, X, y, n_repeats=n_repeats, n_jobs=-1)
    
    # Compute mean and std across trees
    mean_importance = r['importances_mean']
    std_importance = r['importances_std']

    # Create results dataframe
    importance_df = pd.DataFrame({
        "motif_id": motif_ids,
        "motif_name": motif_names,
        "mean_importance": mean_importance,
        "std_importance": std_importance
    }).sort_values("mean_importance", ascending=False)

    # Plot top motifs
    top_df = importance_df.head(top_n_motifs).copy()
    top_df = top_df.iloc[::-1]   # reverse for better top-to-bottom ordering

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_df["motif_name"],
        top_df["mean_importance"],
        xerr=top_df["std_importance"],
        color = "navy",
        alpha = 0.8,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1, "alpha": 0.7}
    )

    ax.set_xlabel("Mean feature importance across trees", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        f"Mean feature importance across trees in the Random Forest model\nTissue: {tissue}. Top {top_n_motifs} features",
        fontsize=13,
        pad=12
    )

    # cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # add subtle x-grid for readability
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # improve label readability
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()

    # Save figure
    figures_path = f"results/figures/RF_permutation_importance"
    os.makedirs(figures_path, exist_ok=True)

    fig_path = os.path.join(
        figures_path,
        f"{tissue}_permutation_importance.pdf"
    )
    plt.savefig(fig_path, dpi=300, format="pdf")

    # Save full table
    data_path = f"results/RF_permutation_importance"
    os.makedirs(data_path, exist_ok=True)

    df_path = os.path.join(
        data_path,
        f"{tissue}_permutation_importance_table.csv"
    )
    importance_df.to_csv(df_path, index=False)

    return importance_df


def main():
    tissues = ["Neuroblasts", "Neurons", "Glia"]
    annot_path = "data/motif_names.tsv"

    for tissue in tissues:
        print_timestamp(f"RF permutation importance analysis for tissue {tissue}...")

        _ = pemutation_analysis(
            classifier_path=f"results/models/time_agnostic/all_data/RF_{tissue}.pkl",
            tissue=tissue,
            top_n_motifs=35,
            motif_annotations_path=annot_path,
            motif_annotations_sep="\t"
        )
    print_timestamp("... done!")


if __name__ == "__main__":
    main()